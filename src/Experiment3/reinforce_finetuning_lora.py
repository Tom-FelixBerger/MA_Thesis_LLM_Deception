import random
import joblib
import numpy as np
from pathlib import Path
import json

import torch
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import sys
sys.path.append(str(Path(__file__).parent.parent))
import utils

CLASSIFIER_PATH_P = Path(__file__).resolve().parents[2] / "model_saves" / "logreg_clf_targets_p.pkl"
CLASSIFIER_PATH_C = Path(__file__).resolve().parents[2] / "model_saves" / "logreg_clf_targets_c.pkl"
FA_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "free_answers.jsonl"

BATCH_SIZE = 10
NUM_UPDATES = 20
SEED = 42
TARGET_LAYERS = list(range(16, 32))


def build_target_modules():
    return utils.build_target_modules(TARGET_LAYERS)


def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    heads_p = utils.top_heads(target='p')
    heads_c = utils.top_heads(target='c')
    assert len(heads_p) == 10 and len(heads_c) == 10, "Expected 10 top heads per probe."

    probe_p = joblib.load(CLASSIFIER_PATH_P)
    probe_c = joblib.load(CLASSIFIER_PATH_C)

    extractor = utils.MistralAttentionHeadExtractor()
    extractor_meta = utils.extractor_metadata(extractor)
    vignettes = utils.load_vignettes([utils.DATASET_NAMES['finetuning']])
    random.shuffle(vignettes)

    for condition in ["with_options", "free_answer"]:
        model_dir = Path(__file__).resolve().parents[2] / "model_saves" / f"mistral_reinforce_lora_ckpt_{condition}"
        model_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir, tokenizer_dir, checkpoint_file = utils.model_save_dirs(model_dir)

        tokenizer = utils.load_tokenizer()
        model = utils.load_model()
        model.config.use_cache = False

        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=build_target_modules(),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        print("Applying LoRA to layers 16-31 only...")
        model = get_peft_model(model, lora_config)

        utils.freeze_model_parameters(model)
        utils.enable_lora_training(model)

        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)

        start_update = 0
        if checkpoint_file.exists():
            print(f"Loading checkpoint: {checkpoint_file}")
            ckpt = torch.load(checkpoint_file, map_location="cpu")
            if "optim_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optim_state"])
                    print("Loaded optimizer state.")
                except Exception as exc:
                    print(f"Warning: could not load optimizer state: {exc}")
            start_update = ckpt.get("update_idx", 0) + 1

        vignette_index = 0
        
        free_answers = []
        for update in range(start_update, NUM_UPDATES):
            print(f"Update {update + 1} of {NUM_UPDATES}")

            if vignette_index + BATCH_SIZE > len(vignettes):
                random.shuffle(vignettes)
                vignette_index = 0
            batch = vignettes[vignette_index:vignette_index + BATCH_SIZE]
            vignette_index += BATCH_SIZE

            batch_rewards = []
            batch_logprobs = []

            for i, vignette in enumerate(batch):
                scenario = vignette["scenario"]
                response_a = vignette["response_a"]
                response_b = vignette["response_b"]
                for question in [vignette["question_1"], vignette["question_2"]]:
                    prompt_text = utils.build_prompt(
                        vignette,
                        instruction_key=f"instruction_{condition}",
                    )
                    prompt_messages = utils.build_chat_messages(
                        vignette,
                        instruction_key=f"instruction_{condition}",
                    )

                    model.eval()
                    with torch.no_grad():
                        full_response, only_new = utils.generate_text(
                            model,
                            tokenizer,
                            prompt_messages,
                        )

                    classification = utils.classify_response(only_new, response_a, response_b) if condition == "with_options" else "no_classification"

                    if classification == 'invalid' or len(only_new) == 0:
                        reward = 0.0
                    else:
                        probe_prompt = scenario + f'You respond: {only_new}\n' + question
                        head_acts = extractor.extract_head_activations(probe_prompt)
                        feat_p = utils.select_activation_subset(head_acts, heads_p, extractor_meta)
                        feat_c = utils.select_activation_subset(head_acts, heads_c, extractor_meta)
                        prob_p = probe_p.predict_proba(feat_p)[0, 1]
                        prob_c = probe_c.predict_proba(feat_c)[0, 1]
                        p_disagree = prob_p * (1.0 - prob_c) + (1.0 - prob_p) * prob_c
                        reward = float(1.0 - 2.0 * p_disagree)
                    print(
                        f"Processing Vignette {i + 1} of {BATCH_SIZE} | "
                        f"Vignette ID: {vignette['id']} | Classification: {classification} | Reward: {reward:.4f}"
                    )
                    if condition == "free_answer":
                        free_answers.append({
                            "vignette_id": vignette['id'],
                            "prompt": prompt_text,
                            "answer": only_new,
                            "reward": f"{reward:.4f}"
                        })

                    model.train()
                    prompt_t = utils.tokenize_input(prompt_messages, tokenizer).to(model.device)
                    logprob_sum = utils.compute_logprob_sequence(model, prompt_t, only_new, tokenizer)

                    batch_rewards.append(reward)
                    batch_logprobs.append(logprob_sum)


            losses = []
            optimizer.zero_grad()
            for reward_value, logprob_value in zip(batch_rewards, batch_logprobs):
                losses.append(-reward_value * logprob_value)

            if losses:
                loss_batch = torch.stack(losses).mean()
                loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_value = loss_batch.item()
            else:
                loss_value = 0.0

            utils.save_training_state(model_dir, model, tokenizer, optimizer, update_idx=update)
            print(f"=== Completed update {update + 1}/{NUM_UPDATES} | loss={loss_value:.4f} ===")

        if condition == "free_answer":
            with FA_OUTPUT_PATH.open('w', encoding='utf-8') as f:
                for fa in free_answers:
                    f.write(json.dumps(fa, ensure_ascii=False) + "\n")
        print(f"Training finished. Final adapters saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
