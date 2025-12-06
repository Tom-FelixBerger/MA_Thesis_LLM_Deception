import pickle
import random
import warnings
import bitsandbytes as bnb
import numpy as np
import torch
import gc

from utils import templates, utils


BATCH_SIZE = 10
MAX_NEW_TOKENS = 32
LEARNING_RATE = 1e-4
TEMPERATURE = 0.7
TOP_P = 0.9
MODEL_SAVE_DIR = utils.DATA_DIR / "Experiment3" / "model_saves"

def run_finetuning(model_key, vignettes, mode="tbi"):

    suffix = "tbi_lora" if mode == "tbi" else "free_lora"
    model_dir = MODEL_SAVE_DIR / f"{model_key}_{suffix}"

    adapter_dir, tokenizer_dir, checkpoint_path = utils.finetuned_checkpoint_paths(model_dir)
    model, tokenizer = utils.prepare_lora_model(model_key, adapter_dir, tokenizer_dir)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)
    start_update = utils.load_finetuned_optimizer_state(optimizer, checkpoint_path)

    probe_c, probe_p = utils.prepare_probes(model_key)

    vignette_index = 0
    random.shuffle(vignettes)

    num_updates = int(len(vignettes)/BATCH_SIZE)
    for update in range(start_update, num_updates):

        batch = vignettes[vignette_index : vignette_index + BATCH_SIZE]
        vignette_index += BATCH_SIZE

        batch_rewards = []
        batch_logprobs = []

        for item in batch:
            # Generate model output
            model.eval()
            with torch.no_grad():
                generation = utils.generate_chat_completion(
                    model,
                    tokenizer,
                    item["messages"],
                    MAX_NEW_TOKENS,
                    TEMPERATURE,
                    TOP_P,
                )

            generated_text = generation["text"]
            generated_ids = generation["generated_ids"].to(model.device)
            prompt_ids = generation["prompt_ids"].to(model.device)
            prompt_mask = generation["prompt_mask"].to(model.device)

            # --- MODE-SPECIFIC REWARD CONDITION ---
            reward = None
            if mode == "tbi":
                classification = utils.classify_response(
                    generated_text, item["response_a"], item["response_b"]
                )
                if classification == "invalid" or len(generated_text) == 0:
                    reward = -0.8

            elif mode == "free":
                eos_id = tokenizer.eos_token_id
                contains_eos = (generated_ids == eos_id).any().item()
                if not contains_eos:
                    reward = -0.8

            # compute reward only if not already penalized
            if reward is None:
                belief_messages_list = utils.build_belief_messages_for_finetuning(item, generated_text)
                attention_tensors = [
                    utils.extract_single_attention_outputs(
                        belief_messages, model, tokenizer, model_key
                    )
                    for belief_messages in belief_messages_list
                ]
                reward = utils.compute_tbi_reward(attention_tensors, probe_c, probe_p)

            # compute logprob
            model.train()
            logprob = utils.compute_logprob_from_generated(
                model, prompt_ids, prompt_mask, generated_ids
            )

            batch_rewards.append(reward)
            batch_logprobs.append(logprob)

        # optimization step
        optimizer.zero_grad()
        losses = [-r * lp for r, lp in zip(batch_rewards, batch_logprobs)]

        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
        else:
            loss_value = 0.0

        # save
        utils.save_finetuned_state(
            model,
            tokenizer,
            optimizer,
            adapter_dir,
            tokenizer_dir,
            checkpoint_path,
            update,
        )

        print(f"[{model_key}][{mode}] Completed update {update + 1}/{num_updates} | loss={loss_value:.4f}")
    del model
    del tokenizer


def main():
    template_ids = templates.TBI_FINETUNE_IDS
    tbi_vignettes = utils.generate_deception_vignettes(template_ids, return_question=True)
    free_vignettes = utils.generate_deception_vignettes(template_ids, return_question=True, mode="free")

    for model_key in utils.MODELS:
        if utils.MODELS[model_key]["excluded"]:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            print(f"=== Running TBI finetuning for {model_key} ===")
            run_finetuning(model_key, tbi_vignettes.copy(), mode="tbi")
            gc.collect()
            torch.cuda.empty_cache()

            print(f"=== Running free answer finetuning for {model_key} ===")
            run_finetuning(model_key, free_vignettes.copy(), mode="free")
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
