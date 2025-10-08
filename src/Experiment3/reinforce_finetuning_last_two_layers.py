import random
import joblib
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.tokenization_utils_base import BatchEncoding
from peft import prepare_model_for_kbit_training

import sys
sys.path.append(str(Path(__file__).parent.parent))
import utils

CLASSIFIER_PATH_P = Path(__file__).resolve().parents[2] / "model_saves" / "logreg_clf_targets_p.pkl"
CLASSIFIER_PATH_C = Path(__file__).resolve().parents[2] / "model_saves" / "logreg_clf_targets_c.pkl"

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "model_saves" / "mistral_reinforce_last_two_layers_ckpt"
MODEL_SAVE_DIR = OUTPUT_DIR / "model"
TOKENIZER_SAVE_DIR = OUTPUT_DIR / "tokenizer"
CHECKPOINT_FILE = OUTPUT_DIR / "optimizer_state.pt"

BATCH_SIZE = 10
NUM_UPDATES = 20
SEED = 42
TRAINABLE_LAYERS = 2


def save_everything(model, tokenizer, optimizer, update_idx=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(TOKENIZER_SAVE_DIR)

    ckpt = {
        "optim_state": optimizer.state_dict(),
        "update_idx": update_idx,
    }
    torch.save(ckpt, CHECKPOINT_FILE)
    print("Saved checkpoint and model weights.")


def compute_logprob_sequence(model, input_prompt_ids, generated_text, tokenizer):
    if isinstance(input_prompt_ids, BatchEncoding):
        input_prompt_ids = input_prompt_ids["input_ids"]

    if not isinstance(input_prompt_ids, torch.Tensor):
        input_prompt_ids = torch.as_tensor(input_prompt_ids, device=model.device)

    if input_prompt_ids.dim() == 1:
        input_prompt_ids = input_prompt_ids.unsqueeze(0)

    input_prompt_ids = input_prompt_ids.to(model.device, dtype=torch.long)
    gen_ids = tokenizer(
        generated_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=False,
    ).input_ids.to(model.device)
    if gen_ids.shape[1] == 0:
        return torch.tensor(0.0, device=model.device)
    full_ids = torch.cat([input_prompt_ids, gen_ids], dim=1).to(model.device)
    outputs = model(full_ids)
    logits = outputs.logits
    prefix_len = input_prompt_ids.shape[1]
    pred_logits = logits[:, prefix_len - 1:-1, :]
    logprobs = F.log_softmax(pred_logits, dim=-1)
    token_logps = logprobs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)
    return token_logps.sum()


def unfreeze_last_layers(model, num_layers=2):
    total_layers = model.config.num_hidden_layers
    first_trainable = max(0, total_layers - num_layers)

    for param in model.parameters():
        if param.dtype in (torch.float16, torch.float32, torch.bfloat16):
            param.requires_grad = False

    trainable_params = []
    for layer_idx in range(first_trainable, total_layers):
        layer = model.model.layers[layer_idx]
        layer.to(torch.float16)
        for param in layer.parameters():
            if param.dtype in (torch.float16, torch.float32, torch.bfloat16):
                param.requires_grad = True
                trainable_params.append(param)
    return trainable_params


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    heads_p = utils.top_heads(target='p')
    heads_c = utils.top_heads(target='c')
    assert len(heads_p) == 10 and len(heads_c) == 10, "Expected 10 top heads per probe."

    probe_p = joblib.load(CLASSIFIER_PATH_P)
    probe_c = joblib.load(CLASSIFIER_PATH_C)

    tokenizer = utils.load_tokenizer()
    model = utils.load_model()
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    trainable_params = unfreeze_last_layers(model, num_layers=TRAINABLE_LAYERS)
    optimizer = AdamW(trainable_params, lr=5e-5)

    start_update = 0
    if CHECKPOINT_FILE.exists():
        print(f"Loading checkpoint: {CHECKPOINT_FILE}")
        ckpt = torch.load(CHECKPOINT_FILE, map_location="cpu")
        if "optim_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optim_state"])
                print("Loaded optimizer state.")
            except Exception as exc:
                print(f"Warning: could not load optimizer state: {exc}")
        start_update = ckpt.get("update_idx", 0) + 1

    extractor = utils.MistralAttentionHeadExtractor()
    extractor_meta = utils.extractor_metadata(extractor)
    vignettes = utils.load_vignettes([utils.DATASET_NAMES['finetuning']])
    random.shuffle(vignettes)

    vignette_index = 0

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
            instruction = vignette["instruction"]
            response_a = vignette["response_a"]
            response_b = vignette["response_b"]
            for question in [vignette["question_1"], vignette["question_2"]]:
                prompt = scenario + instruction

                model.eval()
                with torch.no_grad():
                    full_response, only_new = utils.generate_text(model, tokenizer, prompt)

                classification = utils.classify_response(only_new, response_a, response_b)

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

            model.train()
            prompt_t = utils.tokenize_input(prompt, tokenizer).to(model.device)
            logprob_sum = compute_logprob_sequence(model, prompt_t, only_new, tokenizer)

            batch_rewards.append(reward)
            batch_logprobs.append(logprob_sum)

        losses = []
        optimizer.zero_grad()
        for reward_value, logprob_value in zip(batch_rewards, batch_logprobs):
            losses.append(-reward_value * logprob_value)

        if losses:
            loss_batch = torch.stack(losses).mean()
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            loss_value = loss_batch.item()
        else:
            loss_value = 0.0

        save_everything(model, tokenizer, optimizer, update_idx=update)
        print(f"=== Completed update {update + 1}/{NUM_UPDATES} | loss={loss_value:.4f} ===")

    print(f"Training finished. Final model saved to: {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    main()
