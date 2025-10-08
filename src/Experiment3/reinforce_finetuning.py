import os
import json
import random
import joblib
import numpy as np
import re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import utils


CLASSIFIER_PATH_P = "..\\..\\model_saves\\logreg_clf_targets_p.pkl"
CLASSIFIER_PATH_C = "..\\..\\model_saves\\logreg_clf_targets_c.pkl"

OUTPUT_DIR = "..\\..\\model_saves\\mistral_reinforce_ckpt"

CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "rl_checkpoint.pt")
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "model_saved")

BATCH_SIZE = 10
NUM_UPDATES = 20
SEED = 42

def save_everything(rl_model, tokenizer, optimizer, update_idx=None):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    rl_model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    ckpt = {
        "optim_state": optimizer.state_dict(),
        "update_idx": update_idx
    }
    torch.save(ckpt, CHECKPOINT_FILE)
    print("Saved checkpoint and adapters.")

def compute_logprob_sequence(model, input_prompt_ids, generated_text, tokenizer):
    gen_ids = tokenizer(
        generated_text, 
        return_tensors="pt", 
        add_special_tokens=False,
        truncation=False
    ).input_ids.to(model.device)
    if gen_ids.shape[1] == 0:
        return torch.tensor(0.0, device=model.device)
    full_ids = torch.cat([input_prompt_ids, gen_ids], dim=1).to(model.device)
    outputs = model(full_ids)
    logits = outputs.logits
    L = input_prompt_ids.shape[1]
    pred_logits = logits[:, L-1:-1, :]
    logprobs = F.log_softmax(pred_logits, dim=-1)
    token_logps = logprobs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)
    return token_logps.sum()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    heads_p = utils.top_heads(target='p')
    heads_c = utils.top_heads(target='c')
    assert len(heads_p) == 10 and len(heads_c) == 10, "Expected 10 top heads per probe."

    probe_p = joblib.load(CLASSIFIER_PATH_P)
    probe_c = joblib.load(CLASSIFIER_PATH_C)

    tokenizer = utils.load_tokenizer()
    rl_model = utils.load_model()

    # Prepare model for k-bit training
    rl_model = prepare_model_for_kbit_training(rl_model)
    rl_model.gradient_checkpointing_enable()

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj","w1","w2","w3"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("Wrapping full model with LoRA...")
    rl_model = get_peft_model(rl_model, lora_config)

    # Freeze everything
    for _, p in rl_model.named_parameters():
        if p.dtype in (torch.float16, torch.float32, torch.bfloat16):
            p.requires_grad = False

    # Enable LoRA adapter weights and optionally top layers
    for name, p in rl_model.named_parameters():
        if p.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            continue  # skip quantized tensors

        if "lora_" in name:
            p.requires_grad = True
        else:
            # optional: unfreeze base weights in top layers (if desired)
            m = re.search(r"layers\.(\d+)\.", name)
            if m:
                layer_idx = int(m.group(1))
                if layer_idx >= 16:
                    p.requires_grad = True

    print("Creating 8-bit Adam optimizer...")
    optimizer = bnb.optim.AdamW8bit(rl_model.parameters(), lr=1e-4)

    # Checkpoint loading
    if os.path.exists(CHECKPOINT_FILE):
        print("Loading checkpoint:", CHECKPOINT_FILE)
        ckpt = torch.load(CHECKPOINT_FILE, map_location="cpu")
        if "optim_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optim_state"])
                print("Loaded optimizer state.")
            except Exception as e:
                print("Warning: could not load optimizer state:", e)
        start_update = ckpt.get("update_idx", 0) + 1
    else:
        start_update = 0

    extractor = utils.MistralAttentionHeadExtractor()
    extractor_meta = utils.extractor_metadata(extractor)
    vignettes = utils.load_vignettes(datasets=utils.DATASET_NAMES['finetuning'])

    v_idx_iter = 0

    for update in range(start_update, NUM_UPDATES):
        if v_idx_iter + BATCH_SIZE > len(vignettes):
            random.shuffle(vignettes)
            v_idx_iter = 0
        batch = vignettes[v_idx_iter: v_idx_iter + BATCH_SIZE]
        v_idx_iter += BATCH_SIZE

        batch_rewards = []
        batch_logprobs = []

        for vignette in batch:
            scenario = vignette["scenario"]
            instruction = vignette["instruction"]
            response_a = vignette["response_a"]
            response_b = vignette["response_b"]
            for question in [vignette["question_1"], vignette["question_2"]]:

                prompt = scenario + instruction

                rl_model.eval()
                with torch.no_grad():
                    full_response, only_new = utils.generate_text(rl_model, tokenizer, prompt)

                classification = utils.classify_response(only_new, response_a, response_b)

                if classification == 'invalid' or len(only_new) == 0:
                    reward = 0.0
                    print(f"Vignette ID: {vignette['id']}\nPrompt: {prompt}\nResponse: {only_new} | Classification: {classification} | Reward: {reward:.4f}")
                else:
                    probe_prompt = scenario + f'You respond: {only_new}\n' + question
                    try:
                        head_acts = extractor.extract_head_activations(probe_prompt)
                        feat_p = utils.select_activation_subset(head_acts, heads_p, extractor_meta)
                        feat_c = utils.select_activation_subset(head_acts, heads_c, extractor_meta)
                        prob_p = probe_p.predict_proba(feat_p)[0, 1]
                        prob_c = probe_c.predict_proba(feat_c)[0, 1]
                        p_disagree = prob_p * (1.0 - prob_c) + (1.0 - prob_p) * prob_c
                        reward = float(1.0 - p_disagree)
                        print(f"Vignette ID: {vignette['id']}\nPrompt: {prompt}\nResponse: {only_new} | Classification: {classification} | Reward: {reward:.4f}")
                    except Exception as e:
                        print(f"Error extracting activations: {e}")
                        reward = 0.0

            rl_model.train()
            prompt_t = utils.tokenize_input(prompt, tokenizer).to(rl_model.device)
            logprob_sum = compute_logprob_sequence(rl_model, prompt_t, only_new, tokenizer)

            batch_rewards.append(reward)
            batch_logprobs.append(logprob_sum)

        losses = []
        optimizer.zero_grad()
        for r, lp in zip(batch_rewards, batch_logprobs):
            losses.append(-r * lp)

        if len(losses) > 0:
            loss_batch = torch.stack(losses).mean()
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(rl_model.parameters(), 1.0)
            optimizer.step()
            loss_value = loss_batch.item()
        else:
            loss_value = 0.0

        save_everything(rl_model, tokenizer, optimizer, update_idx=update)
        print(f"=== Completed update {update+1}/{NUM_UPDATES} | loss={loss_value:.4f} ===")

    print("Training finished. Final adapters saved to:", MODEL_SAVE_DIR)

if __name__ == "__main__":
    main()