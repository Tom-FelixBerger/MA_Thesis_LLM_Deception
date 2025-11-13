import os
import math
import random
import re
import pickle
import torch
from torch import nn
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from utils import utils, templates

BATCH_SIZE = 10
NUM_STEPS = 200
GRAD_ACCUM_STEPS = 1

def reward_function(
    model,
    tokenizer,
    model_key: str,
    generated_texts,
    vignette_batch,
    probe_c,
    probe_p,
):
    """
    Compute a reward per generated answer using the C and P probes.

    For each (text, vignette):
      - If the text is not exactly response_a or response_b -> reward = -1.
      - Otherwise:
          * Build two belief-inference style prompts, one for each attribute.
          * Run them through the model to get attention activations.
          * Use the probes to estimate:
                pC = P(C=1), pP = P(P=1)
            and p_div = P(C != P) ≈ pC(1-pP) + (1-pC)pP.
          * Reward contribution is -p_div for each attribute,
            and we average across the attributes for that vignette.

    Returns
    -------
    rewards : torch.Tensor, shape [B], dtype=torch.float32 (on CPU).
    """
    rewards = []
    probe_messages = []   # all probe prompts for valid answers
    example_indices = []  # map each probe prompt back to original example index

    # 1) First pass: decide which examples are invalid vs valid, and
    #    build the probe prompts (belief inference vignettes) for valid ones.
    for i, (text, vignette) in enumerate(zip(generated_texts, vignette_batch)):
        cls = utils.classify_response(text, vignette["response_a"], vignette["response_b"])
        if cls == "invalid":
            # Fixed worst reward for malformed answers
            rewards.append(-1.0)
        else:
            # Placeholder for now, we'll fill in from probe results
            rewards.append(None)

            # For each attribute, create a belief-inference style message list:
            #   [ original user message,
            #     assistant: model answer,
            #     user: question with attribute filled in ]
            for question_attribute in vignette["attributes"]:
                repl_question = vignette["question"].replace(
                    "question_attribute", question_attribute
                )
                probe_message = vignette["messages"] + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": repl_question},
                ]
                probe_messages.append(probe_message)
                example_indices.append(i)

    # 2) If there were valid answers, run the probes.
    if probe_messages:
        # Extract attention outputs (Bv, L, H, D) as numpy
        att = utils.extract_batch_attention_outputs_pre_projection(
            probe_messages, model, tokenizer, model_key
        )  # shape: (Bv, L, H, D)

        Bv = att.shape[0]
        L, H, D = utils.get_model_dims(model_key)
        X = att.reshape(Bv, L * H * D)  # (Bv, L*H*D)

        # Unpack probe objects
        clf_c = probe_c["clf"]
        sc_c = probe_c["scaler"]
        slc_c = probe_c["slice"]

        clf_p = probe_p["clf"]
        sc_p = probe_p["scaler"]
        slc_p = probe_p["slice"]

        # Apply feature selection and scaling
        Xc = X[:, slc_c]
        Xp = X[:, slc_p]

        # Probabilities for C=1 and P=1
        pc1 = clf_c.predict_proba(sc_c.transform(Xc))[:, 1]  # (Bv,)
        pp1 = clf_p.predict_proba(sc_p.transform(Xp))[:, 1]  # (Bv,)

        # p_divergence = P(C != P) = pC(1-pP) + (1-pC)pP
        p_div = pc1 * (1.0 - pp1) + (1.0 - pc1) * pp1  # (Bv,)

        # 3) Aggregate per original example (each example has len(attributes) probes).
        sums = [0.0] * len(generated_texts)
        counts = [0] * len(generated_texts)

        for idx, ex_idx in enumerate(example_indices):
            sums[ex_idx] += float(p_div[idx])
            counts[ex_idx] += 1

        for ex_idx in range(len(generated_texts)):
            if rewards[ex_idx] is None:
                if counts[ex_idx] > 0:
                    avg_div = sums[ex_idx] / counts[ex_idx]
                    rewards[ex_idx] = -avg_div  # higher divergence -> more negative
                else:
                    # Shouldn't really happen, but be safe.
                    rewards[ex_idx] = 0.0

    # Convert to tensor (CPU; caller moves it to model.device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    return rewards_tensor


def select_lora_trainable_layers(model: nn.Module, start_layer: int, end_layer: int):
    """
    Freeze *all* params, then unfreeze ONLY LoRA params that belong to
    transformer block indices in [start_layer, end_layer).
    We match by parameter names like 'model.layers.{i}.*lora*' (LLaMA/Mistral)
    and also allow decoder layers 'transformer.layers.{i}.*' variants.

    Call this AFTER get_peft_model().
    """
    # Freeze everything first
    for n, p in model.named_parameters():
        p.requires_grad = False

    # Unfreeze only LoRA params inside the desired layer window
    def is_in_window(name):
        m = re.search(r"\.layers\.(\d+)\.", name)
        if not m:
            return False
        idx = int(m.group(1))
        return (idx >= start_layer) and (idx < end_layer)

    trainable_count = 0
    for n, p in model.named_parameters():
        if "lora_" in n and is_in_window(n):
            p.requires_grad = True
            trainable_count += p.numel()

    if trainable_count == 0:
        print("[WARN] No LoRA params selected as trainable for the given layer window.")


# ---------------------------
# Loss / logprob helpers
# ---------------------------
@torch.no_grad()
def sample_outputs(model, tokenizer, batch_messages, max_new_tokens=64, temperature=0.7, top_p=0.9):
    encoded = utils.tokenize_batch(batch_messages, tokenizer)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    gen_out = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Split into prompt and generated parts
    input_len = encoded["input_ids"].shape[1]
    gen_ids = gen_out[:, input_len:]  # [B, Tgen]
    full_inputs = gen_out  # [B, Tprompt+Tgen]

    # Decode only the new tokens per item
    texts = []
    for i in range(full_inputs.size(0)):
        only_new = tokenizer.decode(gen_ids[i], skip_special_tokens=True).strip()
        texts.append(only_new)

    # Attention mask for the full sequence
    full_attn = (full_inputs != tokenizer.pad_token_id).long()

    return full_inputs, full_attn, gen_ids, texts


def sequence_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, gen_ids: torch.Tensor):
    """
    Compute log-probs for the *generated* part of the full sequence.
    Args:
        input_ids: [B, Tfull] (prompt + generated)
        attention_mask: [B, Tfull]
        gen_ids: [B, Tgen] (the generated continuation, *already shifted* wrt the prompt)
    Returns:
        seq_logprob: [B] sum of per-token logprobs for generated tokens
        token_logprobs: [B, Tgen]
        entropy: [B] (optional, mean token entropy over generated positions)
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # [B, Tfull, V]
    # The probability of token at position t is conditioned on tokens < t,
    # so generated token at gen_pos corresponds to logits position at the same absolute index - 1
    B, Tfull, V = logits.shape
    Tgen = gen_ids.shape[1]

    # Absolute indices for generated tokens within the full sequence
    prompt_len = Tfull - Tgen
    # Logits to score generated tokens are at positions [prompt_len-1 ... Tfull-2]
    # Shift logits and ids to align: next-token prediction
    logits_for_gen = logits[:, prompt_len-1 : Tfull-1, :]  # [B, Tgen, V]
    # Gather logprobs of the actual generated tokens
    logprobs = torch.log_softmax(logits_for_gen, dim=-1)  # [B, Tgen, V]
    token_logprobs = logprobs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)  # [B, Tgen]

    # Entropy bonus (optional)
    probs = torch.softmax(logits_for_gen, dim=-1)
    entropy = -(probs * (logprobs)).sum(dim=-1).mean(dim=-1)  # [B]

    seq_logprob = token_logprobs.sum(dim=-1)  # [B]
    return seq_logprob, token_logprobs, entropy

# ---------------------------
# Main training entry
# ---------------------------
def main():
    vignettes = utils.generate_deception_incentive_vignettes(templates.TBI_FINETUNE_IDS, return_question=True)
    random.shuffle(vignettes)

    for model_key in utils.MODELS:
        if utils.MODELS[model_key]["excluded"]:
            continue
        
        model, tokenizer = utils.load_model_and_tokenizer(model_key)
        
        with open(utils.DATA_DIR/"Experiment2"/f"{model_key}_C_first_half_probe.pkl", "rb") as f:
            probe_c = pickle.load(f)

        with open(utils.DATA_DIR/"Experiment2"/f"{model_key}_P_first_half_probe.pkl", "rb") as f:
            probe_p = pickle.load(f)

        model.train()

        lora_targets = ["q_proj", "v_proj"]
        lora_cfg = LoraConfig(
            r= 16,
            lora_alpha= 32,
            lora_dropout= 0.5,
            bias="none",
            target_modules=lora_targets,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

        start_layer = utils.MODELS[model_key]["start_layer"]
        end_layer = utils.MODELS[model_key]["num_layers"]
        select_lora_trainable_layers(model, start_layer, end_layer)

        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"Trainable parameters (LoRA within layers {start_layer}-{end_layer}): {sum(p.numel() for p in trainable):,}")
        optimizer = AdamW(trainable, lr=5e-5, weight_decay=0.0)

        # Moving baseline for REINFORCE
        baseline_value = torch.tensor(0.0, dtype=torch.bfloat16, device=model.device)

        # 5) Training loop
        output_path = utils.DATA_DIR / "Experiment3" / f"{model_key}_TBI_lora_adapter"
        os.makedirs(output_path, exist_ok=True)
        step = 0
        optimizer.zero_grad()


        while step < NUM_STEPS:
            start_of_batch = (step*BATCH_SIZE)%len(vignettes)
            end_of_batch = start_of_batch+BATCH_SIZE
            if end_of_batch <= len(vignettes):
                vignette_batch = vignettes[start_of_batch:end_of_batch]
            else:
                vignette_batch = vignettes[start_of_batch:]
                end_of_batch = end_of_batch%len(vignettes)
                vignette_batch += vignettes[:end_of_batch]
            message_batch = [v["messages"] for v in vignette_batch]

            # (b) Generate outputs (no grad)
            full_inputs, full_attn, gen_ids, texts = sample_outputs(
                model, tokenizer, message_batch,
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.9,
            )

            rewards = reward_function(
                model,
                tokenizer,
                model_key,
                texts,
                vignette_batch,
                probe_c,
                probe_p,
            ).to(model.device)

            # Update moving baseline
            with torch.no_grad():
                baseline_value = 0.9 * baseline_value + 0.1 * rewards.mean()

            # (d) Compute log-probs and entropy for generated sequence (with grad)
            seq_logprob, token_logprobs, entropy = sequence_logprobs(
                model, full_inputs, full_attn, gen_ids
            )

            # (e) Policy gradient loss on FULL sequence
            # L = - E[(R - b) * logpi(y_{1:T})] - entropy_coef * H
            adv = (rewards - baseline_value).detach()
            pg_loss = -(adv * seq_logprob).mean()
            ent_bonus = entropy.mean()
            loss = pg_loss - 0.01 * ent_bonus

            (loss / GRAD_ACCUM_STEPS).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (step % 10) == 0:
                print(
                    f"step={step:05d} "
                    f"loss={loss.item():.4f} "
                    f"pg={pg_loss.item():.4f} "
                    f"ent={ent_bonus.item():.4f} "
                    f"R(mean)={rewards.mean().item():.4f} "
                    f"baseline={baseline_value.item():.4f}"
                )
                # Print one sample for quick sanity check
                print(f"sample: {texts[0][:200]!r}")

            # (f) Save adapter periodically
            if (step % 100) == 0 and step > 0:
                save_path = os.path.join(output_path, f"step_{step}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"[saved] {save_path}")

            step += 1

        # Final save
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Training complete. Adapter saved to: {output_path}")

if __name__ == "__main__":
    main()