
import random
import warnings
import gc
import bitsandbytes as bnb
import numpy as np
import torch

from utils import templates, utils

BATCH_SIZE = 10
MAX_NEW_TOKENS = 32
LEARNING_RATE = 1e-4
TEMPERATURE = 0.7
TOP_P = 0.9

MODEL_SAVE_DIR = utils.DATA_DIR / "Experiment4" / "model_saves"

def run_super_deceiver_pretraining(model_key, vignettes):

    model_dir = MODEL_SAVE_DIR / f"{model_key}_superdec_lora"
    adapter_dir, tokenizer_dir, checkpoint_path = utils.finetuned_checkpoint_paths(
        model_dir
    )

    # Start from baseline model with a fresh LoRA adapter (second-half layers)
    model, tokenizer = utils.prepare_lora_model(model_key, adapter_dir, tokenizer_dir)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)
    start_update = utils.load_finetuned_optimizer_state(optimizer, checkpoint_path)

    vignette_index = 0
    random.shuffle(vignettes)

    num_updates = int(len(vignettes)/BATCH_SIZE)
    for update in range(start_update, num_updates):

        batch = vignettes[vignette_index : vignette_index + BATCH_SIZE]
        vignette_index += BATCH_SIZE

        batch_rewards = []
        batch_logprobs = []

        for item in batch:
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

            classification = utils.classify_response(
                generated_text, item["response_a"], item["response_b"]
            )

            if classification == "deceptive":
                reward = 1.0
            elif classification == "invalid":
                reward = -0.8
            else:
                reward = -1.0

            model.train()
            logprob = utils.compute_logprob_from_generated(
                model, prompt_ids, prompt_mask, generated_ids
            )

            batch_rewards.append(reward)
            batch_logprobs.append(logprob)

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

        utils.save_finetuned_state(
            model,
            tokenizer,
            optimizer,
            adapter_dir,
            tokenizer_dir,
            checkpoint_path,
            update,
        )

        print(
            f"[{model_key}][superdec] Completed update {update + 1}/{num_updates} | loss={loss_value:.4f}"
        )
    del model
    del tokenizer



def main():

    template_ids = templates.SUPER_DEC_IDS
    other_vignettes = utils.generate_deception_vignettes(template_ids=template_ids)
    self_vignettes = utils.generate_deception_vignettes(template_ids, mode="self")
    all_vignettes = other_vignettes + self_vignettes

    model_key = "gemma-2-9b" # only apply to gemma
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        print(f"=== Pretraining Super-Deceiver  ===")
        run_super_deceiver_pretraining(model_key, all_vignettes.copy())
        gc.collect()
        torch.cuda.empty_cache()

        # generate baseline assessment responses
        print(f"=== Baseline Assessment Response Generation Super-Deceiver  ===")
        variant = "superdec"
        outfile = utils.DATA_DIR / "Experiment4" / f"{model_key}_{variant}_baseline.jsonl"
        utils.baseline_assessment(model_key, variant, outfile, MODEL_SAVE_DIR)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()