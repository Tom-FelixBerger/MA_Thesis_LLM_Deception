
import random
import warnings
import numpy as np
import torch
import gc
from utils import templates, utils

BATCH_SIZE = 10
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.7
TOP_P = 0.9

MODEL_SAVE_DIR = utils.DATA_DIR / "Experiment4" / "model_saves"

def run_tbi_finetuning_from_superdec(model_key, vignettes):
    model, tokenizer, optimizer, adapter_dir, tokenizer_dir, checkpoint_path, start_update = (
        utils.prepare_superdec_model(model_key, mode="tbi")
    )

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

            # First, TBI validity / structural penalty
            classification = utils.classify_response(
                generated_text, item["response_a"], item["response_b"]
            )
            if classification == "invalid" or len(generated_text) == 0:
                reward = -0.8
            else:
                # Build belief-inference prompts and extract activations
                belief_messages_list = utils.build_belief_messages_for_finetuning(item, generated_text)
                attention_tensors = [
                    utils.extract_single_attention_outputs(
                        belief_messages, model, tokenizer, model_key
                    )
                    for belief_messages in belief_messages_list
                ]
                reward = utils.compute_tbi_reward(attention_tensors, probe_c, probe_p)

            # Logprob of generated sequence
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
            f"[{model_key}][superdec_tbi] Completed update {update + 1}/{num_updates} | loss={loss_value:.4f}"
        )
    del model
    del tokenizer


def main():

    # Use templates reserved for the TBI vs SOO comparison
    template_ids = templates.TBI_VS_SOO_IDS
    tbi_vignettes = utils.generate_deception_vignettes(
        template_ids, return_question=True
    )

    model_key = "gemma-2-9b" # tbi not successful on llama

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        print(f"=== Running TBI finetuning (from super-deceiver) ===")
        run_tbi_finetuning_from_superdec(model_key, tbi_vignettes.copy())
        gc.collect()
        torch.cuda.empty_cache()

        
        # generate baseline assessment responses
        print(f"=== Baseline Assessment Response Generation for TBI finetuning (from super-deceiver) ===")
        variant = "superdec_tbi"
        outfile = utils.DATA_DIR / "Experiment4" / f"{model_key}_{variant}_baseline.jsonl"
        utils.baseline_assessment(model_key, variant, outfile, MODEL_SAVE_DIR)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()