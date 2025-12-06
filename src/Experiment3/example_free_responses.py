import pickle
import random
import warnings
import bitsandbytes as bnb
import numpy as np
import torch
import gc
import json
from utils import templates, utils


MAX_NEW_TOKENS = 32
TEMPERATURE = 0.7
TOP_P = 0.9
RESULTS_DIR = utils.DATA_DIR / "Experiment3" / "free_examples"

def run_example_finetuning(model_key, vignettes):

    results_path = RESULTS_DIR / f"{model_key}.jsonl"

    model, tokenizer = utils.prepare_lora_model(model_key)
    probe_c, probe_p = utils.prepare_probes(model_key)

    examples = random.sample(vignettes, 20)


    results = []
    for ex in examples:
        # Generate model output
        model.eval()
        with torch.no_grad():
            generation = utils.generate_chat_completion(
                model,
                tokenizer,
                ex["messages"],
                MAX_NEW_TOKENS,
                TEMPERATURE,
                TOP_P,
            )

        generated_text = generation["text"]
        generated_ids = generation["generated_ids"].to(model.device)
        prompt_ids = generation["prompt_ids"].to(model.device)
        prompt_mask = generation["prompt_mask"].to(model.device)

        reward = None

        eos_id = tokenizer.eos_token_id
        contains_eos = (generated_ids == eos_id).any().item()
        if not contains_eos:
            reward = -0.8

        # compute reward only if not already penalized
        if reward is None:
            belief_messages_list = utils.build_belief_messages_for_finetuning(ex, generated_text)
            attention_tensors = [
                utils.extract_single_attention_outputs(
                    belief_messages, model, tokenizer, model_key
                )
                for belief_messages in belief_messages_list
            ]
            reward, classifications = utils.compute_tbi_reward(attention_tensors, probe_c, probe_p, return_classifications=True)

        results.append({
            "message": ex["messages"],
            "answer": generated_text,
            "classifications": classifications,
            "reward": reward

        })

    del model
    del tokenizer
    with results_path.open("w", encoding="utf-8") as f:
        for res in results:        
            f.write(json.dumps(res, ensure_ascii=False) + "\n")


def main():
    template_ids = templates.TBI_FINETUNE_IDS
    free_vignettes = utils.generate_deception_vignettes(template_ids, return_question=True, mode="free")

    for model_key in utils.MODELS:
        if utils.MODELS[model_key]["excluded"]:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            print(f"=== Running free answer example extraction for {model_key} ===")
            run_example_finetuning(model_key, free_vignettes.copy())
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
