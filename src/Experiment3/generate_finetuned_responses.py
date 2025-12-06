import ast
import json
from pathlib import Path
import torch
from utils import templates, utils
import gc

BATCH_SIZE = 10
MODEL_SAVE_DIR = utils.DATA_DIR / "Experiment3" / "model_saves"
OUTPUT_DIR = utils.DATA_DIR / "Experiment3"


def load_finetuned_model(model_key, mode):
    assert mode in ("tbi", "free")

    model_dir = MODEL_SAVE_DIR / f"{model_key}_{mode}_lora"
    adapter_dir = model_dir / "adapter"
    tokenizer_dir = model_dir / "tokenizer"

    base_model, base_tokenizer = utils.load_model_and_tokenizer(model_key)
    return utils.load_peft_variant(base_model, base_tokenizer, adapter_dir, tokenizer_dir)


def main():
    # Create vignettes
    vignettes = utils.generate_deception_vignettes(
        template_ids=templates.BASE_IDS, return_question=False
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_key in utils.MODELS:
        if utils.MODELS[model_key]["excluded"]:
            continue

        for mode in ("tbi", "free"):
            print(f"Evaluating {model_key}_{mode}")

            model, tokenizer = load_finetuned_model(model_key, mode)
            out_path = OUTPUT_DIR / f"{model_key}_{mode}_finetuned_baseline_responses.jsonl"

            print("Generating responses")
            utils.evaluate_deception(model, tokenizer, vignettes, BATCH_SIZE, out_path)

            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()