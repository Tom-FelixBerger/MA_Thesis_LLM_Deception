import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import sys
sys.path.append(str(Path(__file__).parent.parent))
import utils


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MODEL_SAVE_DIR = BASE_DIR / "model_saves"
CREDENTIALS_PATH = BASE_DIR / "credentials.txt"
SIGNIFICANT_MODELS_PATH = DATA_DIR / "experiment1" / "significant_models.json"
SIGNIFICANCE_REPORT_PATH = DATA_DIR / "experiment1" / "significance_tests.txt"

BATCH_SIZE = 10
NUM_UPDATES = 10
SEED = 42
TARGET_LAYERS = list(range(16, 32))


def build_target_modules():
    return utils.build_target_modules(TARGET_LAYERS)


def load_significant_model_keys() -> List[str]:
    return utils.load_significant_models(
        SIGNIFICANT_MODELS_PATH,
        fallback_report_path=SIGNIFICANCE_REPORT_PATH,
    )


def prepare_model_and_tokenizer(model_key: str, credentials: Dict[str, str]):
    model, tokenizer = utils.load_model_and_tokenizer(model_key, credentials)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    return model, tokenizer


def run_pretraining_for_model(model_key: str, credentials: Dict[str, str], vignettes: List[dict]):
    config = utils.get_model_config(model_key)
    supports_system_message = bool(config.get("system_message", True))

    model_dir = MODEL_SAVE_DIR / f"{model_key}_reinforce_lora_ckpt_pretrained"
    model_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir, tokenizer_dir, checkpoint_file = utils.model_save_dirs(model_dir)

    model, tokenizer = prepare_model_and_tokenizer(model_key, credentials)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=build_target_modules(),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"[{model_key}] Applying LoRA to layers 16-31 only...")
    model = get_peft_model(model, lora_config)

    utils.freeze_model_parameters(model)
    utils.enable_lora_training(model)

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)

    start_update = 0
    if checkpoint_file.exists():
        print(f"[{model_key}] Loading checkpoint: {checkpoint_file}")
        ckpt = torch.load(checkpoint_file, map_location="cpu")
        if "optim_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optim_state"])
                print("Loaded optimizer state.")
            except Exception as exc:
                print(f"Warning: could not load optimizer state: {exc}")
        start_update = ckpt.get("update_idx", 0) + 1

    vignette_index = 0

    for update in range(start_update, NUM_UPDATES):
        print(f"[{model_key}] Update {update + 1} of {NUM_UPDATES}")

        if vignette_index + BATCH_SIZE > len(vignettes):
            random.shuffle(vignettes)
            vignette_index = 0
        batch = vignettes[vignette_index:vignette_index + BATCH_SIZE]
        vignette_index += BATCH_SIZE

        batch_rewards = []
        batch_logprobs = []

        for i, vignette in enumerate(batch):
            for condition in ["self", "other"]:
                response_a = vignette["response_a"]
                response_b = vignette["response_b"]
                prompt_messages = utils.build_chat_messages(
                    vignette,
                    instruction_key=f"instruction_{condition}",
                    scenario_key=f"scenario_{condition}",
                    supports_system_message=supports_system_message,
                )

                model.eval()
                with torch.no_grad():
                    _, only_new = utils.generate_text(
                        model,
                        tokenizer,
                        prompt_messages,
                    )

                classification = utils.classify_response(only_new, response_a, response_b)

                if classification == 'invalid' or len(only_new) == 0:
                    reward = 0.0
                elif classification == 'deceptive':
                    reward = 1.0
                else:
                    reward = -1.0
                print(
                    f"[{model_key}] Vignette {i + 1} of {BATCH_SIZE} | "
                    f"Condition: {condition} | ID: {vignette['id']} | Classification: {classification} | Reward: {reward:.4f}"
                )

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
        print(f"[{model_key}] Completed update {update + 1}/{NUM_UPDATES} | loss={loss_value:.4f}")

    print(f"[{model_key}] Training finished. Final adapters saved to: {adapter_dir}")

    del model
    torch.cuda.empty_cache()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    credentials = utils.load_credentials(CREDENTIALS_PATH)
    significant_models = load_significant_model_keys()
    if not significant_models:
        raise RuntimeError(
            "No significant models found. Run Experiment1/classification_results.py first to "
            "identify the models for Experiment 4 pretraining."
        )

    dataset_name = utils.DATASET_NAMES['e4_superdeceiver']
    vignettes = utils.load_vignettes([dataset_name])
    if not vignettes:
        raise RuntimeError("No pretraining vignettes available for Experiment 4.")

    for model_key in significant_models:
        print(f"=== Running Experiment 4 pretraining for model '{model_key}' ===")
        model_vignettes = list(vignettes)
        random.shuffle(model_vignettes)
        run_pretraining_for_model(model_key, credentials, model_vignettes)


if __name__ == "__main__":
    main()
