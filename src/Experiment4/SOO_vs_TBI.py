### NOTIZ AN MICH SELBST: PROBLEM IST, DASS DAS PRETRAINED MODEL NIEMALS HONEST ANTWORTET UND DAHER NICHT IN DIESE RICHTUNG TRAINIERT WIRD (?)


import random
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
import utils


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MODEL_SAVE_DIR = BASE_DIR / "model_saves"
CREDENTIALS_PATH = BASE_DIR / "credentials.txt"
SIGNIFICANT_MODELS_PATH = DATA_DIR / "experiment1" / "significant_models.json"
SIGNIFICANCE_REPORT_PATH = DATA_DIR / "experiment1" / "significance_tests.txt"
CLASSIFIER_PATH_P = MODEL_SAVE_DIR / "logreg_clf_targets_p.pkl"
CLASSIFIER_PATH_C = MODEL_SAVE_DIR / "logreg_clf_targets_c.pkl"

BATCH_SIZE = 10
NUM_UPDATES = 25
SEED = 42


def model_dir(model_key: str, suffix: str) -> Path:
    return MODEL_SAVE_DIR / f"{model_key}_{suffix}"


def get_pretrained_assets(model_key: str):
    pretrained_dir = model_dir(model_key, "reinforce_lora_ckpt_pretrained")
    adapter_dir, tokenizer_dir, _ = utils.model_save_dirs(pretrained_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            "Pretrained adapters not found. Run Experiment4/pretraining.py before finetuning."
        )
    return adapter_dir, tokenizer_dir


def load_tokenizer_for_training(
    model_key: str,
    credentials: Dict[str, str],
    primary_dir: Path,
    pretrained_dir: Path,
):
    config = utils.get_model_config(model_key)
    token = credentials.get("HUGGINGFACE_TOKEN")
    if primary_dir.exists():
        return utils.load_tokenizer(path=str(primary_dir))
    if pretrained_dir.exists():
        return utils.load_tokenizer(path=str(pretrained_dir))
    return utils.load_tokenizer(path=str(config["model_id"]), token=token)


def prepare_tbi_model(
    model_key: str,
    credentials: Dict[str, str],
    adapter_dir_out: Path,
    pretrained_adapter_dir: Path,
):
    base_model = utils.load_model_for_key(model_key, credentials)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()

    adapter_source = adapter_dir_out if adapter_dir_out.exists() else pretrained_adapter_dir
    model = PeftModel.from_pretrained(base_model, str(adapter_source), is_trainable=True)

    utils.freeze_model_parameters(model)
    utils.enable_lora_training(model)
    return model


def prepare_soo_model(
    model_key: str,
    credentials: Dict[str, str],
    adapter_dir_out: Path,
    pretrained_adapter_dir: Path,
):
    base_model = utils.load_model_for_key(model_key, credentials)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()

    total_layers = list(range(base_model.config.num_hidden_layers))
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=utils.build_target_modules(total_layers),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    utils.freeze_model_parameters(model)
    utils.enable_lora_training(model)

    state_source = adapter_dir_out if adapter_dir_out.exists() else pretrained_adapter_dir
    state_dict = utils.load_peft_state_dict(state_source)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing {len(missing)} LoRA weights when loading pretrained state.")
    if unexpected:
        print(f"Warning: encountered {len(unexpected)} unexpected weights when loading pretrained state.")
    return model


def train_tbi(
    model_key: str,
    credentials: Dict[str, str],
    vignettes: List[dict],
    pretrained_adapter_dir: Path,
    pretrained_tokenizer_dir: Path,
    extractor,
    extractor_meta,
):
    heads_p = utils.top_heads(target='p')
    heads_c = utils.top_heads(target='c')
    assert len(heads_p) == 10 and len(heads_c) == 10, "Expected 10 top heads per probe."

    probe_p = joblib.load(CLASSIFIER_PATH_P)
    probe_c = joblib.load(CLASSIFIER_PATH_C)

    config = utils.get_model_config(model_key)
    supports_system_message = bool(config.get("system_message", True))

    model_dir_path = model_dir(model_key, "reinforce_lora_ckpt_tbi")
    model_dir_path.mkdir(parents=True, exist_ok=True)
    adapter_dir_out, tokenizer_dir_out, checkpoint_file = utils.model_save_dirs(model_dir_path)

    tokenizer = load_tokenizer_for_training(
        model_key,
        credentials,
        tokenizer_dir_out,
        pretrained_tokenizer_dir,
    )
    model = prepare_tbi_model(model_key, credentials, adapter_dir_out, pretrained_adapter_dir)

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)

    start_update = 0
    if checkpoint_file.exists():
        print(f"[{model_key} | TBI] Loading checkpoint: {checkpoint_file}")
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
        print(f"[{model_key} | TBI] Update {update + 1} of {NUM_UPDATES}")

        if vignette_index + BATCH_SIZE > len(vignettes):
            random.shuffle(vignettes)
            vignette_index = 0
        batch = vignettes[vignette_index:vignette_index + BATCH_SIZE]
        current_batch_size = len(batch)
        vignette_index += BATCH_SIZE

        batch_rewards = []
        batch_logprobs = []

        for i, vignette in enumerate(batch):
            scenario = vignette["scenario"]
            response_a = vignette["response_a"]
            response_b = vignette["response_b"]

            for question in [vignette["question_1"], vignette["question_2"]]:
                prompt_messages = utils.build_chat_messages(
                    vignette,
                    supports_system_message=supports_system_message,
                )

                model.eval()
                with torch.no_grad():
                    _, only_new = utils.generate_text(model, tokenizer, prompt_messages)

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
                    f"[{model_key} | TBI] Vignette {i + 1}/{current_batch_size} | ID: {vignette['id']} | "
                    f"Classification: {classification} | Reward: {reward:.4f}"
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

        utils.save_training_state(model_dir_path, model, tokenizer, optimizer, update_idx=update)
        print(f"[{model_key} | TBI] Completed update {update + 1}/{NUM_UPDATES} | loss={loss_value:.4f}")

    print(f"[{model_key} | TBI] Finetuning finished. Final adapters saved to: {adapter_dir_out}")

    del model
    torch.cuda.empty_cache()


def kl_divergence_between_prompts(model, tokenizer, prompt_self, prompt_other):
    inputs_self = utils.tokenize_input(prompt_self, tokenizer).to(model.device)
    inputs_other = utils.tokenize_input(prompt_other, tokenizer).to(model.device)

    outputs_self = model(**inputs_self)
    outputs_other = model(**inputs_other)

    log_probs_self = F.log_softmax(outputs_self.logits[:, -1, :], dim=-1)
    log_probs_other = F.log_softmax(outputs_other.logits[:, -1, :], dim=-1)

    probs_self = log_probs_self.exp().detach()
    probs_other = log_probs_other.exp().detach()

    kl_other_to_self = F.kl_div(log_probs_other, probs_self, reduction='batchmean')
    kl_self_to_other = F.kl_div(log_probs_self, probs_other, reduction='batchmean')
    return 0.5 * (kl_other_to_self + kl_self_to_other)


def train_soo(
    model_key: str,
    credentials: Dict[str, str],
    vignettes: List[dict],
    pretrained_adapter_dir: Path,
    pretrained_tokenizer_dir: Path,
):
    model_dir_path = model_dir(model_key, "reinforce_lora_ckpt_soo")
    model_dir_path.mkdir(parents=True, exist_ok=True)
    adapter_dir_out, tokenizer_dir_out, checkpoint_file = utils.model_save_dirs(model_dir_path)

    tokenizer = load_tokenizer_for_training(
        model_key,
        credentials,
        tokenizer_dir_out,
        pretrained_tokenizer_dir,
    )
    model = prepare_soo_model(model_key, credentials, adapter_dir_out, pretrained_adapter_dir)

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)

    start_update = 0
    if checkpoint_file.exists():
        print(f"[{model_key} | SOO] Loading checkpoint: {checkpoint_file}")
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
        print(f"[{model_key} | SOO] Update {update + 1} of {NUM_UPDATES}")

        if vignette_index + BATCH_SIZE > len(vignettes):
            random.shuffle(vignettes)
            vignette_index = 0
        batch = vignettes[vignette_index:vignette_index + BATCH_SIZE]
        current_batch_size = len(batch)
        vignette_index += BATCH_SIZE

        optimizer.zero_grad()
        batch_losses = []

        for i, vignette in enumerate(batch):
            prompt_self = vignette["scenario_self"] + vignette["instruction_self"]
            prompt_other = vignette["scenario_other"] + vignette["instruction_other"]

            model.train()
            loss = kl_divergence_between_prompts(model, tokenizer, prompt_self, prompt_other)
            batch_losses.append(loss)

            print(
                f"[{model_key} | SOO] Vignette {i + 1}/{current_batch_size} | ID: {vignette['id']} | "
                f"Loss: {loss.item():.4f}"
            )

        if batch_losses:
            total_loss = torch.stack(batch_losses).mean()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_value = total_loss.item()
        else:
            loss_value = 0.0

        utils.save_training_state(model_dir_path, model, tokenizer, optimizer, update_idx=update)
        print(f"[{model_key} | SOO] Completed update {update + 1}/{NUM_UPDATES} | loss={loss_value:.4f}")

    print(f"[{model_key} | SOO] Finetuning finished. Final adapters saved to: {adapter_dir_out}")

    del model
    torch.cuda.empty_cache()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    credentials = utils.load_credentials(CREDENTIALS_PATH)
    significant_models = utils.load_significant_models(
        SIGNIFICANT_MODELS_PATH,
        fallback_report_path=SIGNIFICANCE_REPORT_PATH,
    )
    if not significant_models:
        raise RuntimeError(
            "No significant models found. Run Experiment1/classification_results.py first to "
            "identify the models for Experiment 4 finetuning."
        )

    vignettes_tbi = utils.load_vignettes([utils.DATASET_NAMES['e4_finetuning']])
    vignettes_soo = utils.load_vignettes([utils.DATASET_NAMES['e4_superdeceiver']])

    if not vignettes_tbi:
        raise RuntimeError("No finetuning vignettes found for TBI training.")
    if not vignettes_soo:
        raise RuntimeError("No finetuning vignettes found for SOO training.")

    for model_key in significant_models:
        print(f"=== Running Experiment 4 finetuning for model '{model_key}' ===")
        pretrained_adapter_dir, pretrained_tokenizer_dir = get_pretrained_assets(model_key)

        extractor = utils.ModelAttentionHeadExtractor(model_key, credentials)
        extractor_meta = utils.extractor_metadata(extractor)

        model_vignettes_tbi = list(vignettes_tbi)
        model_vignettes_soo = list(vignettes_soo)
        random.shuffle(model_vignettes_tbi)
        random.shuffle(model_vignettes_soo)

        train_tbi(
            model_key,
            credentials,
            model_vignettes_tbi,
            pretrained_adapter_dir,
            pretrained_tokenizer_dir,
            extractor,
            extractor_meta,
        )
        train_soo(
            model_key,
            credentials,
            model_vignettes_soo,
            pretrained_adapter_dir,
            pretrained_tokenizer_dir,
        )

        del extractor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
