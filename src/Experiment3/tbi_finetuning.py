import pickle
import random

import bitsandbytes as bnb
import numpy as np
import torch

from utils import templates, utils


BATCH_SIZE = 10
NUM_UPDATES = 20
MAX_NEW_TOKENS = 64
LEARNING_RATE = 1e-4
TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 42
MODEL_SAVE_DIR = utils.DATA_DIR / "Experiment3" / "model_saves"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_first_half_probe(model_key, target):
    probe_path = utils.DATA_DIR / "Experiment2" / f"{model_key}_{target}_first_half_probe.pkl"
    with open(probe_path, "rb") as f:
        probe = pickle.load(f)
    return probe


def prepare_probes(model_key):
    probe_c = load_first_half_probe(model_key, "C")
    probe_p = load_first_half_probe(model_key, "P")
    return probe_c, probe_p


def probe_probability(attention_tensor, probe):
    flat = attention_tensor.reshape(1, -1)
    features = flat[:, probe["slice"]]
    scaled = probe["scaler"].transform(features)
    probs = probe["clf"].predict_proba(scaled)[0, 1]
    return float(probs)


def compute_reward(attention_tensors, probe_c, probe_p):
    prob_c_0 = probe_probability(attention_tensors[0], probe_c)
    prob_p_0 = probe_probability(attention_tensors[0], probe_p)
    prob_c_1 = probe_probability(attention_tensors[1], probe_c)
    prob_p_1 = probe_probability(attention_tensors[1], probe_p)
    p_disagree_0 = prob_p_0 * (1.0 - prob_c_0) + (1.0 - prob_p_0) * prob_c_0
    p_disagree_1 = prob_p_1 * (1.0 - prob_c_1) + (1.0 - prob_p_1) * prob_c_1
    reward = 1.0 - p_disagree_0 - p_disagree_1
    return float(reward)


def build_belief_messages(vignette, response_text):
    question_0 = vignette["question"].replace("{question_attribute}", vignette["attributes"][0])
    question_1 = vignette["question"].replace("{question_attribute}", vignette["attributes"][1])
    return [
        [
            dict(role="user", content=vignette["messages"][0]["content"]),
            dict(role="assistant", content=response_text),
            dict(role="user", content=question),
        ] for question in [question_0, question_1]]



def run_tbi_finetuning(model_key, vignettes):
    model_dir = MODEL_SAVE_DIR / f"{model_key}_tbi_lora"
    adapter_dir, tokenizer_dir, checkpoint_path = utils.tbi_checkpoint_paths(model_dir)
    model, tokenizer = utils.prepare_tbi_lora_model(model_key, adapter_dir, tokenizer_dir)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)
    start_update = utils.load_tbi_optimizer_state(optimizer, checkpoint_path)
    probe_c, probe_p = prepare_probes(model_key)

    vignette_index = 0
    random.shuffle(vignettes)

    for update in range(start_update, NUM_UPDATES):
        if vignette_index + BATCH_SIZE > len(vignettes):
            random.shuffle(vignettes)
            vignette_index = 0

        batch = vignettes[vignette_index:vignette_index + BATCH_SIZE]
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

            classification = utils.classify_response(generated_text, item["response_a"], item["response_b"])

            if classification == "invalid" or len(generated_text) == 0:
                reward = 0.0
            else:
                belief_messages_list = build_belief_messages(item, generated_text)
                attention_tensors = [utils.extract_single_attention_outputs(
                    belief_messages,
                    model,
                    tokenizer,
                    model_key,
                ) for belief_messages in belief_messages_list]
                reward = compute_reward(attention_tensors, probe_c, probe_p)

            model.train()
            logprob = utils.compute_logprob_from_generated(
                model,
                prompt_ids,
                prompt_mask,
                generated_ids,
            )

            batch_rewards.append(reward)
            batch_logprobs.append(logprob)

        optimizer.zero_grad()
        losses = []

        for reward_value, logprob_value in zip(batch_rewards, batch_logprobs):
            losses.append(-reward_value * logprob_value)

        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
        else:
            loss_value = 0.0

        utils.save_tbi_state(
            model,
            tokenizer,
            optimizer,
            adapter_dir,
            tokenizer_dir,
            checkpoint_path,
            update,
        )

        print(f"[{model_key}] Completed update {update + 1}/{NUM_UPDATES} | loss={loss_value:.4f}")


def main():
    set_seed(SEED)
    template_ids = templates.TBI_FINETUNE_IDS
    vignettes = utils.generate_deception_incentive_vignettes(template_ids, return_question=True)

    for model_key in utils.MODELS:
        if utils.MODELS[model_key]["excluded"]:
            continue

        print(f"=== Running TBI finetuning for {model_key} ===")
        run_tbi_finetuning(model_key, vignettes.copy())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
