import random
import warnings
import torch
import torch.nn.functional as F
import gc

from utils import templates, utils

BATCH_SIZE = 10
MODEL_SAVE_DIR = utils.DATA_DIR / "Experiment4" / "model_saves"

def build_soo_message_pairs(template_ids):
    pairs = []

    for t_id in template_ids:
        templ = templates.TEMPLATES[t_id]

        other_protagonists = templ["other_referencing_protagonists"]
        self_verb = templ["self_verb"]

        for obj, (prot, verb), (attr_a, attr_b) in (
            (o, p, ab)
            for o in templ["objects"]
            for p in other_protagonists
            for ab in [
                (templ["attributes"][0], templ["attributes"][1]),
                (templ["attributes"][1], templ["attributes"][0]),
            ]
        ):
            other_fill = {
                "object": obj,
                "protagonist": prot,
                "verb": verb,
                "attribute_a": attr_a,
                "attribute_b": attr_b,
            }
            self_fill = {
                "object": obj,
                "protagonist": "you",
                "verb": self_verb,
                "attribute_a": attr_a,
                "attribute_b": attr_b,
            }

            other_scenario, _, _, _, other_instruction = utils.fill(
                template=templ,
                instruction=templates.INSTRUCTION_DECEPTION_INCENTIVE,
                fill_comb=other_fill,
            )

            self_scenario, _, _, _, self_instruction = utils.fill(
                template=templ,
                instruction=templates.INSTRUCTION_SELF_DECEPTION,
                fill_comb=self_fill,
            )

            messages_other = [
                {"role": "user", "content": other_scenario + other_instruction}
            ]
            messages_self = [
                {"role": "user", "content": self_scenario + self_instruction}
            ]

            pairs.append(
                {
                    "template_id": t_id,
                    "messages_self": messages_self,
                    "messages_other": messages_other,
                }
            )

    return pairs


def kl_between_message_lists(model, tokenizer, messages_self, messages_other):
    input_ids_self, attn_self = utils.chat_prompt_tensors(
        messages_self, tokenizer, model.device
    )
    input_ids_other, attn_other = utils.chat_prompt_tensors(
        messages_other, tokenizer, model.device
    )

    outputs_self = model(input_ids=input_ids_self, attention_mask=attn_self)
    outputs_other = model(input_ids=input_ids_other, attention_mask=attn_other)

    log_probs_self = F.log_softmax(outputs_self.logits[:, -1, :], dim=-1)
    log_probs_other = F.log_softmax(outputs_other.logits[:, -1, :], dim=-1)

    probs_other = log_probs_other.exp().detach()

    kl = F.kl_div(log_probs_self, probs_other, reduction="batchmean")

    return kl

def run_soo_finetuning(
    model_key,
    vignette_pairs,
    superdec,
):
    if superdec:
        (
            model,
            tokenizer,
            optimizer,
            adapter_dir,
            tokenizer_dir,
            checkpoint_path,
            start_update,
        ) = utils.prepare_superdec_model(
            model_key,
            mode="soo"
        )
        tag = "superdec_soo"
    else:
        (
            model,
            tokenizer,
            optimizer,
            adapter_dir,
            tokenizer_dir,
            checkpoint_path,
            start_update,
        ) = utils.prepare_standard_soo_model(model_key)
        tag = "standard_soo"

    pair_index = 0
    random.shuffle(vignette_pairs)

    num_updates = int(len(vignette_pairs)/BATCH_SIZE)
    for update in range(start_update, num_updates):

        batch = vignette_pairs[pair_index : pair_index + BATCH_SIZE]
        pair_index += BATCH_SIZE

        losses = []

        for item in batch:
            loss = kl_between_message_lists(
                model, tokenizer, item["messages_self"], item["messages_other"]
            )
            losses.append(loss)

        if not losses:
            loss_value = 0.0
        else:
            optimizer.zero_grad()
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_value = float(batch_loss.detach().cpu().item())

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
            f"[{model_key}][{tag}] Completed update {update + 1}/{num_updates} | loss={loss_value:.4f}"
        )
    del model
    del tokenizer



def main():

    # Templates reserved for TBI vs SOO comparison
    template_ids = templates.TBI_VS_SOO_IDS
    soo_pairs = build_soo_message_pairs(template_ids)

    model_key = "gemma-2-9b" # tbi finetuning not successful for llama

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for variant, s in [("standard_soo", False), ("superdec_soo", True)]:
            # Super-deceiver + SOO finetuning
            print(f"=== Running {variant} finetuning ===")
            run_soo_finetuning(model_key, soo_pairs.copy(), superdec=s)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # generate baseline assessment responses
            print(f"=== Baseline Assessment Response Generation for {variant} ===")
            outfile = utils.DATA_DIR / "Experiment4" / f"{model_key}_{variant}_baseline.jsonl"
            utils.baseline_assessment(model_key, variant, outfile, MODEL_SAVE_DIR)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()