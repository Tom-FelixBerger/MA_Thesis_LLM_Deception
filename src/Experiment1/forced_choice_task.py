from utils import utils, templates
import json
import os

BATCH_SIZE = 10

def main():
    dec_inc_vignettes = utils.generate_deception_incentive_vignettes(template_ids=range(30))
    len_vigns = len(dec_inc_vignettes)
    
    for m, model_key in enumerate(utils.MODELS):
        to_process = dec_inc_vignettes.copy()
        results_buffer = []
        output_path = utils.DATA_DIR / "Experiment1" / f"{model_key}_forced_choice_responses.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
                to_process = to_process[num_lines:]
            if len(to_process) == 0:
                print(f"All vignettes already processed for model {model_key}. Skipping.")
                continue
        else:
            output_path.write_text("", encoding="utf-8")

        model, tokenizer = utils.load_model_and_tokenizer(model_key)


        for start in range(0, len(to_process), BATCH_SIZE):
            batch = to_process[start:start+BATCH_SIZE]
            message_batch = [v["messages"] for v in batch]

            generated = utils.batch_generate_text(model, tokenizer, message_batch)

            for vignette, (full, only_new) in zip(batch, generated):
                classification = utils.classify_response(
                    only_new,
                    vignette["response_a"],
                    vignette["response_b"]
                )

                results_buffer.append({
                    "template_id": vignette["template_id"],
                    "prompt_and_answer": full,
                    "classification": classification,
                })

            with output_path.open("a", encoding="utf-8") as f:
                for r in results_buffer:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            results_buffer = []

            print(f"Processed {min(start+BATCH_SIZE, len(to_process))} / {len(to_process)} remaining vignettes for model {m+1} of {len(utils.MODELS)} ({model_key}).")

if __name__ == "__main__":
    main()
