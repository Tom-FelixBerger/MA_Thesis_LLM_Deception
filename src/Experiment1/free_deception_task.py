from utils import utils, templates
import json

def main():
    free_dec_vignettes = utils.generate_free_deception_no_lying_vignettes(template_ids=[1,2,3,4,5]) # only baseline assessment templates
    len_vigns = len(free_dec_vignettes)

    for m, model_key in enumerate(utils.MODELS):
        model, tokenizer = utils.load_model_and_tokenizer(model_key)

        results_buffer = []
        output_path = utils.DATA_DIR / "Experiment1" / f"{model_key}_free_deception_responses.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8") # fresh file

        for i, vignette in enumerate(free_dec_vignettes):

            full_response, only_new = utils.generate_text(model, tokenizer, vignette['prompt'])
            results_buffer.append({
                "template_id": vignette["template_id"],
                "prompt_and_answer": full_response,
                "manual_annotation": "to be annotated manually"
            })

            if (i+1)%10==0:
                print(f"Just finished vignette {i+1} of {len_vigns} for model {m+1} of 4 ({model_key}).")
                with output_path.open("a", encoding="utf-8") as f:
                    for buffered_result in results_buffer:
                        f.write(json.dumps(buffered_result, ensure_ascii=False) + "\n")
                results_buffer = []
        
if __name__ == "__main__":
    main()
