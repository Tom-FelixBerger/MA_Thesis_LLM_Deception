import h5py
import numpy as np
import itertools
from utils import utils, templates

BATCH_SIZE = 10

def count_saved_attention_outputs(output_path):
    if not output_path.exists():
        return 0
    with h5py.File(output_path, "r") as f:
        return f["template_id"].shape[0]

def save_activation_batch(output_path, batch_metadata, attention_tensor):
    B = attention_tensor.shape[0]

    with h5py.File(output_path, "a") as f:
        first_time = ("attention" not in f)

        # --- Save metadata datasets ---
        for key, v in batch_metadata.items():
            v = np.asarray(v)

            if first_time:
                # must create dataset with shape, dtype first
                dset = f.create_dataset(
                    key,
                    shape=(B,),
                    maxshape=(None,),
                    dtype=v.dtype
                )
                dset[:] = v
            else:
                old = f[key].shape[0]
                f[key].resize((old + B,))
                f[key][old:old + B] = v

        # --- Save attention tensor ---
        if first_time:
            # attention_tensor has shape (B, L, H, D)
            att_shape = attention_tensor.shape
            max_shape = (None,) + att_shape[1:]

            dset = f.create_dataset(
                "attention",
                shape=att_shape,
                maxshape=max_shape,
                chunks=True,
                dtype=attention_tensor.dtype
            )
            dset[:] = attention_tensor

        else:
            old = f["attention"].shape[0]
            new_shape = (old + B,) + attention_tensor.shape[1:]

            f["attention"].resize(new_shape)
            f["attention"][old:old + B] = attention_tensor

def main():
    vignettes = utils.generate_belief_inference_vignettes(template_ids=templates.TRAIN_IDS+templates.TEST_IDS)

    for m, model_key in enumerate(utils.MODELS):
        if utils.MODELS[model_key]["excluded"]:
            continue
        
        to_process = vignettes.copy()
        output_path = utils.DATA_DIR / "Experiment2" / f"{model_key}_attention_outputs.h5"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        start_idx = count_saved_attention_outputs(output_path)
        if start_idx >= len(to_process):
            print(f"All vignettes already processed for {model_key}.")
            continue
    
        model, tokenizer = utils.load_model_and_tokenizer(model_key)
        batch_data = {
            "target_p": [],
            "target_c": [],
            "template_id": []
        }
        
        for idx in range(start_idx, len(to_process), BATCH_SIZE):
            print(f"Processing index {idx} to {min(idx+BATCH_SIZE, len(to_process))} of {len(to_process)} for model {model_key}.")

            batch = to_process[idx:idx+BATCH_SIZE]
            message_batch = [v["messages"] for v in batch]

            # (B, L, H, D)
            attention_tensor = utils.extract_batch_attention_outputs(
                message_batch, model, tokenizer, model_key
            )

            # metadata for this batch
            batch_metadata = {
                "target_p": [v["target_p"] for v in batch],
                "target_c": [v["target_c"] for v in batch],
                "template_id": [v["template_id"] for v in batch],
            }

            save_activation_batch(output_path, batch_metadata, attention_tensor)

if __name__ == "__main__":
    main()