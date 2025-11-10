
from utils import utils



def main():
    vignettes = utils.generate_belief_inference_vignettes(template_ids=[6,7,8,9,10,11,12,13])

    ##### CONTINUE HERE: Set up activation extraction #####

    for m, model_key in enumerate(utils.MODELS):
        if utils.MODELS[model_key]["excluded"]:
            continue

        print("\n" + "=" * 80)
        print(f"Extracting activations for model: {model_key}")
        print("=" * 80)
        process_model(model_key, vignettes)


if __name__ == "__main__":
    main()