
from utils import utils



def main():
    vignettes = load_vignettes()

    for m, model_key in enumerate(utils.MODELS):
        if utils.MODELS[model_key]["excluded"]:
            continue

        print("\n" + "=" * 80)
        print(f"Extracting activations for model: {model_key}")
        print("=" * 80)
        process_model(model_key, vignettes)


if __name__ == "__main__":
    main()