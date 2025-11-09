
def main() -> None:
    EXPERIMENT2_DIR.mkdir(parents=True, exist_ok=True)

    credentials = utils.load_credentials(CREDENTIALS_PATH)
    significant_models = utils.load_significant_models(
        SIGNIFICANT_MODELS_PATH, SIGNIFICANCE_REPORT_PATH
    )
    eligible_models = [model for model in TARGET_MODEL_KEYS if model in significant_models]

    if not eligible_models:
        print("No models exceeded the deception significance threshold. Nothing to do.")
        return

    ignored = sorted(set(significant_models) - set(eligible_models))
    if ignored:
        print(
            "Skipping models without Experiment 2 support: "
            + ", ".join(ignored)
        )

    vignettes = _load_vignettes()
    if not vignettes:
        print("No vignettes found for the requested datasets. Aborting.")
        return

    for model_key in eligible_models:
        print("\n" + "=" * 80)
        print(f"Extracting activations for model: {model_key}")
        print("=" * 80)
        _process_model(model_key, vignettes, credentials)


if __name__ == "__main__":
    main()