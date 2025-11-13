import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import utils


def plot_heatmap(model_key):
    path = utils.DATA_DIR / "Experiment2" / f"{model_key}_heatmap_data.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)

    heat_c, heat_p = data["heat_c"], data["heat_p"]

    L, H = heat_c.shape

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im0 = ax[0].imshow(heat_c, aspect="auto", vmin=0.45, vmax=1)
    ax[0].set_title(f"{model_key} – Head Accuracy (C)")
    ax[0].set_xlabel("Head")
    ax[0].set_ylabel("Layer")
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(heat_p, aspect="auto", vmin=0.45, vmax=1)
    ax[1].set_title(f"{model_key} – Head Accuracy (P)")
    ax[1].set_xlabel("Head")
    ax[1].set_ylabel("Layer")
    fig.colorbar(im1, ax=ax[1])

    plt.tight_layout()
    plt.savefig(utils.PLOTS_DIR / f"{model_key}_head_heatmaps.png")
    plt.close()


if __name__ == "__main__":
    for model_key in utils.MODELS:
        if not utils.MODELS[model_key]["excluded"]:
            plot_heatmap(model_key)