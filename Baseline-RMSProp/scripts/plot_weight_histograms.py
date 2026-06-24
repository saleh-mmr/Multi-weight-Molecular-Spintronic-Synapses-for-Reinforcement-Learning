from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


plt.rcParams.update({"font.size": 14})

# ---------------------------------------------------------
# Change only these
# ---------------------------------------------------------
LAYER = "FC.4"
BINS = 50
CHECKPOINT_NAME = "cartpole_3.pth"

BASE_DIR = Path(__file__).resolve().parent
BASE_FOLDER = BASE_DIR


def load_state_dict(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state_dict = torch.load(path, map_location="cpu")

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    return state_dict


def load_tensor(state_dict, key):
    if key not in state_dict:
        raise KeyError(f"{key} not found. Available keys: {list(state_dict.keys())}")

    return state_dict[key].detach().cpu().numpy().flatten()


def plot_histogram(ax, values, title, xlabel):
    ax.hist(values, bins=BINS, edgecolor="black", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)


def main():
    folder = BASE_FOLDER.resolve()
    checkpoint_path = folder / CHECKPOINT_NAME

    weight_key = f"{LAYER}.weight"
    bias_key = f"{LAYER}.bias"

    print("Looking in:", folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    state_dict = load_state_dict(checkpoint_path)

    weight = load_tensor(state_dict, weight_key)
    bias = load_tensor(state_dict, bias_key)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes = np.atleast_1d(axes)

    plot_histogram(
        axes[0],
        weight,
        title=f"{weight_key} histogram",
        xlabel="Weight value",
    )

    plot_histogram(
        axes[1],
        bias,
        title=f"{bias_key} histogram",
        xlabel="Bias value",
    )

    plt.tight_layout()
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()