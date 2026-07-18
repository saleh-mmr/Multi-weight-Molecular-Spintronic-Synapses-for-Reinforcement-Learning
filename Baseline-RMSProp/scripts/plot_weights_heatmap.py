
"""Plots heatmaps of a selected layer's weights and biases for baseline models."""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update({"font.size": 14})

# ---------------------------------------------------------
# Change only these
# ---------------------------------------------------------
LAYER = "FC.4"
CHECKPOINT_NAME = "cartpole_3.pth"

BASE_FOLDER = Path(__file__).resolve().parent


def load_state_dict(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    # Support both raw state_dict checkpoints and wrappers that store it under a key.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    return checkpoint


def get_tensor(state_dict, key):
    if key not in state_dict:
        raise KeyError(f"{key} not found. Available keys: {list(state_dict.keys())}")

    return state_dict[key].detach().cpu().numpy()


def plot_heatmaps(items, key, title_prefix, xlabel, ylabel, figsize):
    max_abs = max(np.max(np.abs(values)) for _, values, _, _ in items)

    fig, axes = plt.subplots(1, len(items), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (name, values, pole_length, pole_mass) in zip(axes, items):
        sns.heatmap(
            values,
            ax=ax,
            cmap="seismic",
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            xticklabels=5,
            yticklabels=5 if values.shape[0] > 1 else False,
            cbar=True,
        )

        ax.set_title(f"{title_prefix} {key} - {name}\nL={pole_length}, M={pole_mass}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    if "agg" not in plt.get_backend().lower():
        plt.show()


def main():
    folder = BASE_FOLDER.resolve()
    print("Looking in:", folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    checkpoint_path = folder / CHECKPOINT_NAME
    state_dict = load_state_dict(checkpoint_path)

    weight_key = f"{LAYER}.weight"
    bias_key = f"{LAYER}.bias"

    weight = get_tensor(state_dict, weight_key)
    bias = get_tensor(state_dict, bias_key).reshape(1, -1)

    weight_items = [(CHECKPOINT_NAME.replace(".pth", ""), weight, "N/A", "N/A")]
    bias_items = [(CHECKPOINT_NAME.replace(".pth", ""), bias, "N/A", "N/A")]

    plot_heatmaps(
        weight_items,
        key=weight_key,
        title_prefix="Weights",
        xlabel="Input neuron index",
        ylabel="Output neuron index",
        figsize=(8, 6),
    )

    plot_heatmaps(
        bias_items,
        key=bias_key,
        title_prefix="Biases",
        xlabel="Bias neuron index",
        ylabel="",
        figsize=(8, 4),
    )


if __name__ == "__main__":
    main()