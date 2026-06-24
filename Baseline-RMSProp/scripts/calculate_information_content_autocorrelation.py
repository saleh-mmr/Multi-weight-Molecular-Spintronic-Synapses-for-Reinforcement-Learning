from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update({"font.size": 14})

# ---------------------------------------------------------
# Change only these
# ---------------------------------------------------------
LAYER = "FC.2"
CHECKPOINT_NAME = "cartpole_1.pth"

# Number of bins used to estimate entropy / information content
N_BINS = 50

# Maximum autocorrelation lag to plot
MAX_LAG = 50

BASE_DIR = Path(__file__).resolve().parent
BASE_FOLDER = BASE_DIR
OUTPUT_FOLDER_NAME = "information_content_autocorrelation"


def load_state_dict(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    return checkpoint


def get_tensor(state_dict, key):
    if key not in state_dict:
        raise KeyError(f"{key} not found. Available keys: {list(state_dict.keys())}")

    return state_dict[key].detach().cpu().numpy()


def entropy_bits(values, n_bins=50):
    """
    Estimates information content as Shannon entropy in bits.

    The tensor values are discretized into bins first, because neural network
    weights are continuous numbers.
    """
    x = np.asarray(values).ravel()
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan

    counts, _ = np.histogram(x, bins=n_bins)
    probs = counts[counts > 0] / counts.sum()

    return float(-np.sum(probs * np.log2(probs)))


def normalized_entropy_bits(values, n_bins=50):
    """
    Entropy normalized to [0, 1].

    0 means all values fall into one bin.
    1 means values are close to uniformly spread across the used bins.
    """
    h = entropy_bits(values, n_bins=n_bins)

    if not np.isfinite(h) or n_bins <= 1:
        return np.nan

    return float(h / np.log2(n_bins))


def autocorrelation_1d(x, max_lag=50):
    """
    Normalized autocorrelation for a 1D sequence.

    Lag 0 is always 1 unless the sequence has zero variance.
    """
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]

    if x.size < 2:
        return np.full(max_lag + 1, np.nan)

    x = x - np.mean(x)
    variance = np.dot(x, x)

    if variance == 0:
        return np.full(max_lag + 1, np.nan)

    max_lag = min(max_lag, x.size - 1)
    ac = np.empty(max_lag + 1)

    for lag in range(max_lag + 1):
        ac[lag] = np.dot(x[:-lag or None], x[lag:]) / variance

    return ac


def autocorrelation_rows(values, max_lag=50):
    """
    Autocorrelation along each output neuron's incoming weights.

    For a weight matrix shaped [output_neurons, input_neurons], this measures
    whether neighboring input-neuron weights have similar values.
    """
    values = np.asarray(values)

    if values.ndim == 1:
        return autocorrelation_1d(values, max_lag=max_lag)

    row_acs = []
    for row in values:
        row_acs.append(autocorrelation_1d(row, max_lag=max_lag))

    min_len = min(len(ac) for ac in row_acs)
    row_acs = np.array([ac[:min_len] for ac in row_acs])

    return np.nanmean(row_acs, axis=0)


def autocorrelation_columns(values, max_lag=50):
    """
    Autocorrelation along columns / output-neuron direction.

    For a weight matrix shaped [output_neurons, input_neurons], this measures
    whether neighboring output-neuron weights have similar values.
    """
    values = np.asarray(values)

    if values.ndim == 1:
        return autocorrelation_1d(values, max_lag=max_lag)

    col_acs = []
    for col in values.T:
        col_acs.append(autocorrelation_1d(col, max_lag=max_lag))

    min_len = min(len(ac) for ac in col_acs)
    col_acs = np.array([ac[:min_len] for ac in col_acs])

    return np.nanmean(col_acs, axis=0)


def plot_entropy_bar(summary_df, tensor_type, output_path):
    df = summary_df[summary_df["tensor_type"] == tensor_type].copy()

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df,
        x="model",
        y="entropy_bits",
        hue="model",
        dodge=False,
        legend=False,
    )
    plt.title(f"Information content / Shannon entropy: {tensor_type}")
    plt.xlabel("Model")
    plt.ylabel("Entropy [bits]")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    if "agg" not in plt.get_backend().lower():
        plt.show()


def plot_autocorrelation(ac_df, tensor_type, direction, output_path):
    df = ac_df[
        (ac_df["tensor_type"] == tensor_type)
        & (ac_df["direction"] == direction)
    ].copy()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="lag", y="autocorrelation", hue="model", marker="o")
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Autocorrelation: {tensor_type}, {direction}")
    plt.xlabel("Lag")
    plt.ylabel("Normalized autocorrelation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    if "agg" not in plt.get_backend().lower():
        plt.show()


def main():
    folder = BASE_FOLDER.resolve()
    print("Looking in:", folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    output_dir = folder / OUTPUT_FOLDER_NAME
    output_dir.mkdir(exist_ok=True, parents=True)

    weight_key = f"{LAYER}.weight"
    bias_key = f"{LAYER}.bias"

    checkpoint_path = folder / CHECKPOINT_NAME
    state_dict = load_state_dict(checkpoint_path)

    weight = get_tensor(state_dict, weight_key)
    bias = get_tensor(state_dict, bias_key)

    model_name = CHECKPOINT_NAME.replace(".pth", "")
    pole_length = "N/A"
    pole_mass = "N/A"

    summary_rows = []
    ac_rows = []

    tensors = {
        "weight": weight,
        "bias": bias,
    }

    for tensor_type, values in tensors.items():
        values_flat = np.asarray(values).ravel()

        summary_rows.append(
            {
                "model": model_name,
                "layer": LAYER,
                "tensor_type": tensor_type,
                "shape": str(tuple(values.shape)),
                "n_parameters": int(values_flat.size),
                "mean": float(np.mean(values_flat)),
                "std": float(np.std(values_flat)),
                "min": float(np.min(values_flat)),
                "max": float(np.max(values_flat)),
                "entropy_bits": entropy_bits(values_flat, n_bins=N_BINS),
                "normalized_entropy": normalized_entropy_bits(values_flat, n_bins=N_BINS),
                "n_bins": N_BINS,
                "pole_length": pole_length,
                "pole_mass": pole_mass,
            }
        )

        # Autocorrelation of flattened tensor
        ac_flat = autocorrelation_1d(values_flat, max_lag=MAX_LAG)
        for lag, ac_value in enumerate(ac_flat):
            ac_rows.append(
                {
                    "model": model_name,
                    "layer": LAYER,
                    "tensor_type": tensor_type,
                    "direction": "flattened",
                    "lag": lag,
                    "autocorrelation": ac_value,
                    "pole_length": pole_length,
                    "pole_mass": pole_mass,
                }
            )

        # Extra directional autocorrelation for weight matrices
        if tensor_type == "weight" and np.asarray(values).ndim == 2:
            for direction, ac_values in {
                "input_direction_rows": autocorrelation_rows(values, max_lag=MAX_LAG),
                "output_direction_columns": autocorrelation_columns(values, max_lag=MAX_LAG),
            }.items():
                for lag, ac_value in enumerate(ac_values):
                    ac_rows.append(
                        {
                            "model": model_name,
                            "layer": LAYER,
                            "tensor_type": tensor_type,
                            "direction": direction,
                            "lag": lag,
                            "autocorrelation": ac_value,
                            "pole_length": pole_length,
                            "pole_mass": pole_mass,
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    ac_df = pd.DataFrame(ac_rows)

    checkpoint_stem = checkpoint_path.stem
    summary_csv = output_dir / f"{checkpoint_stem}_{LAYER}_information_content_summary.csv"
    ac_csv = output_dir / f"{checkpoint_stem}_{LAYER}_autocorrelation.csv"

    summary_df.to_csv(summary_csv, index=False)
    ac_df.to_csv(ac_csv, index=False)

    print("\nSaved summary CSV:")
    print(summary_csv)

    print("\nSaved autocorrelation CSV:")
    print(ac_csv)

    print("\nInformation content summary:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(summary_df.to_string(index=False))

    plot_entropy_bar(
        summary_df,
        tensor_type="weight",
        output_path=output_dir / f"{checkpoint_stem}_{LAYER}_weight_entropy.png",
    )

    plot_entropy_bar(
        summary_df,
        tensor_type="bias",
        output_path=output_dir / f"{checkpoint_stem}_{LAYER}_bias_entropy.png",
    )

    plot_autocorrelation(
        ac_df,
        tensor_type="weight",
        direction="flattened",
        output_path=output_dir / f"{checkpoint_stem}_{LAYER}_weight_autocorrelation_flattened.png",
    )

    plot_autocorrelation(
        ac_df,
        tensor_type="weight",
        direction="input_direction_rows",
        output_path=output_dir / f"{checkpoint_stem}_{LAYER}_weight_autocorrelation_input_direction.png",
    )

    plot_autocorrelation(
        ac_df,
        tensor_type="weight",
        direction="output_direction_columns",
        output_path=output_dir / f"{checkpoint_stem}_{LAYER}_weight_autocorrelation_output_direction.png",
    )

    plot_autocorrelation(
        ac_df,
        tensor_type="bias",
        direction="flattened",
        output_path=output_dir / f"{checkpoint_stem}_{LAYER}_bias_autocorrelation_flattened.png",
    )


if __name__ == "__main__":
    main()
