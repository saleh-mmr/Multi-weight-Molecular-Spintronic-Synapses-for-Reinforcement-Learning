"""Computes 1D/2D autocorrelation summaries and correlation lengths for checkpoint tensors."""

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
RUN_FOLDER = "run_2026-06-13_07-38-17"
STEP = 48395
LAYER = "FC.2"

BASE_DIR = Path(__file__).resolve().parent
BASE_FOLDER = BASE_DIR / "three_problems"

N_BINS = 50


def load_state_dict(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def get_tensor(state_dict, key):
    if key not in state_dict:
        raise KeyError(f"Key {key} not found. Available keys: {list(state_dict.keys())}")
    return state_dict[key].detach().cpu().numpy()


def entropy_bits(values, n_bins=50):
    values = np.asarray(values).ravel()
    hist, _ = np.histogram(values, bins=n_bins, density=False)
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def normalized_entropy_bits(values, n_bins=50):
    h = entropy_bits(values, n_bins=n_bins)
    return float(h / np.log2(n_bins))


def autocorrelation_1d(x):
    x = np.asarray(x).ravel()
    x = x - np.mean(x)

    denom = np.sum(x ** 2)
    if denom == 0:
        return np.zeros(2 * len(x) - 1)

    ac = np.correlate(x, x, mode="full")
    ac = ac / denom
    return ac


def autocorrelation_2d(W):
    W = np.asarray(W)
    W = W - np.mean(W)

    denom = np.sum(W ** 2)
    if denom == 0:
        return np.zeros((2 * W.shape[0] - 1, 2 * W.shape[1] - 1))

    shape = (2 * W.shape[0] - 1, 2 * W.shape[1] - 1)

    F = np.fft.fft2(W, s=shape)
    ac = np.fft.ifft2(np.abs(F) ** 2).real
    ac = np.fft.fftshift(ac)

    ac = ac / denom
    return ac


def radial_average(ac):
    h, w = ac.shape
    cy, cx = h // 2, w // 2

    y, x = np.indices(ac.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_int = r.astype(int)

    radial_sum = np.bincount(r_int.ravel(), weights=ac.ravel())
    radial_count = np.bincount(r_int.ravel())

    radial_mean = radial_sum / np.maximum(radial_count, 1)
    radii = np.arange(len(radial_mean))

    return radii, radial_mean


def estimate_correlation_length(radii, radial_ac, threshold=np.exp(-1)):
    below = np.where(radial_ac <= threshold)[0]
    if len(below) == 0:
        return np.nan
    return float(radii[below[0]])


def plot_autocorrelation_2d(ac, title, save_path):
    plt.figure(figsize=(7, 6))
    sns.heatmap(ac, cmap="coolwarm", center=0)
    plt.title(title)
    plt.xlabel("Lag x")
    plt.ylabel("Lag y")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_radial_average(radii, radial_ac, title, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(radii, radial_ac, marker="o")
    plt.axhline(np.exp(-1), linestyle="--", label="1/e")
    plt.title(title)
    plt.xlabel("Radius / lag distance")
    plt.ylabel("Normalized autocorrelation")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    folder = (BASE_FOLDER / RUN_FOLDER).resolve()
    print("Looking in:", folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    log_path = folder / "details_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"details_log.csv not found: {log_path}")

    output_dir = folder / "autocorrelation"
    output_dir.mkdir(exist_ok=True)

    weight_key = f"{LAYER}.weight"
    bias_key = f"{LAYER}.bias"

    paths = [
        folder / f"MC1_{STEP}.pth",
        folder / f"MC2_{STEP}.pth",
        folder / f"MC3_{STEP}.pth",
    ]

    log = pd.read_csv(log_path).iloc[0]

    state_dicts = [load_state_dict(path) for path in paths]

    weights = [get_tensor(sd, weight_key) for sd in state_dicts]
    biases = [get_tensor(sd, bias_key) for sd in state_dicts]

    pole_lengths = [
        log.get("CP_pole_length_1", "N/A"),
        log.get("CP_pole_length_2", "N/A"),
        log.get("CP_pole_length_3", "N/A"),
    ]

    pole_masses = [
        log.get("CP_pole_mass_1", "N/A"),
        log.get("CP_pole_mass_2", "N/A"),
        log.get("CP_pole_mass_3", "N/A"),
    ]

    model_names = ["MC1", "MC2", "MC3"]

    summary_rows = []
    ac_rows = []
    radial_rows = []

    for model, weight, bias, pole_length, pole_mass in zip(
        model_names, weights, biases, pole_lengths, pole_masses
    ):
        tensors = {
            "weight": weight,
            "bias": bias,
        }

        for tensor_type, values in tensors.items():
            values = np.asarray(values)
            values_flat = values.ravel()

            summary_rows.append(
                {
                    "model": model,
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

            if values.ndim == 1:
                ac = autocorrelation_1d(values)

                center = len(ac) // 2
                lags = np.arange(-center, center + 1)

                for lag, ac_value in zip(lags, ac):
                    ac_rows.append(
                        {
                            "model": model,
                            "layer": LAYER,
                            "tensor_type": tensor_type,
                            "lag_x": int(lag),
                            "lag_y": np.nan,
                            "autocorrelation": float(ac_value),
                            "pole_length": pole_length,
                            "pole_mass": pole_mass,
                        }
                    )

                plt.figure(figsize=(7, 5))
                plt.plot(lags, ac, marker="o")
                plt.title(f"{model} {LAYER} {tensor_type} 1D autocorrelation")
                plt.xlabel("Lag")
                plt.ylabel("Normalized autocorrelation")
                plt.tight_layout()
                plt.savefig(output_dir / f"{model}_{tensor_type}_autocorrelation_1d.png", dpi=300)
                plt.close()

            elif values.ndim == 2:
                ac = autocorrelation_2d(values)

                h, w = ac.shape
                cy, cx = h // 2, w // 2

                for y in range(h):
                    for x in range(w):
                        ac_rows.append(
                            {
                                "model": model,
                                "layer": LAYER,
                                "tensor_type": tensor_type,
                                "lag_x": int(x - cx),
                                "lag_y": int(y - cy),
                                "autocorrelation": float(ac[y, x]),
                                "pole_length": pole_length,
                                "pole_mass": pole_mass,
                            }
                        )

                radii, radial_ac = radial_average(ac)
                corr_length = estimate_correlation_length(radii, radial_ac)

                summary_rows[-1]["correlation_length_1_over_e"] = corr_length
                summary_rows[-1]["central_peak"] = float(ac[cy, cx])

                for r, val in zip(radii, radial_ac):
                    radial_rows.append(
                        {
                            "model": model,
                            "layer": LAYER,
                            "tensor_type": tensor_type,
                            "radius": int(r),
                            "radial_autocorrelation": float(val),
                            "pole_length": pole_length,
                            "pole_mass": pole_mass,
                        }
                    )

                plot_autocorrelation_2d(
                    ac,
                    f"{model} {LAYER} {tensor_type} 2D autocorrelation",
                    output_dir / f"{model}_{tensor_type}_autocorrelation_2d.png",
                )

                plot_radial_average(
                    radii,
                    radial_ac,
                    f"{model} {LAYER} {tensor_type} radial autocorrelation",
                    output_dir / f"{model}_{tensor_type}_radial_autocorrelation.png",
                )

            else:
                print(f"Skipping {model} {tensor_type}: unsupported shape {values.shape}")

    summary_df = pd.DataFrame(summary_rows)
    ac_df = pd.DataFrame(ac_rows)
    radial_df = pd.DataFrame(radial_rows)

    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)
    ac_df.to_csv(output_dir / "autocorrelation_values.csv", index=False)
    radial_df.to_csv(output_dir / "radial_autocorrelation.csv", index=False)

    print("Saved results to:", output_dir)


if __name__ == "__main__":
    main()