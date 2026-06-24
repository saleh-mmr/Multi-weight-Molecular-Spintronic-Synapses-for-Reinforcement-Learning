import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
from pathlib import Path
import pandas as pd
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
from learning.trainer import Trainer


def load_hyperparams(folder):
    row = pd.read_csv(folder / "details_log.csv").iloc[0]
    return {
        "discount_factor": float(row["discount_factor"]),
        "batch_size": int(row["batch_size"]),
        "warmup_size": int(row["warmup_size"]),
        "network_size": int(row["network_size"]),
        "max_steps_per_episode": int(row["max_steps_per_episode"]),
        "max_episodes": int(row["max_episodes"]),
        "epsilon_max": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": float(row["epsilon_decay"]),
        "memory_capacity": int(row["memory_size"]),

        "g_ap": float(row["G_ap_coefficient"]),
        "g_p": float(row["G_p_coefficient"]),
        "shift_parameter": float(row["shift parameter"]),
        "g_bias": float(row["G_bias_coefficient"]),
        "regularization_C": 0.00000001,
        "noise_stddev": float(row["noise_stddev"]),

        "CP_pole_length_1": float(row["CP_pole_length_1"]),
        "CP_pole_mass_1": float(row["CP_pole_mass_1"]),
        "CP_pole_length_2": float(row["CP_pole_length_2"]),
        "CP_pole_mass_2": float(row["CP_pole_mass_2"]),
        "CP_pole_length_3": float(row["CP_pole_length_3"]),
        "CP_pole_mass_3": float(row["CP_pole_mass_3"]),
    }



if __name__ == "__main__":
    folder_name = "run_2026-06-05_08-05-50"
    cartpole_selector = 2
    weight_selector = 0
    weigh_step = 377060
    num_tests = 5000
    folder = SCRIPT_DIR / "three_problems" / folder_name
    hyperparams = load_hyperparams(folder)
    keyword = ["MC1", "MC2", "MC3"][weight_selector]
    path = f"{folder}/{keyword}_{weigh_step}.pth"
    trainer = Trainer(hyperparams, seed=None, folder=folder)
    test_log = trainer.test(model_path=path, num_tests=num_tests, cartpole=cartpole_selector)
    print(
        f"mean={test_log['reward'].mean():.6f} | "
        f"std={test_log['reward'].std(ddof=0):.6f}"
    )