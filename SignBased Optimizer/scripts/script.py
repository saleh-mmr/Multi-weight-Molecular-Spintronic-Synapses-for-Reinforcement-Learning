"""Main script to train or test the sign-based optimizer CartPole agent."""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib import pyplot as plt
import random
import numpy as np
import torch
from learning.trainer import Trainer

def plot_per_episode(values, label, color, ylabel):
    """Plot a single training metric with the same visual configuration."""
    plt.figure(figsize=(10, 6))
    plt.plot(values, label=label, color=color, linewidth=4, alpha=0.9)
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_per_step(values, label, color, ylabel):
    """Plot a single training metric with the same visual configuration."""
    plt.figure(figsize=(10, 6))
    plt.plot(values, label=label, color=color, linewidth=4, alpha=0.9)
    plt.xlabel("Step", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


controller = {
    1: "manhattan_linear",
    2: "manhattan_logarithmic",
    3: "sgd",
    4: "rmsprop"
}

problem = {
    1: "CartPole",
    2: "MountainCar",
}

hyperparams = {
    "discount_factor": 0.99,
    "network_size": 20,
    "batch_size": 100,
    "max_episodes": 2000,
    "max_steps": 100,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.0002,
    "memory_capacity": 100000,
    "problem": 1,
    "controller": 1,
}

train_mode = True


if __name__ == "__main__":

    if train_mode:
        # Keep one explicit seed for reproducibility of this run.
        seed = 49
        print(f"Training with seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        trainer = Trainer(hyperparams, seed)
        rewards, loss, epsilon = trainer.train()

        # Plot all training metrics with the same configuration.
        plot_per_episode(rewards, "reward", "blue", "Reward")
        plot_per_step(loss, "loss", "orange", "Loss")
        plot_per_step(epsilon, "epsilon", "green", "Epsilon")

    else:
        trainer = Trainer(hyperparams, seed=None)
        trainer.test("2H-100.pth")