"""
Plots the translated sigmoid for several betas.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set(font_scale=1.5)

ALPHA = 6.9077542789816375

FIG_PATH = Path(__file__).parent.resolve()


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def softplus(beta: np.ndarray):
    return np.log(1 + np.exp(beta))


def translated_sigmoid(x: np.ndarray, beta: float):
    return sigmoid((x - beta * ALPHA) / beta)


def plot_translated_sigmoid(beta: float, ax: plt.Axes):
    x = np.linspace(0, 7, 200)
    y = translated_sigmoid(x, beta=beta)
    sns.lineplot(x=x, y=y, ax=ax, linewidth=4.5)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(3 * 7, 7))
    betas = [0.5, 0.1, 0.05]
    for beta, ax in zip(betas, axes):
        print(f"beta: {beta}, softplus(beta): {softplus(beta)}")
        plot_translated_sigmoid(beta, ax)
        ax.set_title(r"$\beta=" + f"{beta}$")

    axes[0].set_xlabel("minDist(z)")
    axes[0].set_ylabel(r"$\alpha(z;\beta)$")

    fig.tight_layout()
    fig.savefig(FIG_PATH / "multiple_betas.jpg", dpi=120, bbox_inches="tight")

    plt.show()
