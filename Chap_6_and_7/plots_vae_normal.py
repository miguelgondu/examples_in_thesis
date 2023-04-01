"""
Loads up the network and plots stuff.
"""
from pathlib import Path
from itertools import product

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import seaborn as sns

from vae import VAENormal, load_data
from vae_geometric import VAENormalGeometry

ROOT_DIR = Path(__file__).parent.resolve()

sns.set_style("whitegrid")
sns.set(font_scale=1.5)

BETAS = [0.5, 0.1, 0.05]
PADDING_PERCENTAGE = 0.45


def plot_images(z: np.ndarray, images: np.ndarray, ax: plt.Axes):
    """
    A function that plots all images in {images}
    at coordinates {z}.

    Some black magic for better scatterplots.

    Taken and adapted from:
    https://stackoverflow.com/questions/59373626/matplotlib-scatter-different-images-mnist-instead-of-plots-for-tsne
    """
    for zi, img in zip(z, images):
        im = OffsetImage(img, zoom=0.5)
        ab = AnnotationBbox(im, zi, xycoords="data", frameon=False)
        ax.add_artist(ab)
        ax.update_datalim([zi])
        ax.autoscale()


def latent_space_plot_w_decoded_uncertainties():
    encodings = vae.encode(training_data).mean
    fig_latent, ax = plt.subplots(1, 1, figsize=(7, 7))
    viz_limits_x, viz_limits_y = vae.viz_limits(
        padding_percentage=PADDING_PERCENTAGE, center=(-0.6, 0.8)
    )

    axis_x = torch.linspace(*viz_limits_x, 75)
    axis_y = torch.linspace(*viz_limits_y, 75)
    fine_grid_in_latent_space = torch.Tensor(
        [[x, y] for x, y in product(axis_x, axis_y)]
    )
    stds = vae.decode(fine_grid_in_latent_space).scale.mean(dim=(1, 2))
    std_map = {
        (x.item(), y.item()): std
        for ((x, y), std) in zip(product(axis_x, axis_y), stds)
    }

    positions = {
        (x.item(), y.item()): (i, j)
        for j, x in enumerate(axis_x)
        for i, y in enumerate(reversed(axis_y))
    }
    stds_organized = np.zeros((75, 75))
    for z, std in std_map.items():
        i, j = positions[z]
        stds_organized[i, j] = std

    ax.scatter(
        x=encodings[:, 0].detach().numpy(),
        y=encodings[:, 1].detach().numpy(),
        c="white",
        edgecolors="black",
        # alpha=0.5,
    )
    plot = ax.imshow(
        stds_organized,
        extent=[*viz_limits_x, *viz_limits_y],
        # vmin=0.0,
        vmax=16.0,
        cmap="Blues_r",
    )

    ax.axis("off")
    plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
    # ax.set_title("No extrapolation")
    # ax.set_aspect("auto")
    fig_latent.savefig(
        ROOT_DIR / "latent.jpg",
        bbox_inches="tight",
        dpi=120,
    )


def latent_space_w_images():
    subsampling_rate = 10
    some_training_images = training_data[::subsampling_rate]
    encodings = vae.encode(some_training_images).mean
    fig_latent, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_images(
        z=encodings.detach().numpy(),
        images=some_training_images.detach().numpy(),
        ax=ax,
    )


def multiple_betas_plot():
    viz_limits_x, viz_limits_y = vae.viz_limits(
        padding_percentage=PADDING_PERCENTAGE, center=(-0.6, 0.8)
    )
    axis_x = torch.linspace(*viz_limits_x, 75)
    axis_y = torch.linspace(*viz_limits_y, 75)
    fine_grid_in_latent_space = torch.Tensor(
        [[x, y] for x, y in product(axis_x, axis_y)]
    )

    fig_beta_comparison, axes = plt.subplots(
        1, 3, figsize=(3 * 7, 7), constrained_layout=True
    )
    for beta, ax in zip(BETAS, axes):
        vae_g = VAENormalGeometry()
        vae_g.load_state_dict(torch.load(ROOT_DIR / "vaenormal.pt"))
        vae_g.update_cluster_centers(beta=beta)
        stds = vae_g.decode(fine_grid_in_latent_space).scale.mean(dim=(1, 2))
        encodings = vae_g.encodings()

        std_map = {
            (x.item(), y.item()): std
            for ((x, y), std) in zip(product(axis_x, axis_y), stds)
        }

        positions = {
            (x.item(), y.item()): (i, j)
            for j, x in enumerate(axis_x)
            for i, y in enumerate(reversed(axis_y))
        }
        stds_organized = np.zeros((75, 75))
        for z, std in std_map.items():
            i, j = positions[z]
            stds_organized[i, j] = std

        plot = ax.imshow(
            stds_organized,
            extent=[*viz_limits_x, *viz_limits_y],
            vmin=0.0,
            vmax=16.0,
            cmap="Blues_r",
        )
        ax.scatter(
            encodings[:, 0].detach().numpy(),
            encodings[:, 1].detach().numpy(),
            c="white",
            edgecolors="black",
        )

        ax.set_title(r"$\beta=" + f"{beta}$")
        # ax.set_aspect("auto")
        ax.axis("off")

    fig_beta_comparison.colorbar(plot, ax=axes)
    fig_beta_comparison.savefig(ROOT_DIR / "var.jpg", bbox_inches="tight", dpi=120)


def plot_metric_volume():
    viz_limits_x, viz_limits_y = vae.viz_limits(
        padding_percentage=PADDING_PERCENTAGE, center=(-0.6, 0.8)
    )
    axis_x = torch.linspace(*viz_limits_x, 75)
    axis_y = torch.linspace(*viz_limits_y, 75)
    fine_grid_in_latent_space = torch.Tensor(
        [[x, y] for x, y in product(axis_x, axis_y)]
    )

    fig_beta_comparison, axes = plt.subplots(
        1, 3, figsize=(3 * 7, 7), constrained_layout=True
    )
    for beta, ax in zip(BETAS, axes):
        vae_g = VAENormalGeometry()
        vae_g.load_state_dict(torch.load(ROOT_DIR / "vaenormal.pt"))
        vae_g.update_cluster_centers(beta=beta, n_clusters=400)
        metrics = vae_g.metric(fine_grid_in_latent_space)
        metric_vols = 0.5 * torch.log(torch.det(metrics))
        # stds = vae_g.decode(fine_grid_in_latent_space).scale.mean(dim=(1, 2))
        encodings = vae_g.encodings()

        metric_vol_map = {
            (x.item(), y.item()): metric_vol
            for ((x, y), metric_vol) in zip(product(axis_x, axis_y), metric_vols)
        }

        positions = {
            (x.item(), y.item()): (i, j)
            for j, x in enumerate(axis_x)
            for i, y in enumerate(reversed(axis_y))
        }
        metric_vols_organized = np.zeros((75, 75))
        for z, metric_vol in metric_vol_map.items():
            i, j = positions[z]
            metric_vols_organized[i, j] = metric_vol

        plot = ax.imshow(
            metric_vols_organized,
            extent=[*viz_limits_x, *viz_limits_y],
            vmin=0.0,
            vmax=12.5,
            cmap="viridis",
        )
        ax.scatter(
            encodings[:, 0].detach().numpy(),
            encodings[:, 1].detach().numpy(),
            c="white",
            edgecolors="black",
        )
        # ax.scatter(
        #     vae_g.cluster_centers[:, 0].detach().numpy(),
        #     vae_g.cluster_centers[:, 1].detach().numpy(),
        # )
        # ax.set_aspect("auto")
        ax.set_title(r"$\beta=" + f"{beta}$")
        ax.axis("off")

    plt.colorbar(
        plot,
        ax=axes,
        # fraction=0.046,
        # pad=0.04,
    )
    # fig_beta_comparison.tight_layout()
    fig_beta_comparison.savefig(
        ROOT_DIR / "metric.jpg",
        bbox_inches="tight",
        dpi=120,
    )


if __name__ == "__main__":
    vae = VAENormal()
    vae.load_state_dict(torch.load(ROOT_DIR / "vaenormal.pt"))
    vae.eval()

    training_data, _ = load_data(digits=[1])
    training_data = training_data.tensors[0]

    # Chapter 6
    latent_space_w_images()

    # Chapter 7
    latent_space_plot_w_decoded_uncertainties()
    multiple_betas_plot()
    plot_metric_volume()

    plt.show()
