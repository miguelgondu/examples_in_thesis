"""
A simple implementation of a VAE that decodes to the normal
"""
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer

from torch.distributions import Normal, kl_divergence

from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def load_data(digits: List[int] = [0, 1]) -> List[TensorDataset]:
    """
    This function returns 2 TensorDatasets:
    train_dataset, test_dataset.

    These are no normal Datasets, unfortunately:
    the data stored inside them is in dataset.tensors.
    """
    data_path = str(DATA_DIR)
    train_dataset = datasets.MNIST(
        root=data_path, train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root=data_path, train=False, transform=transforms.ToTensor(), download=True
    )
    train_labels = train_dataset.targets
    test_labels = test_dataset.targets

    # Getting only the ones with said digits.
    train_cond = torch.zeros_like(train_labels)
    test_cond = torch.zeros_like(test_labels)
    for digit in digits:
        train_cond = torch.logical_or(train_cond, train_labels == digit)
        test_cond = torch.logical_or(test_cond, test_labels == digit)

    train_dataset = TensorDataset(
        train_dataset.data[train_cond] / 255.0, train_labels[train_cond]
    )
    test_dataset = TensorDataset(
        test_dataset.data[test_cond] / 255.0, test_labels[test_cond]
    )

    return train_dataset, test_dataset


class VAENormal(nn.Module):
    def __init__(self):
        super(VAENormal, self).__init__()
        self.device = "cpu"
        self.train_data, self.test_data = load_data(digits=[1])
        one_example = self.train_data.tensors[0]
        self.h = one_example.shape[1]
        self.w = one_example.shape[2]
        self.input_dim = self.h * self.w
        self.z_dim = 2

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.Tanh(),
        ).to(self.device)
        self.enc_mu = nn.Sequential(nn.Linear(64, self.z_dim)).to(self.device)
        self.enc_var = nn.Sequential(nn.Linear(64, self.z_dim)).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 64),
            nn.Tanh(),
        ).to(self.device)
        self.dec_mu = nn.Linear(64, self.input_dim).to(self.device)
        self.dec_var = nn.Linear(64, self.input_dim).to(self.device)

        self.p_z = Normal(
            torch.zeros(self.z_dim, device=self.device),
            torch.ones(self.z_dim, device=self.device),
        )

    def encode(self, x: torch.Tensor) -> Normal:
        # Returns q(z | x) = Normal(mu, sigma)
        x = x.view(-1, self.input_dim).to(self.device)
        result = self.encoder(x)
        mu = self.enc_mu(result)
        log_var = self.enc_var(result)

        return Normal(mu, torch.exp(0.5 * log_var))

    def decode(self, z: torch.Tensor) -> Normal:
        result = self.decoder(z.to(self.device))
        dec_mu = self.dec_mu(result).reshape(-1, self.h, self.w)
        dec_log_var = self.dec_var(result).reshape(-1, self.h, self.w)

        return Normal(dec_mu, torch.exp(0.5 * dec_log_var))

    def forward(self, x: torch.Tensor) -> Tuple[Normal, Normal]:
        q_z_given_x = self.encode(x.to(self.device))
        z = q_z_given_x.rsample()
        p_x_given_z = self.decode(z.to(self.device))

        return [q_z_given_x, p_x_given_z]

    def encodings(self) -> torch.Tensor:
        """
        Returns the encodings of the training data
        """
        training_data = self.train_data.tensors[0]
        return self.encode(training_data).mean

    def viz_limits(
        self, padding_percentage: float = 0.05, center: Tuple[float, float] = None
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Returns the x-lims and the y-lims in a single tuple.
        """
        encodings = self.encodings()
        min_x = encodings[:, 0].min().item()
        max_x = encodings[:, 0].max().item()
        min_y = encodings[:, 1].min().item()
        max_y = encodings[:, 1].max().item()

        if center is not None:
            center_x, center_y = center
        else:
            center_x, center_y = encodings.median(dim=0).values.tolist()

        dist_ = max(
            center_x - min_x,
            max_x - center_x,
            center_y - min_y,
            max_y - center_y,
        )
        padding = max(max_x - min_x, max_y - min_y) * padding_percentage

        return (center_x - dist_ - padding, center_x + dist_ + padding), (
            center_y - dist_ - padding,
            center_y + dist_ + padding,
        )

    def elbo_loss_function(
        self, x: torch.Tensor, q_z_given_x: Normal, p_x_given_z: Normal
    ) -> torch.Tensor:
        rec_loss = -p_x_given_z.log_prob(x).sum(dim=(1, 2))  # b
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)  # b

        return (rec_loss + kld).mean()


def fit(
    model: VAENormal,
    optimizer: Optimizer,
    data_loader: DataLoader,
    device: str,
):
    model.train()
    running_loss = 0.0
    for levels, _ in data_loader:
        levels = levels.to(device)
        optimizer.zero_grad()
        q_z_given_x, p_x_given_z = model.forward(levels)
        loss = model.elbo_loss_function(levels, q_z_given_x, p_x_given_z)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss / len(data_loader)


if __name__ == "__main__":
    """Trains the VAE"""
    vae = VAENormal()
    train_loader = DataLoader(vae.train_data, batch_size=64)
    adam = Adam(vae.parameters(), lr=1e-3)

    for _ in range(50):
        print(fit(vae, adam, train_loader, "cpu"))

    torch.save(vae.state_dict(), ROOT_DIR / "vaenormal.pt")
