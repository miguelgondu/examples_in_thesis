import torch
from torch.distributions import Normal

from sklearn.cluster import KMeans

from vae import VAENormal

from stochman.manifold import Manifold


class TranslatedSigmoid(torch.nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta
        self.ALPHA = 6.9077542789816375

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((x - self.beta * self.ALPHA) / self.beta)


def fd_jacobian(function, x, h=1e-4, input_size=28 * 28):
    """
    Compute finite difference Jacobian of given function
    at a single location x. This function is mainly considered
    useful for debugging."""

    no_batch = x.dim() == 1
    if no_batch:
        x = x.unsqueeze(0)
    elif x.dim() > 2:
        raise Exception("The input should be a D-vector or a BxD matrix")
    B, D = x.shape

    # Compute finite differences
    E = h * torch.eye(D, device=x.device)
    Jnum = torch.cat(
        [
            (
                (
                    function(x[b] + E).view(-1, input_size)
                    - function(x[b].unsqueeze(0)).view(-1, input_size)
                ).t()
                / h
            ).unsqueeze(0)
            for b in range(B)
        ]
    )

    if no_batch:
        Jnum = Jnum.squeeze(0)

    return Jnum


def approximate_metric(decode, z, h=0.01, input_size=28 * 28):
    dec_mu = lambda z: decode(z).mean
    dec_std = lambda z: decode(z).scale
    J_std = fd_jacobian(dec_std, z, h=h, input_size=input_size)
    J_mu = fd_jacobian(dec_mu, z, h=h, input_size=input_size)
    # J_mu = torch.zeros_like(J_std)

    if len(J_mu.shape) > 2:
        return torch.bmm(J_mu.transpose(1, 2), J_mu) + torch.bmm(
            J_std.transpose(1, 2), J_std
        )
    else:
        return J_mu.T @ J_mu + J_std.T @ J_std


class VAENormalGeometry(VAENormal, Manifold):
    def __init__(self):
        super().__init__()

    # This method overwrites the decode of the vanilla one.
    def decode(self, z: torch.Tensor, reweight: bool = True) -> Normal:
        if reweight:
            similarity = (
                self.translated_sigmoid(self.min_distance(z))
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            intermediate_normal = super().decode(z)
            dec_mu, dec_std = intermediate_normal.mean, intermediate_normal.scale
            reweighted_std = (1 - similarity) * dec_std + similarity * (
                10.0 * torch.ones_like(dec_std)
            )
            p_x_given_z = Normal(dec_mu, reweighted_std)
        else:
            p_x_given_z = super().decode(z)

        return p_x_given_z

    def update_cluster_centers(self, beta: float = 0.1, n_clusters: int = 50):
        """
        Updates the cluster centers with the support of the data.
        If only_playable is True, the support becomes only the
        playable levels in the training set.
        """
        training_data = self.train_data.tensors[0]
        encodings = self.encode(training_data).mean

        self.kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
        self.kmeans.fit(encodings.cpu().detach().numpy())
        cluster_centers = self.kmeans.cluster_centers_

        self.cluster_centers = torch.from_numpy(cluster_centers).type(torch.float32)

        self.translated_sigmoid = TranslatedSigmoid(beta=beta)

    def min_distance(self, z: torch.Tensor) -> torch.Tensor:
        """
        A function that measures the main distance w.r.t
        the cluster centers.
        """
        zsh = z.shape
        z = z.view(-1, z.shape[-1])  # Nx(zdim)

        z_norm = (z**2).sum(1, keepdim=True)  # Nx1
        center_norm = (self.cluster_centers**2).sum(1).view(1, -1)  # 1x(num_clusters)
        d2 = (
            z_norm
            + center_norm
            - 2.0 * torch.mm(z, self.cluster_centers.transpose(0, 1))
        )  # Nx(num_clusters)
        d2.clamp_(min=0.0)  # Nx(num_clusters)
        min_dist, _ = d2.min(dim=1)  # N

        return min_dist.view(zsh[:-1])

    def metric(self, z: torch.Tensor) -> torch.Tensor:
        return approximate_metric(self.decode, z)
