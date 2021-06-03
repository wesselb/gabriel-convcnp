import lab.torch as B
import torch
import torch.nn as nn


__all__ = ["SetConv1dEncoder"]


class SetConv1dEncoder(nn.Module):
    def __init__(self, discretisation):
        nn.Module.__init__(self)
        self.log_scale = nn.Parameter(
            B.log(torch.tensor(2 / discretisation.points_per_unit)),
            requires_grad=True,
        )

    def forward(self, xz, z, x_grid):
        with B.device(B.device(z)):
            # Construct density channel.
            density_channel = B.ones(B.dtype(z), *B.shape(z)[:2], 1)

        # Prepend density channel.
        z = B.concat(density_channel, z, axis=2)

        # Compute interpolation weights.
        dists2 = B.pw_dists2(x_grid, xz)
        weights = B.exp(-0.5 * dists2 / B.exp(2 * self.log_scale))

        # Interpolate to grid.
        z = B.matmul(weights, z)

        # Normalise by density channel.
        z = B.concat(z[:, :, :1], z[:, :, 1:] / (z[:, :, :1] + 1e-8), axis=2)

        # Put feature channel second.
        z = B.transpose(z)

        return z
