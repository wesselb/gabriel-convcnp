import lab.torch as B
import torch
import torch.nn as nn

__all__ = ["SetConv1dDecoder"]


class SetConv1dDecoder(nn.Module):
    def __init__(self, discretisation):
        nn.Module.__init__(self)
        self.log_scale = nn.Parameter(
            B.log(torch.tensor(2 / discretisation.points_per_unit)),
            requires_grad=True,
        )

    def forward(self, xz, z, x):
        # Compute interpolation weights.
        dists2 = B.pw_dists2(x, xz)
        weights = B.exp(-0.5 * dists2 / B.exp(2 * self.log_scale))

        # Put feature channel last.
        z = B.transpose(z)

        # Interpolate to `x`.
        z = B.matmul(weights, z)

        return z
