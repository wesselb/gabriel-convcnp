import lab.torch as B
import torch
import torch.nn as nn

from .decoder import SetConv1dDecoder
from .discretisation import Discretisation1d
from .encoder import SetConv1dEncoder
from .unet import UNet
from .util import convert_batched_data

__all__ = ["DualConvCNP"]


class DualConvCNP(nn.Module):
    def __init__(
        self,
        sigma: float = 0.1,
        points_per_unit: float = 32,
        small: bool = False,
    ):
        super(DualConvCNP, self).__init__()

        # Construct CNN:
        self.conv = UNet(
            dimensionality=1,
            in_channels=4,  # Two for regression and two for classification
            out_channels=3,  # Two for mean and variance and one for class. prob.
            channels=(8, 16, 16, 32) if small else (8, 16, 16, 32, 32, 64),
        )

        # Construct discretisation:
        self.disc = Discretisation1d(
            points_per_unit=points_per_unit,
            multiple=2 ** self.conv.num_halving_layers,
            margin=0.1,
        )

        # Construct encoder and decoder:
        self.encoder = SetConv1dEncoder(self.disc)
        self.decoder = SetConv1dDecoder(self.disc)

        # Learnable observation noise for regression:
        self.log_sigma = nn.Parameter(
            B.log(torch.tensor(sigma, dtype=torch.float32)),
            requires_grad=True,
        )

    def forward(self, batch):
        # Ensure that inputs are of the right shape.
        batch = {k: convert_batched_data(v) for k, v in batch.items()}

        # Construct discretisation.
        with B.device(B.device(batch["x_context_class"])):
            x_grid = self.disc(
                batch["x_context_class"],
                batch["x_target_class"],
                batch["x_context_reg"],
                batch["x_target_reg"],
            )[None, :, None]

        # Run encoders.
        z_class = self.encoder(
            batch["x_context_class"],
            batch["y_context_class"],
            x_grid,
        )
        z_reg = self.encoder(
            batch["x_context_reg"],
            batch["y_context_reg"],
            x_grid,
        )

        # Run CNN.
        z = B.concat(z_class, z_reg, axis=1)
        z = self.conv(z)
        z_class = z[:, :1, :]
        z_reg = z[:, 1:, :]

        # Run decoders.
        z_class = self.decoder(x_grid, z_class, batch["x_target_class"])
        z_reg = self.decoder(x_grid, z_reg, batch["x_target_reg"])

        # Return parameters for classification and regression.
        return B.sigmoid(z_class), (z_reg[:, :, :1], B.exp(z_reg[:, :, 1:]))
