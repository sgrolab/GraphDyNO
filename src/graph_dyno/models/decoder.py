import torch
from torch import nn
from torch.nn import functional as F

class Decoder(nn.Module):
    def __init__(
        self,
        input_size=128,
        hidden_channels=[128, 64],
        output_channels=32,
        output_size=64,
    ):
        """
        Args:
            latent_dim (int): size of the latent vector (z)
            hidden_channels (list[int]): list of channels for each upsampling stage
            output_channels (int): number of output image channels (e.g. 3 for RGB)
            output_size (int): final image size (assumes square, e.g., 64 -> 64x64)
        """
        super().__init__()

        # Compute how many upsampling stages are needed
        num_upsamples = len(hidden_channels)
        init_size = output_size // (2**num_upsamples)

        # Start by mapping latent vector -> small feature map
        self.fc = nn.Linear(input_size, hidden_channels[0] * init_size * init_size)
        self.init_size = init_size

        # Build upsampling layers
        layers = []
        in_chs = hidden_channels[0]
        for out_chs in hidden_channels[1:]:
            layers += [
                nn.ConvTranspose2d(in_chs, out_chs, kernel_size=4, stride=2, padding=1),
                # nn.InstanceNorm2d(out_chs, affine=True),
                nn.ReLU(True),
            ]
            in_chs = out_chs

        # Final layer -> image
        layers += [
            nn.ConvTranspose2d(
                in_chs, output_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), -1, self.init_size, self.init_size)
        return self.decoder(x)
