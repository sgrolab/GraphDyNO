import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_dyno.decoders.base import ShapeDecoder


class DINODecoder(ShapeDecoder):
    """Linear projection to bottleneck + convolutional decoder for SDF reconstruction."""

    def __init__(self, input_dim=384, bottleneck_dim=64, output_size=224,
                 channels=(256, 128, 64, 32, 16),
                 intermediate_conv_layers=0,
                 proj_hidden_layers=0):
        """
        Args:
            input_dim: Dimension of input embeddings (CLS or CLS+patches).
            bottleneck_dim: Dimension of the bottleneck vector.
            output_size: Spatial size of output SDF image.
            channels: Tuple of channel sizes for the decoder stages.
                Each stage doubles the spatial resolution via ConvTranspose2d.
                A final ConvTranspose2d maps the last channel to 1.
            intermediate_conv_layers: Number of same-resolution Conv2d layers
                inserted after each upsampling stage (before the final output layer).
        """
        super().__init__()
        self.output_size = output_size
        self._bottleneck_dim = bottleneck_dim
        self.channels = channels
        self.init_spatial = 7

        # input_dim should be D (CLS only) or 2*D (CLS + mean patches)
        self.proj_hidden_layers = proj_hidden_layers
        if proj_hidden_layers == 0:
            self.linear_proj = nn.Linear(input_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(input_dim, bottleneck_dim), nn.GELU()]
            for _ in range(proj_hidden_layers - 1):
                layers += [nn.Linear(bottleneck_dim, bottleneck_dim), nn.GELU()]
            layers.append(nn.Linear(bottleneck_dim, bottleneck_dim))
            self.linear_proj = nn.Sequential(*layers)

        # Decode from bottleneck vector to spatial output
        self.fc_decode = nn.Linear(bottleneck_dim, channels[0] * self.init_spatial * self.init_spatial)

        # Build decoder layers dynamically
        layers = []
        for i in range(len(channels) - 1):
            # Upsampling: doubles spatial resolution
            layers.append(nn.ConvTranspose2d(channels[i], channels[i + 1],
                                             kernel_size=4, stride=2, padding=1))
            layers.append(nn.GroupNorm(1, channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))

            # Intermediate same-resolution conv layers
            for _ in range(intermediate_conv_layers):
                layers.append(nn.Conv2d(channels[i + 1], channels[i + 1],
                                        kernel_size=3, stride=1, padding=1))
                layers.append(nn.GroupNorm(1, channels[i + 1]))
                layers.append(nn.ReLU(inplace=True))

        # Final upsampling to 1 channel (no BN/ReLU — raw logits for tanh)
        layers.append(nn.ConvTranspose2d(channels[-1], 1,
                                         kernel_size=4, stride=2, padding=1))

        self.decoder = nn.Sequential(*layers)

    @property
    def bottleneck_dim(self) -> int:
        return self._bottleneck_dim

    @property
    def input_dim(self) -> int:
        return self.linear_proj.in_features

    def get_bottleneck(self, cls_token):
        """Extract bottleneck embedding from DINO CLS token.

        Args:
            cls_token: (B, input_dim) CLS token from DINO
        Returns:
            bottleneck: (B, bottleneck_dim)
        """
        return self.linear_proj(cls_token)

    def decode_from_bottleneck(self, bottleneck):
        """Decode a bottleneck vector to an SDF image.

        Args:
            bottleneck: (B, bottleneck_dim)
        Returns:
            mask: (B, output_size, output_size) predicted SDF in [-1, 1]
        """
        B = bottleneck.shape[0]

        x = self.fc_decode(bottleneck)
        x = x.view(B, self.channels[0], self.init_spatial, self.init_spatial)

        x = self.decoder(x)  # (B, 1, H, W)

        x = F.interpolate(x, size=(self.output_size, self.output_size),
                          mode='bilinear', align_corners=False)

        return torch.tanh(x.squeeze(1))  # (B, H, W) in [-1, 1]

    def forward(self, cls_token):
        """
        Args:
            cls_token: (B, input_dim) CLS token from DINO
        Returns:
            mask: (B, output_size, output_size) predicted SDF in [-1, 1]
        """
        bottleneck = self.get_bottleneck(cls_token)  # (B, bottleneck_dim)
        return self.decode_from_bottleneck(bottleneck)

    def training_step(self, embeddings, targets):
        """Compute loss for a training step.

        Args:
            embeddings: (B, input_dim) CLS token from DINO
            targets: (B, output_size, output_size) ground truth SDF in [-1, 1]
        Returns:
            loss: Dictionary of loss terms
        """
        bottleneck = self.get_bottleneck(embeddings)  # (B, bottleneck_dim)
        preds = self.decode_from_bottleneck(bottleneck)  # (B, output_size, output_size)
        loss = {
            'image_mse': F.mse_loss(preds, targets.float()),
            'embedding_l2': torch.mean(bottleneck ** 2)
        }

        return loss
