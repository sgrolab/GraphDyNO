import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_dyno.decoders.base import ShapeDecoder


class ConvDecoder(ShapeDecoder):
    """Convolutional decoder for SDF reconstruction from bottleneck vectors.

    Implements ShapeDecoder with identity get_bottleneck — the conv encoder
    already produces bottleneck vectors, so no projection is needed.
    """

    def __init__(self, bottleneck_dim=64, output_size=224,
                 channels=(256, 128, 64, 32, 16),
                 intermediate_conv_layers=0):
        """
        Args:
            bottleneck_dim: Dimension of the input bottleneck vector.
            output_size: Spatial size of output SDF image.
            channels: Channel sizes for decoder stages. Each stage doubles
                spatial resolution via ConvTranspose2d.
            intermediate_conv_layers: Same-resolution Conv2d layers per stage.
        """
        super().__init__()
        self._bottleneck_dim = bottleneck_dim
        self.output_size = output_size
        self.channels = channels
        self.init_spatial = 7

        self.fc_decode = nn.Linear(bottleneck_dim,
                                   channels[0] * self.init_spatial * self.init_spatial)

        layers = []
        for i in range(len(channels) - 1):
            layers += [
                nn.ConvTranspose2d(channels[i], channels[i + 1],
                                   kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(1, channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
            for _ in range(intermediate_conv_layers):
                layers += [
                    nn.Conv2d(channels[i + 1], channels[i + 1],
                              kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(1, channels[i + 1]),
                    nn.ReLU(inplace=True),
                ]

        # Final upsampling to 1 channel (raw logits for tanh)
        layers.append(nn.ConvTranspose2d(channels[-1], 1,
                                         kernel_size=4, stride=2, padding=1))
        self.decoder = nn.Sequential(*layers)

    @property
    def bottleneck_dim(self) -> int:
        return self._bottleneck_dim

    @property
    def input_dim(self) -> int:
        # Embeddings are already bottleneck vectors (identity projection)
        return self._bottleneck_dim

    def get_bottleneck(self, embeddings):
        """Identity — conv encoder already produces bottleneck vectors.

        Args:
            embeddings: (B, bottleneck_dim)
        Returns:
            bottleneck: (B, bottleneck_dim)
        """
        return embeddings

    def decode_from_bottleneck(self, bottleneck):
        """Decode bottleneck vectors to SDF images.

        Args:
            bottleneck: (B, bottleneck_dim)
        Returns:
            sdf: (B, output_size, output_size) in [-1, 1]
        """
        B = bottleneck.shape[0]

        x = self.fc_decode(bottleneck)
        x = x.view(B, self.channels[0], self.init_spatial, self.init_spatial)
        x = self.decoder(x)  # (B, 1, H, W)

        x = F.interpolate(x, size=(self.output_size, self.output_size),
                          mode='bilinear', align_corners=False)

        return torch.tanh(x.squeeze(1))  # (B, H, W) in [-1, 1]

    def training_step(self, embeddings, targets):
        """Compute loss for a training step.

        Args:
            embeddings: (B, bottleneck_dim) bottleneck vectors
            targets: (B, output_size, output_size) ground truth SDF in [-1, 1]
        Returns:
            Dictionary of loss terms
        """
        bottleneck = self.get_bottleneck(embeddings)
        preds = self.decode_from_bottleneck(bottleneck)
        return {
            'image_mse': F.mse_loss(preds, targets.float()),
            'embedding_l2': torch.mean(bottleneck ** 2),
        }
