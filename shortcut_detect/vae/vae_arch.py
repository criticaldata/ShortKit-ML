"""
VAE architecture and classifier for shortcut detection.

Adapted from Müller et al., Fraunhofer-AISEC/shortcut-detection-vae.
https://github.com/Fraunhofer-AISEC/shortcut-detection-vae
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50

try:
    from torchvision.models import ResNet50_Weights
except ImportError:
    ResNet50_Weights = None  # type: ignore[misc, assignment]


class ResnetVAE(nn.Module):
    """
    Beta-VAE with ResNet-50 encoder for image shortcut detection.

    Encodes images to latent (mu, log_var), decodes for reconstruction.
    Supports optional classification head for predictiveness analysis.
    """

    def __init__(
        self,
        input_size: int = 128,
        latent_dim: int = 32,
        input_channels: int = 3,
        kld_weight: float = 0.001,
        cls_weight: float = 0.0,
        hidden_dims: list[int] | None = None,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        self.cls_weight = cls_weight
        self.num_classes = num_classes or 2
        self.input_size = input_size

        # Build decoder hidden dims from input size
        self.hidden_dims = hidden_dims or []
        if not self.hidden_dims:
            current_x = self.input_size
            p, s, k = 1, 2, 3
            h = 32
            while current_x >= 4:
                self.hidden_dims.append(h)
                current_x = (current_x + 2 * p - k) // s + 1
                h *= 2

        # Encoder: ResNet-50 backbone
        weights = ResNet50_Weights.IMAGENET1K_V1 if ResNet50_Weights else None
        resnet = resnet50(weights=weights)
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_var = nn.Linear(2048, latent_dim)

        self.classification_layer = nn.Linear(self.latent_dim, self.num_classes)

        # Decoder
        rev_dims = list(reversed(self.hidden_dims))
        self.decoder_input = nn.Linear(latent_dim, rev_dims[0] * 16)

        modules_dec = []
        for i in range(len(rev_dims) - 1):
            modules_dec.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        rev_dims[i],
                        rev_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(rev_dims[i + 1]),
                    nn.ReLU(),
                )
            )
        self.decoder = nn.Sequential(*modules_dec)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                rev_dims[-1],
                rev_dims[-1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(rev_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(
                rev_dims[-1],
                out_channels=self.input_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to (mu, log_var)."""
        if self.input_channels == 1:
            # ResNet-50 expects 3-channel input; tile grayscale channel.
            x = x.repeat(1, 3, 1, 1)
        x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)
        mu = self.fc_mu(features)
        log_var = self.fc_var(features)
        return mu, log_var

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        rec = self.decoder_input(x)
        rec = rec.view(-1, self.hidden_dims[-1], 4, 4)
        rec = self.decoder(rec)
        rec = self.final_layer(rec)
        return rec

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(
        self, input: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass: loss and intermediate tensors."""
        mu, log_var = self.encode(input)
        pred = self.classification_layer(mu)
        z = self.reparametrize(mu, log_var)
        output = self.decode(z)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        rec_loss = F.binary_cross_entropy(output, input, reduction="none")
        rec_loss = torch.mean(torch.sum(rec_loss, dim=(1, 2, 3)), dim=0)
        cls_loss = F.cross_entropy(pred, labels, reduction="mean")

        loss = rec_loss + self.kld_weight * kld_loss + self.cls_weight * cls_loss
        return loss, [output, input, mu, log_var, z, kld_loss, rec_loss]


class VAEClassifier(nn.Module):
    """
    Linear classifier on frozen VAE encoder latent codes.

    Used to compute per-dimension predictiveness (abs weight sums).
    """

    def __init__(
        self,
        vae: ResnetVAE,
        latent_dim: int,
        classes: int,
    ) -> None:
        super().__init__()
        self.backbone = vae
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(latent_dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu, _ = self.backbone.encode(x)
        return self.fc(mu)
