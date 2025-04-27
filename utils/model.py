import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.constants import INPUT_IMG_SIZE


class CustomVGG(nn.Module):
    """
    Custom multi-class classification model
    with VGG16 feature extractor, pretrained on ImageNet
    and custom classification head.
    Parameters for the first convolutional blocks are freezed.

    Returns class scores when in train mode.
    Returns class probs and normalized feature maps when in eval mode.
    """

    def __init__(self, n_classes=2, pretrained=True):
        super(CustomVGG, self).__init__()

        # Load VGG16 feature extractor, pretrained on ImageNet
        self.feature_extractor = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        ).features[:-1]

        # Define custom classification head
        self.classification_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(
                kernel_size=(INPUT_IMG_SIZE[0] // 2**5, INPUT_IMG_SIZE[1] // 2**5)
            ),
            nn.Flatten(),
            nn.Linear(
                in_features=self.feature_extractor[-2].out_channels,
                out_features=n_classes,
            ),
        )

        # Freeze parameters for the first convolutional blocks of the feature extractor
        self._freeze_params()

    def _freeze_params(self):
        # Loop through all parameters for the first 23 convolutional blocks
        for param in self.feature_extractor[:23].parameters():
            # Freeze parameters
            param.requires_grad = False

    def forward(self, x):
        # Compute feature maps using VGG16 feature extractor
        feature_maps = self.feature_extractor(x)

        # Compute class scores using custom classification head
        scores = self.classification_head(feature_maps)

        # If in training mode, return class scores
        if self.training:
            return scores

        # If in evaluation mode, return class probabilities and normalized feature maps
        else:
            # Compute class probabilities from class scores using softmax activation function
            probs = nn.functional.softmax(scores, dim=-1)

            # Compute normalized feature maps from classification head weights and feature maps
            weights = self.classification_head[3].weight
            weights = (
                weights.unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(0)
                .repeat(
                    (
                        x.size(0),
                        1,
                        1,
                        INPUT_IMG_SIZE[0] // 2**4,
                        INPUT_IMG_SIZE[0] // 2**4,
                    )
                )
            )
            feature_maps = feature_maps.unsqueeze(1).repeat((1, probs.size(1), 1, 1, 1))
            location = torch.mul(weights, feature_maps).sum(axis=2)
            location = F.interpolate(location, size=INPUT_IMG_SIZE, mode="bilinear")

            # Normalize feature maps to range [0, 1]
            maxs, _ = location.max(dim=-1, keepdim=True)
            maxs, _ = maxs.max(dim=-2, keepdim=True)
            mins, _ = location.min(dim=-1, keepdim=True)
            mins, _ = mins.min(dim=-2, keepdim=True)
            norm_location = (location - mins) / (maxs - mins)

            # Return class probabilities and normalized feature maps
            return probs, norm_location
