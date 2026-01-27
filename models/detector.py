import torch.nn as nn
from torchvision import models


class Detector(nn.Module):
    def __init__(self, backbone_name="resnet50"):
        super().__init__()

        if backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

            self.feature_dim = backbone.fc.in_features

            # extract everything except final fc
            self.backbone = nn.Sequential(
                *(list(backbone.children())[:-1])
            )

        else:
            raise ValueError("Unsupported backbone")

        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, x):
        features = self.backbone(x)          # (B, 2048, 1, 1)
        features = features.view(x.size(0), -1)  # (B, 2048)
        logits = self.classifier(features)
        return features, logits
