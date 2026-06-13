"""DL model architectures for side-channel analysis."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron (ASCAD baseline architecture)."""

    def __init__(self, input_size, num_classes=256, hidden_sizes=None, dropout=0.0):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [200, 200, 200, 200, 200, 200]

        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_features, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_features = h

        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """1D Convolutional Neural Network for SCA trace classification."""

    def __init__(self, input_size, num_classes=256, conv_blocks=4,
                 filters=None, kernel_size=11, pool_size=2, fc_size=256,
                 dropout=0.25):
        super().__init__()
        if filters is None:
            filters = [64, 128, 256, 512]

        self.conv = nn.Sequential()
        in_channels = 1
        current_len = input_size

        for i in range(conv_blocks):
            out_channels = filters[min(i, len(filters) - 1)]
            self.conv.add_module(f"conv{i+1}",
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            self.conv.add_module(f"bn{i+1}", nn.BatchNorm1d(out_channels))
            self.conv.add_module(f"relu{i+1}", nn.ReLU())
            self.conv.add_module(f"pool{i+1}", nn.AvgPool1d(pool_size))
            self.conv.add_module(f"drop{i+1}", nn.Dropout1d(dropout * 0.5))
            in_channels = out_channels
            current_len = current_len // pool_size

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * current_len, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(fc_size, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(fc_size, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, L) -> (B, 1, L)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResidualBlock(nn.Module):
    """1D residual block with Conv1d + BN + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=11, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResNet(nn.Module):
    """ResNet for 1D SCA trace classification."""

    def __init__(self, input_size, num_classes=256, num_blocks=4,
                 filters=None, kernel_size=11, fc_size=256):
        super().__init__()
        if filters is None:
            filters = [64, 128, 256, 512]

        self.init_conv = nn.Sequential(
            nn.Conv1d(1, filters[0], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential()
        in_ch = filters[0]
        current_len = input_size
        for i in range(num_blocks):
            out_ch = filters[min(i, len(filters) - 1)]
            stride = 2 if i > 0 else 1
            self.res_blocks.add_module(f"resblock{i+1}",
                ResidualBlock(in_ch, out_ch, kernel_size, stride))
            in_ch = out_ch
            if stride == 2:
                current_len = current_len // 2

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, L) -> (B, 1, L)
        x = self.init_conv(x)
        x = self.res_blocks(x)
        x = self.avg_pool(x).squeeze(-1)
        return self.fc(x)


def get_model(model_name, input_size, num_classes=256):
    """Factory function for model creation."""
    from .config import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model_name]

    if model_name == "mlp":
        return MLP(input_size, num_classes,
                   hidden_sizes=cfg["hidden_sizes"],
                   dropout=cfg["dropout"])
    elif model_name == "cnn":
        return CNN(input_size, num_classes,
                   conv_blocks=cfg["conv_blocks"],
                   filters=cfg["filters"],
                   kernel_size=cfg["kernel_size"],
                   fc_size=cfg["fc_size"],
                   dropout=cfg.get("dropout", 0.0))
    elif model_name == "resnet":
        return ResNet(input_size, num_classes,
                      num_blocks=cfg["num_blocks"],
                      filters=cfg["filters"],
                      kernel_size=cfg["kernel_size"],
                      fc_size=cfg["fc_size"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
