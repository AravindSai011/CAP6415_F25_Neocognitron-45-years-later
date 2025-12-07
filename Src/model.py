import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Baseline modern CNN (you already used this one)
# ---------------------------------------------------------
class ImprovedNeoCNN(nn.Module):
    """
    Modern Neocognitron-inspired CNN:
    S-layers = Conv + BN + ReLU
    C-layers = MaxPooling
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # S1 / C1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # S2 / C2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Deeper S-layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------
# True S / C style blocks with lateral inhibition
# ---------------------------------------------------------

class SCellBlock(nn.Module):
    """
    S-cell block:
      - Excitatory convolution (learnable)
      - Surround inhibition (fixed depthwise conv)
      - Nonlinearity

    Output ~ ReLU( excitatory - inhibition ).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        inh_kernel_size: int = 7,
        inh_strength: float = 0.5,
    ):
        super().__init__()

        padding = kernel_size // 2
        inh_padding = inh_kernel_size // 2

        # Excitatory conv (like classical S-cell)
        self.conv_exc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # Depthwise inhibitory filter applied on the excitatory response
        self.conv_inh = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=inh_kernel_size,
            padding=inh_padding,
            groups=out_channels,
            bias=False,
        )

        # Initialize inhibitory kernel as a Difference-of-Gaussians–like pattern
        with torch.no_grad():
            w = self._make_inhibition_kernel(inh_kernel_size)
            # Same kernel for every channel
            self.conv_inh.weight.copy_(w.repeat(out_channels, 1, 1, 1))

        # Freeze inhibition weights (not learned)
        for p in self.conv_inh.parameters():
            p.requires_grad = False

        self.inh_strength = inh_strength

    def _make_inhibition_kernel(self, k: int) -> torch.Tensor:
        """
        Create a simple center-surround kernel:
        center negative, surround positive.
        """
        grid = torch.arange(k) - (k - 1) / 2.0
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        r2 = xx**2 + yy**2

        # two Gaussians: narrow center & broader surround
        center = torch.exp(-r2 / (2 * (k / 6.0) ** 2))
        surround = torch.exp(-r2 / (2 * (k / 3.0) ** 2))

        # surround - center → positive ring, negative center
        kernel = surround - center
        kernel = kernel - kernel.mean()  # zero-mean

        # shape (1,1,k,k) for depthwise conv template
        return kernel.view(1, 1, k, k)

    def forward(self, x):
        # excitatory drive
        exc = self.conv_exc(x)

        # inhibitory surround applied on excitatory activity
        inh = self.conv_inh(exc) * self.inh_strength

        # subtract inhibition, then rectification
        y = exc - F.relu(inh)
        return F.relu(y)


class CCellBlock(nn.Module):
    """
    C-cell block: pooling for position tolerance.
    """
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.pool(x)


# ---------------------------------------------------------
# Lateral-inhibition Neocognitron-style network
# ---------------------------------------------------------

class LateralInhibitionNeocognitron(nn.Module):
    """
    Neocognitron-like network built from S- and C-cell blocks
    with explicit lateral inhibition in S-layers.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.s1 = SCellBlock(1, 32, kernel_size=5, inh_kernel_size=7, inh_strength=0.5)
        self.c1 = CCellBlock(kernel_size=2, stride=2)

        self.s2 = SCellBlock(32, 64, kernel_size=5, inh_kernel_size=7, inh_strength=0.5)
        self.c2 = CCellBlock(kernel_size=2, stride=2)

        # a third S-layer without pooling afterwards
        self.s3 = SCellBlock(64, 128, kernel_size=3, inh_kernel_size=5, inh_strength=0.4)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # S1 → C1
        x = self.s1(x)
        x = self.c1(x)

        # S2 → C2
        x = self.s2(x)
        x = self.c2(x)

        # S3
        x = self.s3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
