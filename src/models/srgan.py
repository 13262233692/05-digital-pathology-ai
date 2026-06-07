import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class ResidualBlock(nn.Module):
    def __init__(self, num_filters: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class UpsampleBlock(nn.Module):
    def __init__(self, num_filters: int = 64, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(num_filters, num_filters * scale_factor ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.pixel_shuffle(self.conv(x)))


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = 64,
        num_residual_blocks: int = 16,
        scale_factor: int = 4
    ):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()
        
        residual_blocks = []
        for _ in range(num_residual_blocks):
            residual_blocks.append(ResidualBlock(num_filters))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_filters)
        
        upsample_blocks = []
        num_upsample = 2 if scale_factor == 4 else 1
        for _ in range(num_upsample):
            upsample_blocks.append(UpsampleBlock(num_filters, scale_factor=2))
        self.upsample_blocks = nn.Sequential(*upsample_blocks)
        
        self.conv3 = nn.Conv2d(num_filters, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.prelu(self.conv1(x))
        out = self.residual_blocks(out1)
        out = self.bn(self.conv2(out)) + out1
        out = self.upsample_blocks(out)
        return torch.tanh(self.conv3(out))


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, num_filters: int = 64):
        super().__init__()
        
        def discriminator_block(in_filters: int, out_filters: int, stride: int = 1, use_bn: bool = True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.blocks = nn.Sequential(
            discriminator_block(in_channels, num_filters, stride=1, use_bn=False),
            discriminator_block(num_filters, num_filters, stride=2),
            discriminator_block(num_filters, num_filters * 2, stride=1),
            discriminator_block(num_filters * 2, num_filters * 2, stride=2),
            discriminator_block(num_filters * 2, num_filters * 4, stride=1),
            discriminator_block(num_filters * 4, num_filters * 4, stride=2),
            discriminator_block(num_filters * 4, num_filters * 8, stride=1),
            discriminator_block(num_filters * 8, num_filters * 8, stride=2),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_filters * 8, 1024)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blocks(x)
        out = self.adaptive_pool(out)
        out = self.flatten(out)
        out = self.leaky_relu(self.fc1(out))
        return torch.sigmoid(self.fc2(out))


class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer: int = 35):
        super().__init__()
        vgg = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer + 1])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class SRGAN(nn.Module):
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        feature_extractor: VGGFeatureExtractor,
        device: torch.device = None
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.feature_extractor.to(self.device)

    def forward(self, lr_images: torch.Tensor) -> torch.Tensor:
        return self.generator(lr_images.to(self.device))

    def compute_content_loss(self, sr_images: torch.Tensor, hr_images: torch.Tensor) -> torch.Tensor:
        sr_features = self.feature_extractor(sr_images)
        hr_features = self.feature_extractor(hr_images.to(self.device))
        return F.mse_loss(sr_features, hr_features)

    def compute_adversarial_loss(self, sr_images: torch.Tensor, real: bool = True) -> torch.Tensor:
        validity = self.discriminator(sr_images)
        target = torch.ones_like(validity) if real else torch.zeros_like(validity)
        return F.binary_cross_entropy(validity, target.to(self.device))

    def compute_pixel_loss(self, sr_images: torch.Tensor, hr_images: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(sr_images, hr_images.to(self.device))

    @staticmethod
    def init_weights(m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @classmethod
    def build(
        cls,
        in_channels: int = 3,
        num_filters: int = 64,
        num_residual_blocks: int = 16,
        scale_factor: int = 4,
        device: torch.device = None
    ) -> "SRGAN":
        generator = Generator(in_channels, num_filters, num_residual_blocks, scale_factor)
        discriminator = Discriminator(in_channels, num_filters)
        feature_extractor = VGGFeatureExtractor()
        
        generator.apply(cls.init_weights)
        discriminator.apply(cls.init_weights)
        
        return cls(generator, discriminator, feature_extractor, device)
