import torch.nn as nn


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=4):
        super(SimpleUNet, self).__init__()

        # Encoder (contracting path)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (expanding path)
        self.dec4 = self.conv_block(1024, 512)
        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)

        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoding path
        dec4 = self.dec4(bottleneck)
        dec3 = self.dec3(dec4 + enc4)
        dec2 = self.dec2(dec3 + enc3)
        dec1 = self.dec1(dec2 + enc2)

        # Output layer
        output = self.final_conv(dec1)
        return self.sigmoid(output)  # Output confidence map (0-1)


class CNNBackbone(nn.Module):
    def __init__(self, in_channels=8, out_channels=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.final(x)
        return self.sigmoid(x)


from torchvision.models import resnet18

class ResNet18Backbone(nn.Module):
    def __init__(self, in_channels=8, out_channels=4):
        super().__init__()
        resnet = resnet18(weights=None)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            resnet.layer1,  # 64
            resnet.layer2,  # 128
            resnet.layer3   # 256
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.Conv2d(64, out_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)   # 输出: [B, 256, H/4, W/4]
        x = self.decoder(x)   # 上采样回原始尺寸
        return self.sigmoid(x)


class MLPBackbone(nn.Module):
    def __init__(self, in_channels=8, hidden_dim=64, out_channels=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            alpha: (B, 1, H, W) - confidence map
        """
        B, C, H, W = x.shape
        # [B, C, H, W] → [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        # → [B*H*W, C]
        x = x.reshape(-1, C)
        # → [B*H*W, 1]
        alpha = self.mlp(x)
        # → [B, 1, H, W]
        alpha = alpha.view(B, H, W, -1).permute(0, 3, 1, 2)
        return alpha


import torch.nn as nn

class TransformerBackbone(nn.Module):
    def __init__(self, in_channels=8, out_channels=4, dim=128, num_heads=4, num_layers=2):
        super().__init__()

        # -----------------------------
        # Patch embedding: (B, C, H, W) → (B, dim, H/4, W/4)
        # -----------------------------
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=4, stride=4)

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # -----------------------------
        # Upsample back to original size
        # -----------------------------
        self.up = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        # Final conv (B, dim/4, H, W) → (B, out_channels, H, W)
        self.final = nn.Conv2d(dim // 4, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, 8, H, W)
        Returns:
            alpha: (B, 4, H, W)
        """

        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)          # → (B, dim, H/4, W/4)
        Bh, Bw = x.shape[2], x.shape[3]

        # Flatten → Transformer tokens
        x = x.flatten(2).transpose(1, 2)  # → (B, N, dim), N = (H/4 * W/4)

        # Transformer encoding
        x = self.transformer(x)          # (B, N, dim)

        # Restore spatial map
        x = x.transpose(1, 2).view(B, -1, Bh, Bw)  # → (B, dim, H/4, W/4)

        # Upsampling
        x = self.up(x)                   # → (B, dim/4, H, W)

        # Confidence map
        x = self.final(x)                # → (B, out_channels, H, W)

        return self.sigmoid(x)

