import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))

        dec3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))

        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        return self.final(dec1)  # no sigmoid here, let BCEWithLogits handle that


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, dropout=0.0):
        super().__init__()
        f = init_features
        self.pool = nn.MaxPool2d(2)
        self.up = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder
        self.conv00 = DoubleConv(in_channels, f, dropout)
        self.conv10 = DoubleConv(f, f*2, dropout)
        self.conv20 = DoubleConv(f*2, f*4, dropout)

        # Decoder / Nested connections
        self.conv01 = DoubleConv(f + f*2, f, dropout)
        self.conv11 = DoubleConv(f*2 + f*4, f*2, dropout)
        self.conv02 = DoubleConv(f + f*2, f, dropout)

        self.final = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x01 = self.conv01(torch.cat([x00, self.up(x10)], 1))

        x20 = self.conv20(self.pool(x10))
        x11 = self.conv11(torch.cat([x10, self.up(x20)], 1))
        x02 = self.conv02(torch.cat([x01, self.up(x11)], 1))

        return self.final(x02)

if __name__ == "__main__":
    model = UNetPlusPlus()
    x = torch.randn((1, 1, 256, 256))
    y = model(x)
    print(y.shape)
