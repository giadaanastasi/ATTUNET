import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 8)
        self.enc2 = self.conv_block(8, 16)
        self.enc3 = self.conv_block(16, 32)
        self.enc4 = self.conv_block(32, 64)

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)

        # Decoder
        self.upconv4 = self.upconv_block(128 + 64, 64)
        self.upconv3 = self.upconv_block(64 + 32, 32)
        self.upconv2 = self.upconv_block(32 + 16, 16)
        self.upconv1 = self.upconv_block(16 + 8, 8)

        # Output
        self.output = nn.Conv2d(8, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
    # Encoder
      enc1 = self.enc1(x)
      enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
      enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
      enc4 = self.enc4(nn.MaxPool2d(2)(enc3))

      # Bottleneck
      bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))

      # Decoder (make sure to resize the feature maps to match before concatenating)
      dec4 = self.upconv4(torch.cat([nn.functional.interpolate(bottleneck, size=enc4.shape[2:], mode='bilinear', align_corners=False), enc4], dim=1))
      dec3 = self.upconv3(torch.cat([nn.functional.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=False), enc3], dim=1))
      dec2 = self.upconv2(torch.cat([nn.functional.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=False), enc2], dim=1))
      dec1 = self.upconv1(torch.cat([nn.functional.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False), enc1], dim=1))
# Resize the output to the target size (128x128)
      output = torch.sigmoid(self.output(dec1))
      output_resized = nn.functional.interpolate(output, size=(128, 128), mode='bilinear', align_corners=False)

      return output_resized
# Create model instance

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)


    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out



class AttentionUNet(nn.Module):

    def __init__(self, img_ch=1, output_ch=1):
        super(AttentionUNet, self).__init__()

        self.name = "ATT_UNET"

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 8)
        self.Conv2 = ConvBlock(8, 16)
        self.Conv3 = ConvBlock(16, 32)
        self.Conv4 = ConvBlock(32, 64)
        #self.Conv5 = ConvBlock(512, 1024)

        #self.Up5 = UpConv(1024, 512)
        #self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        #self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(64, 32)
        self.Att4 = AttentionBlock(F_g=32, F_l=32, n_coefficients=16)
        self.UpConv4 = ConvBlock(64, 32)

        self.Up3 = UpConv(32, 16)
        self.Att3 = AttentionBlock(F_g=16, F_l=16, n_coefficients=8)
        self.UpConv3 = ConvBlock(32, 16)

        self.Up2 = UpConv(16, 8)
        self.Att2 = AttentionBlock(F_g=8, F_l=8, n_coefficients=4)
        self.UpConv2 = ConvBlock(16, 8)

        self.Conv = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)


        d4 = self.Up4(e4)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)

        # Resize the output to the target size(128 x128)
        #output = torch.sigmoid(out)
        #output_resized = nn.functional.interpolate(output, size=(128, 128), mode='bilinear', align_corners=False)

        attention_mask = {'att1': s1, 'att2': s2, 'att3': s3}

        return out, attention_mask

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        out, _ = self.model(x)  # Only get the main output
        return out