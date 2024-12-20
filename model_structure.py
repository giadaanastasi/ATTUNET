# -*- coding: utf-8 -*-
#%% Libraries
import torch.nn as nn
import torch


#%% Attention U-Net
class AttentionUNet(nn.Module):

    def __init__(self, img_ch=1, output_ch=1):
        super(AttentionUNet, self).__init__()

        self.name = "ATT_UNET"

        # Max pooling
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional blocks
        self.Conv1 = ConvBlock(img_ch, 8)
        
        self.Conv2 = ConvBlock(8, 16)
        
        self.Conv3 = ConvBlock(16, 32)
        
        self.Conv4 = ConvBlock(32, 64)

        # Up-convolutional blocks (with attention gates)
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
        '''
        e : encoder
        d : decoder
        s : skip-connections from encoder layers to decoder layers
        '''
        
        # Encoding:
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        # Decoding:
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
        #output = torch.sigmoid(out)  

        attention_mask = {'att1': s1, 'att2': s2, 'att3': s3}

        return out, attention_mask
    
    
    
#%% Convolutional block
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        '''
        in_channels:  number of filters in the previous layer
        out_channels: number of filters in the current layer
        '''
        
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



#%% Up-convolutional block
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



#%% Attention Block
class AttentionBlock(nn.Module):
    '''
    Trainable attention block.
    '''
    
    def __init__(self, F_g, F_l, n_coefficients):
        '''
        F_g: number of feature maps (channels) in previous layer
        F_l: number of feature maps in corresponding encoder layer
        n_coefficients: number of trainable multi-dimensional attention coefficients
        '''
        
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
        '''
        gate: gating signal (from previous layer)
        skip_connection: activation from corresponding encoder layer
        '''
        
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        out = skip_connection * psi
        
        return out