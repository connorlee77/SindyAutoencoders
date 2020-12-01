import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision 

import numpy as np 
import matplotlib.pyplot as plt 

def SindyLibrary(z, latent_dim, poly_order, include_sine=False, device='cuda:0'):
    B, C = z.shape
    library = [torch.ones(B).to(device)]

    z_combined = z
    for i in range(latent_dim):
        library.append(z_combined[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(z_combined[:,i]*z_combined[:,j])

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p]*z_combined[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z_combined[:,i]))

    stacked_library = torch.stack(library, dim=1)
    return stacked_library



def SindyLibrary2ndOrder(z, dz, latent_dim, poly_order, include_sine=False, device='cuda:0'):
    B, C = z.shape
    library = [torch.ones(B).to(device)]

    z_combined = torch.cat([z, dz], dim=1)

    for i in range(2*latent_dim):
        library.append(z_combined[:,i])

    if poly_order > 1:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                library.append(z_combined[:,i]*z_combined[:,j])

    if poly_order > 2:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k])

    if poly_order > 3:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p])

    if poly_order > 4:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        for q in range(p,2*latent_dim):
                            library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p]*z_combined[:,q])

    if include_sine:
        for i in range(2*latent_dim):
            library.append(torch.sin(z_combined[:,i]))

    for l in library:
        print(l.shape)

    stacked_library = torch.stack(library, dim=1)
    return stacked_library

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SingleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        )

        self.conv = SingleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class CNNAutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(CNNAutoEncoder, self).__init__()

        self.inc = SingleConv(1, 4)
        self.down1 = Down(4, 8)
        self.down2 = Down(8, latent_dim)

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.up1 = Up(latent_dim, 8)
        self.up2 = Up(8, 4)
        self.outc = nn.Conv2d(4, 1, kernel_size=1)

        self.encoder = nn.Sequential(
            self.inc,
            nn.ReLU(inplace=True),
            self.down1,
            nn.ReLU(inplace=True),
            self.down2
        )

        self.decoder = nn.Sequential(
            self.up1,
            nn.ReLU(inplace=True),
            self.up2,
            nn.ReLU(inplace=True),
            self.outc
        )


    def forward(self, x):
        encoded = self.encoder(x)
        
        B, C, H, W = encoded.shape 
        z = self.gap(encoded)

        upsampled_z = F.upsample(z, size=(H,W))
        decoded = self.decoder(upsampled_z)
        return z, decoded, encoded

if __name__ == "__main__":
    
    cnn_ae = CNNAutoEncoder(latent_dim=10)
    cnn_ae.to('cuda:0')

    img = torch.zeros(1, 1, 31, 31).to('cuda:0')
    e, d = cnn_ae(img)
    print(d.shape, e.shape)
    SindyLibrary(torch.ones((5,1)), torch.ones(5,1), latent_dim=1, poly_order=2, include_sine=False)

