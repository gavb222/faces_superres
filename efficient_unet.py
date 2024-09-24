import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, mult, emb_dim=256):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv_stack = nn.ModuleList([
            DoubleConv(in_channels, in_channels, residual=True) for _ in range(mult)
        ])
        self.out_conv = DoubleConv(in_channels, out_channels)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, e=None):
        x = self.maxpool(x)
        for conv in self.conv_stack:
            x = conv(x)
        if e is not None:
            emb = self.emb_layer(e)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            return self.out_conv(x) + emb
        else:
            return self.out_conv(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip, e=None):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        if e is not None:
            emb = self.emb_layer(e)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            return x + emb
        else:
            return x
        
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth_mult = [1,2,3,3], emb_dim=64):
        super(UNet, self).__init__()
        self.embed_dim = 64
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128, depth_mult[0], emb_dim)
        self.down2 = Down(128, 256, depth_mult[1],emb_dim)
        self.down3 = Down(256, 512, depth_mult[2],emb_dim)
        self.down4 = Down(512, 512, depth_mult[3],emb_dim)
        self.up1 = Up(1024, 256, emb_dim)
        self.up2 = Up(512, 128, emb_dim)
        self.up3 = Up(256, 64, emb_dim)
        self.up4 = Up(128, 64, emb_dim)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    #e for embedding, is the time step for the model.
    def forward(self, x, e, factor=1/math.sqrt(2)):
        e = self.pos_encoding(e, self.embed_dim)
        x1 = self.inc(x)
        x2 = self.down1(x1, e)
        x3 = self.down2(x2, e)
        x4 = self.down3(x3, e)
        x5 = self.down4(x4, e)
        x = self.up1(x5, x4*factor, e)
        x = self.up2(x, x3*factor, e)
        x = self.up3(x, x2*factor, e)
        x = self.up4(x, x1*factor, e)
        x = self.outc(x)
        return x
    

