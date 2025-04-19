import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes=10, time_dim=256):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.time_dim =  time_dim
        self.time_mlp = PositionalEmbedding(256, 1.0)
        self.encoder_input = DoubleConv(3, 64)
        self.encoder_downsample1 = Down(64,128)
        self.encoder_attention1 = SelfAttention(128,16)
        self.encoder_downsample2 = Down(128,256)
        self.encoder_attention2 = SelfAttention(256,8)
        self.encoder_downsample3 = Down(256, 256)
        self.encoder_attention3 = SelfAttention(256, 4)

        self.conv1 = DoubleConv(256,512)
        self.conv2 = DoubleConv(512,512)
        self.conv3 = DoubleConv(512,256)

        self.decoder_upsample1 = Up(512, 128)
        self.decoder_attention1 = SelfAttention(128, 8)
        self.decoder_upsample2 = Up(256, 64)
        self.decoder_attention2 = SelfAttention(64, 16)
        self.decoder_upsample3 = Up(128, 64)
        self.decoder_attention3 = SelfAttention(64, 32)
        self.decoder_output = nn.Conv2d(64, 3, kernel_size = 1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, t, x, y=None):
        t = self.time_mlp(t)
        if y is not None:
          t += self.label_emb(y)

        x1 = self.encoder_input(x)
        x2 = self.encoder_downsample1(x1,t)
        x2 = self.encoder_attention1(x2)
        x3 = self.encoder_downsample2(x2, t)
        x3 = self.encoder_attention2(x3)
        x4 = self.encoder_downsample3(x3, t)
        x4 = self.encoder_attention3(x4)

        x4 = self.conv1(x4)
        x4 = self.conv2(x4)
        x4 = self.conv3(x4)

        x3 = self.decoder_upsample1(x4, x3, t)
        x3 = self.decoder_attention1(x3)
        x2 = self.decoder_upsample2(x3, x2, t)
        x2 = self.decoder_attention2(x2)
        x1 = self.decoder_upsample3(x2, x1, t)
        x = self.decoder_attention3(x1)
        return self.decoder_output(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size)).to(self.device)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, mid_c = None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_c:
            mid_c = out_c
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size = 3, padding = 1, bias=False),
            nn.GroupNorm(1, mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c, out_c, kernel_size = 3, padding = 1, bias=False),
            nn.GroupNorm(1, out_c)
        )
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, in_c, residual=True),
            DoubleConv(in_c,out_c)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_c)
        )
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_c, out_c, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_c, in_c, residual=True),
            DoubleConv(in_c,out_c,in_c//2)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_c)
        )
    def forward(self, x, skip_x, t):
        x= self.up(x)
        x = torch.cat([skip_x,x],dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        return x + emb

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.channels, self.size = channels, size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = x.view(-1, self.channels, self.size*self.size).swapaxes(1,2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln,x_ln,x_ln)
        attention_value = attention_value * x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2,1).view(-1, self.channels, self.size, self.size)
