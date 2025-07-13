import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#################################################################################
#                                       AE                                      #
#################################################################################
class AE(nn.Module):
    def __init__(self, input_width=3, output_emb_width=4, width=512, depth=3, ch_mult=(1,1,1)):
        super().__init__()
        self.output_emb_width = output_emb_width
        self.encoder = Encoder(input_width, output_emb_width, width, depth, in_ch_mult=ch_mult[:-1], ch_mult=ch_mult[1:])
        self.decoder = Decoder(input_width, output_emb_width, width, depth, in_ch_mult=ch_mult[::-1][1:], ch_mult=ch_mult[::-1][:-1])

    def preprocess(self, x):
        x = x.permute(0, 3, 1, 2).float()
        return x

    def encode(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        return x_encoder

    def forward(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_out = self.decoder(x_encoder)
        return x_out

    def decode(self, x):
        x_out = self.decoder(x)
        return x_out

#################################################################################
#                                       VAE                                     #
#################################################################################
class VAE(nn.Module):
    def __init__(self, input_width=3, output_emb_width=4, width=512, depth=3, ch_mult=(1,1,1)):
        super().__init__()
        self.output_emb_width = output_emb_width
        self.encoder = Encoder(input_width, output_emb_width*2, width, depth, in_ch_mult=ch_mult[:-1], ch_mult=ch_mult[1:])
        self.decoder = Decoder(input_width, output_emb_width, width, depth, in_ch_mult=ch_mult[::-1][1:], ch_mult=ch_mult[::-1][:-1])

    def preprocess(self, x):
        x = x.permute(0, 3, 1, 2).float()
        return x

    def encode(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = DiagonalGaussianDistribution(x_encoder)
        x_encoder = x_encoder.sample()
        return x_encoder

    def forward(self, x, need_loss=False):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = DiagonalGaussianDistribution(x_encoder)
        kl_loss = x_encoder.kl()
        x_encoder = x_encoder.sample()
        x_out = self.decoder(x_encoder)
        if need_loss:
            # sigma vae for better quality
            log_sigma = ((x - x_out) ** 2).mean([1,2,3], keepdim=True).sqrt().log()
            log_sigma = -6 + F.softplus(log_sigma - (-6))
            rec = 0.5 * torch.pow((x - x_out) / log_sigma.exp(), 2) + log_sigma
            rec = rec.sum(dim=(1,2,3))
            loss = {
                    "rec": rec.mean(),
                    "kl": kl_loss.mean()}
            return x_out, loss
        else:
            return x_out

    def decode(self, x):
        x_out = self.decoder(x)
        return x_out

#################################################################################
#                                     AE Zoos                                   #
#################################################################################
def ae(**kwargs):
    return AE(output_emb_width=4, width=512, depth=3, ch_mult=(1,1,1), **kwargs)
def vae(**kwargs):
    return VAE(output_emb_width=4, width=512, depth=3, ch_mult=(1,1,1), **kwargs)
AE_models = {
    'AE_Model': ae, 'VAE_Model': vae
}
#################################################################################
#                                 Inner Architectures                           #
#################################################################################
class Encoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=4, width=512, depth=3, in_ch_mult=(1,1), ch_mult=(1,1)):
        super().__init__()
        self.model = nn.ModuleList()
        self.conv_in = nn.Conv2d(input_emb_width, width, (3, 1), (1, 1), (1, 1))

        block_in = width * in_ch_mult[0]
        for i in range(len(in_ch_mult)):
            block_in = width * in_ch_mult[i]
            block_out = width * ch_mult[i]
            self.model.append(nn.Conv2d(width, width, (4, 1), (2, 1), (1, 1)))
            for j in range(depth):
                self.model.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dil=2-j))
                block_in = block_out

        self.conv_out = torch.nn.Conv2d(block_in, output_emb_width, (3, 1), (1, 1), (1, 1))
    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.model:
                x = layer(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=4, width=512, depth=3, in_ch_mult=(1,1), ch_mult=(1,1)):
        super().__init__()
        self.model = nn.ModuleList()
        block_in = width * ch_mult[0]
        self.conv_in = nn.Conv2d(output_emb_width, block_in, (3,1), (1,1), (1,1))

        for i in range(len(in_ch_mult)):
            block_in = width * ch_mult[i]
            block_out = width * in_ch_mult[i]
            for j in range(depth):
                self.model.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dil=2-j))
                block_in = block_out
            self.model.append(Upsample(block_in))

        self.conv_out1 = torch.nn.Conv2d(block_in, block_in, (3, 1), (1,1), (1,1))
        self.conv_out2 = torch.nn.Conv2d(block_in, input_emb_width, (3, 1), (1, 1), (1, 1))

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.model:
            x = layer(x)
        x = self.conv_out1(x)
        x = x * torch.sigmoid(x)
        x = self.conv_out2(x)
        return x.permute(0,2,3,1)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels,(3, 1), (1, 1), (1, 1))

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=(2.0, 1.0), mode="nearest")
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dil=0, conv_shortcut=False, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=(3, 1),
                                     stride=(1, 1),
                                     padding=(3 ** dil, 0),
                                     dilation=(3 ** dil, 1),
                                     )
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=(0, 0),
                                    )

    def forward(self, x):
        h = x
        h = h*torch.sigmoid(h)
        h = self.conv1(h)

        h = h*torch.sigmoid(h)
        h = self.conv2(h)
        h = self.dropout(h)
        return x+h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
