import math
from typing import Optional, Tuple, Union, List
import numpy as np
import torch
from torch import nn
from src.sobel import Sobel, Laplacian


class Swish(nn.Module):
    """
    ### Swish actiavation function
    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb

###############################################
class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x

class CAB(nn.Module):
    def __init__(self, in_channels):
        super(CAB, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, in_channels, 3),
                      nn.InstanceNorm2d(in_channels),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, in_channels, 3),
                      nn.InstanceNorm2d(in_channels)]

        self.conv_block = nn.Sequential(*conv_block)
        act = nn.LeakyReLU(0.2)
        self.act = act
        
        self.gcnet = ContextBlock(in_channels)
    def forward(self, x):
        res = self.conv_block(x)
        res = self.act(self.gcnet(res))
        res += x
        return res

import torch
import torch.nn as nn
import torch.nn.functional as F

class DRAB(nn.Module):
    def __init__(self, in_channels):
        super(DRAB, self).__init__()
        self.conv_00 = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv_01 = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_02 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv_03 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_):
        x = self.conv_00(input_)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.conv_01(x)
        x = self.bn1(x)
        x = F.relu(x)
        y = self.conv_02(x)
        y = self.bn2(y)
        y = torch.tanh(y)
        y = self.conv_03(y)
        y = torch.sigmoid(y)
        o = y * x + x
        return o

###############################################

class ResidualBlock(nn.Module):
    """
    ### Residual block
    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 dropout: float = 0.1, is_noise: bool = True):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.is_noise = is_noise
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer

        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        if self.is_noise:
            self.time_emb = nn.Linear(time_channels, out_channels)
            self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)
        
        

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(x))
        
        # Add time embeddings
        if self.is_noise:
            h += self.time_emb(self.time_act(t))[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(h)))
        
        
        

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, is_noise: bool = True):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, is_noise=is_noise)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        return x


class UpBlock(nn.Module):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, is_noise: bool = True):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels, is_noise=is_noise)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int, is_noise: bool = True):
        super().__init__()

        self.cab = CAB(n_channels)
        self.drab = DRAB(n_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        
        x = self.cab(x)
        x = self.cab(x)
        x = self.cab(x)
        x = self.cab(x)
        x = self.cab(x)
        x = self.cab(x)
        
        #print(x.shape)
        
        x1= self.drab(x)
        x1= self.drab(x1)
        x1= self.drab(x1)
        x1= self.drab(x1)
        x1= self.drab(x1)
        x1= self.drab(x1)
        
        #print(x1.shape)
        x = x + x1

        
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, input_channels: int = 2, output_channels: int = 1, n_channels: int = 32,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 n_blocks: int = 2, is_noise: bool = True):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(input_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.is_noise = is_noise
        if is_noise:
            self.time_emb = TimeEmbedding(n_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_noise=is_noise))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, is_noise=False)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = n_channels * ch_mults[i]
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_noise=is_noise))
            # Final block to reduce the number of channels
            in_channels = n_channels * (ch_mults[i-1] if i >= 1 else 1)
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_noise=is_noise))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.act = Swish()
        self.final = nn.Conv2d(in_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor=torch.tensor([0]).cuda()):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        if self.is_noise:
            t = self.time_emb(t)
        else:
            t = None
        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)
        
        

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                # print(x.shape, s.shape)
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(x))


class DocDiff(nn.Module):
    def __init__(self, input_channels: int = 2, output_channels: int = 1, n_channels: int = 32,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 n_blocks: int = 1):
        super(DocDiff, self).__init__()
        self.denoiser = UNet(input_channels, output_channels, n_channels, ch_mults, n_blocks, is_noise=True)
        self.init_predictor = UNet(input_channels//2, output_channels, n_channels, ch_mults, n_blocks, is_noise=False)
        # self.init_predictor = UNet(input_channels, output_channels, 2 * n_channels, ch_mults, n_blocks)

    def forward(self, x, condition, t, diffusion):
        x_ = self.init_predictor(condition, t)
        residual = x - x_
        noisy_image, noise_ref = diffusion.noisy_image(t, residual)
        x__ = self.denoiser(torch.cat((noisy_image, x_.clone().detach()), dim=1), t)
        return x_, x__, noisy_image, noise_ref


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


if __name__ == '__main__':
    from src.config import load_config
    import argparse
    from schedule.diffusionSample import GaussianDiffusion
    from schedule.schedule import Schedule
    import torchsummary
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../conf.yml', help='path to the config.yaml file')
    args = parser.parse_args()
    config = load_config(args.config)
    print('Config loaded')
    model = DocDiff(input_channels=config.CHANNEL_X + config.CHANNEL_Y,
            output_channels=config.CHANNEL_Y,
            n_channels=config.MODEL_CHANNELS,
            ch_mults=config.CHANNEL_MULT,
            n_blocks=config.NUM_RESBLOCKS)
    schedule = Schedule(config.SCHEDULE, config.TIMESTEPS)
    diffusion = GaussianDiffusion(model, config.TIMESTEPS, schedule)
    model.eval()
    print(torchsummary.summary(model.init_predictor.cuda(), [(3, 128, 128)], batch_size=32))

