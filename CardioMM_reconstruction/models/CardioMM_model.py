"""
CardioMM model with CSM estimation using SEMnetwork - pytorch - (TextCond at the last part of each UNet level and lightweight prompt) - two lmhead for metadata and undersampling
- Modified KspaceACSExtractor for 1D and 2D undersampling
- Modified Data consistency
Created on 2025/04/28
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.

Some codes are modified based on https://arxiv.org/abs/2309.13839
"""


import torch
import torch.nn as nn
import fastmri
import math
import torch.nn.functional as F
from fastmri.data import transforms
from typing import (
    List,
    Optional,
    Tuple,
)
from utils import KspaceACSExtractor


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    """
    Create a same-padding 2D convolution used throughout the model blocks.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        kernel_size: Convolution kernel size. Padding is set to keep spatial size.
        bias: Whether to include a bias term.
        stride: Convolution stride.

    Returns:
        ``nn.Conv2d`` layer.
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


class CALayer(nn.Module):
    """
    Channel-attention layer.

    The layer squeezes spatial information with global average pooling and uses a
    small bottleneck MLP implemented as 1x1 convolutions to predict per-channel
    gates.
    """

    def __init__(self, channel, reduction=16, bias=False):
        """
        Args:
            channel: Number of input/output feature channels.
            reduction: Bottleneck reduction factor for channel attention.
            bias: Whether attention 1x1 convolutions include bias.
        """
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Apply channel attention to a feature map.

        Args:
            x: Feature tensor shaped ``[batch, channels, height, width]``.

        Returns:
            Feature tensor with the same shape as ``x``.
        """
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    """
    Channel-attention block with a residual connection.

    The block applies two same-padding convolutions with an activation in
    between, optionally followed by channel attention, then adds the input.
    """

    def __init__(self, n_feat, kernel_size, reduction, bias, act, no_use_ca=False):
        """
        Args:
            n_feat: Number of feature channels.
            kernel_size: Kernel size for the two convolution layers.
            reduction: Channel-attention reduction factor.
            bias: Whether convolutions include bias.
            act: Activation module inserted between convolutions.
            no_use_ca: If true, replace channel attention with identity.
        """
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        if not no_use_ca:
            self.CA = CALayer(n_feat, reduction, bias=bias)
        else:
            self.CA = nn.Identity()
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        """
        Run the residual channel-attention block.

        Args:
            x: Feature tensor shaped ``[batch, channels, height, width]``.

        Returns:
            Tensor with the same shape as ``x``.
        """
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
# ---------- Text Metadata Condition Block -----------------------
class TextConditionBlock(nn.Module):
    """
    Text-conditioned feature modulation block.

    Modified from the Instruction Condition Block (ICB), Paper Section 3.3.
    The block turns a language embedding into channel gates and applies learned
    feature scaling before a CAB refinement.
    """
    def __init__(self, feature_dim, kernel_size, reduction, text_dim=256):
        """
        Args:
            feature_dim: Number of image feature channels.
            kernel_size: CAB convolution kernel size.
            reduction: CAB channel-attention reduction factor.
            text_dim: Dimension of the projected language embedding.
        """
        super(TextConditionBlock, self).__init__()
        self.fc    = nn.Linear(text_dim, feature_dim)
        self.block = CAB(feature_dim, kernel_size, reduction, bias=False, act=nn.PReLU(), no_use_ca=False)
        self.beta  = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)

    def forward(self, x, lm_embd_adapt):
        """
        Modulate features with metadata text embeddings.

        Args:
            x: Feature tensor shaped ``[batch * coils, channels, height, width]``.
            lm_embd_adapt: Metadata embedding shaped like
                ``[batch_or_slice, 1, text_dim]``.

        Returns:
            Text-conditioned feature tensor with the same shape as ``x``.
        """
        B, _, _, _ = x.shape
        BZreal, _, _ = lm_embd_adapt.shape
        ncoil = int(B / BZreal)

        # Broadcast one text gate per real batch item across its coil-expanded features.
        gating_factors = torch.sigmoid(self.fc(lm_embd_adapt.squeeze(1)))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1).repeat(ncoil, 1, 1, 1)  # [B, C, 1, 1]

        f = x * self.gamma + self.beta  # 1) Learned feature scaling/modulation
        f = f * gating_factors          # 2) (soft) Feature routing based on text
        f = self.block(f)               # 3) Block feature enhancement

        return f + x  # [B, C, H, W] prompt output x


# ---------- Text Undersampling Prompt Block (lightweight without data input and with undersampling only) -----------------------
class USPromptBlock_lightweight(nn.Module):
    """
    Lightweight undersampling-prompt generator.

    The module keeps a small bank of prompt tensors and uses the undersampling
    text embedding to blend them into a spatial prompt for a decoder level.
    """

    def __init__(self, prompt_dim=128, prompt_len=3, prompt_size=96, lin_dim=192, text_dim=256, learnable_input_prompt=False):
        """
        Args:
            prompt_dim: Number of channels in each prompt component.
            prompt_len: Number of prompt components in the prompt bank.
            prompt_size: Spatial size of the learned prompt bank.
            lin_dim: Unused compatibility argument retained in the signature.
            text_dim: Dimension of the projected undersampling text embedding.
            learnable_input_prompt: Whether the prompt bank is trainable.
        """
        super(USPromptBlock_lightweight, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(
            1, prompt_len, prompt_dim, prompt_size, prompt_size), requires_grad=learnable_input_prompt)
        self.linear_layer2 = nn.Linear(text_dim, prompt_len)
        self.dec_conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, lm_embd_adapt2):
        """
        Generate a decoder prompt from undersampling text embeddings.

        Args:
            x: Decoder feature tensor shaped ``[batch * coils, channels, H, W]``.
            lm_embd_adapt2: Undersampling embedding shaped like
                ``[batch_or_slice, 1, text_dim]``.

        Returns:
            Prompt tensor shaped ``[batch * coils, prompt_dim, H, W]``.
        """

        B, C, H, W = x.shape
        BZreal, _, _ = lm_embd_adapt2.shape
        ncoil = int(B / BZreal)

        # Text weights select a soft mixture of prompt components.
        us_weights = F.softmax(self.linear_layer2(lm_embd_adapt2.squeeze(1)), dim=1)  # [BZreal, C]
        us_weights = us_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(ncoil, 1, 1, 1, 1)

        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = us_weights * prompt_param
        prompt = torch.sum(prompt, dim=1)

        # Match the current decoder spatial resolution before fusion.
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.dec_conv3x3(prompt)
        return prompt


class DownBlock(nn.Module):
    """Encoder block that applies CABs and downsamples by a stride-2 convolution."""

    def __init__(self, input_channel, output_channel, n_cab, kernel_size, reduction, bias, act, no_use_ca=False, first_act=False):
        """
        Args:
            input_channel: Number of input feature channels.
            output_channel: Number of output channels after downsampling.
            n_cab: Number of CABs in the encoder body.
            kernel_size: CAB convolution kernel size.
            reduction: CAB channel-attention reduction factor.
            bias: Whether convolutions include bias.
            act: Activation module for CABs.
            no_use_ca: If true, disable channel attention inside CABs.
            first_act: If true, use PReLU in the first CAB.
        """
        super(DownBlock, self).__init__()
        if first_act:
            self.encoder = [CAB(input_channel, kernel_size, reduction, bias=bias, act=nn.PReLU(), no_use_ca=no_use_ca)]
            self.encoder = nn.Sequential(*(self.encoder+[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab-1)]))
        else:
            self.encoder = nn.Sequential(
                *[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])
        self.down = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        """
        Encode features and return both downsampled and skip tensors.

        Args:
            x: Feature tensor shaped ``[batch, channels, height, width]``.

        Returns:
            Tuple ``(x_down, enc)`` where ``enc`` is the skip feature.
        """
        enc = self.encoder(x)
        x = self.down(enc)
        return x, enc


class UpBlock(nn.Module):
    """Decoder block that fuses an undersampling prompt and a skip connection."""

    def __init__(self, in_dim, out_dim, prompt_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False):
        """
        Args:
            in_dim: Number of decoder input channels.
            out_dim: Number of channels after upsampling.
            prompt_dim: Number of prompt channels concatenated with decoder input.
            n_cab: Number of CABs used for prompt fusion.
            kernel_size: CAB convolution kernel size.
            reduction: CAB channel-attention reduction factor.
            bias: Whether convolutions include bias.
            act: Activation module for CABs.
            no_use_ca: If true, disable channel attention inside CABs.
        """
        super(UpBlock, self).__init__()

        self.fuse = nn.Sequential(*[CAB(in_dim+prompt_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])
        self.reduce = nn.Conv2d(in_dim+prompt_dim, in_dim, kernel_size=1, bias=bias)

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False))

        self.ca = CAB(out_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)

    def forward(self,x,prompt_dec,skip):
        """
        Fuse prompt, upsample features, and add the encoder skip connection.

        Args:
            x: Decoder feature tensor.
            prompt_dec: Prompt tensor from ``USPromptBlock_lightweight``.
            skip: Matching encoder skip tensor.

        Returns:
            Refined decoder feature tensor.
        """

        # Concatenate the learned undersampling prompt before decoder fusion.
        x = torch.cat([x, prompt_dec], dim=1)
        x = self.fuse(x)
        x = self.reduce(x)

        # Upsampled decoder features are aligned with and added to the skip path.
        x = self.up(x) + skip
        x = self.ca(x)

        return x


class SkipBlock(nn.Module):
    """Optional skip-path refinement before decoder addition."""

    def __init__(self, enc_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False):
        """
        Args:
            enc_dim: Number of skip feature channels.
            n_cab: Number of CABs to apply; zero uses identity.
            kernel_size: CAB convolution kernel size.
            reduction: CAB channel-attention reduction factor.
            bias: Whether convolutions include bias.
            act: Activation module for CABs.
            no_use_ca: If true, disable channel attention inside CABs.
        """
        super(SkipBlock, self).__init__()
        if n_cab == 0:
            self.skip_attn = nn.Identity()
        else:
            self.skip_attn = nn.Sequential(
                *[CAB(enc_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])

    def forward(self, x):
        """Refine or pass through an encoder skip feature tensor."""
        x = self.skip_attn(x)

        return x


class CardioMMUnet(nn.Module):
    """
    Text-conditioned U-Net backbone used inside CardioMM cascades.

    The network processes real/imaginary channels as standard 2D feature maps,
    injects undersampling prompts during decoding, and applies metadata text
    conditioning after each decoder level.
    """

    def __init__(self,
                 in_chans=10,
                 out_chans=10,
                 n_feat0=48,
                 feature_dim = [72, 96, 120],
                 prompt_dim = [24, 48, 72],
                 len_prompt = [3, 3, 3],
                 prompt_size = [64, 32, 16],
                 n_enc_cab = [2, 3, 3],
                 n_dec_cab = [2, 2, 3],
                 n_skip_cab = [1, 1, 1],
                 n_bottleneck_cab = 3,
                 no_use_ca = False,
                 learnable_input_prompt=False,
                 kernel_size=3,
                 reduction=4,
                 act=nn.PReLU(),
                 bias=False,
                 llm_embd_dim = 256,
                 ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            n_feat0: Number of output channels in the first convolution layer.
            feature_dim: Number of output channels in each level of the encoder.
            prompt_dim: Number of channels in the prompt at each level of the decoder.
            len_prompt: number of components in the prompt at each level of the decoder.
            prompt_size: spatial size of the prompt at each level of the decoder.
            n_enc_cab: number of channel attention blocks (CAB) in each level of the encoder.
            n_dec_cab: number of channel attention blocks (CAB) in each level of the decoder.
            n_skip_cab: number of channel attention blocks (CAB) in each skip connection.
            n_bottleneck_cab: number of channel attention blocks (CAB) in the bottleneck.
            kernel_size: kernel size of the convolution layers.
            reduction: reduction factor for the channel attention blocks (CAB).
            act: activation function.
            bias: whether to use bias in the convolution layers.
            no_use_ca: whether to *not* use channel attention blocks (CAB).
            learnable_input_prompt: whether to learn the input prompt in the PromptBlock.
            llm_embd_dim: Dimension of the text embeddings provided by the
                language projection heads.
        """
        super(CardioMMUnet, self).__init__()

        # Feature extraction
        self.feat_extract = conv(in_chans, n_feat0, kernel_size, bias=bias)

        # Encoder - 3 DownBlocks
        self.enc_level1 = DownBlock(n_feat0, feature_dim[0], n_enc_cab[0], kernel_size, reduction, bias, act, no_use_ca=no_use_ca, first_act=True)
        self.enc_level2 = DownBlock(feature_dim[0], feature_dim[1], n_enc_cab[1], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)
        self.enc_level3 = DownBlock(feature_dim[1], feature_dim[2], n_enc_cab[2], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)

        # Skip Connections - 3 SkipBlocks
        self.skip_attn1 = SkipBlock(n_feat0, n_skip_cab[0], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
        self.skip_attn2 = SkipBlock(feature_dim[0], n_skip_cab[1], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
        self.skip_attn3 = SkipBlock(feature_dim[1], n_skip_cab[2], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[CAB(feature_dim[2], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_bottleneck_cab)])

        # Decoder - 3 UpBlocks
        self.prompt_level3 = USPromptBlock_lightweight(prompt_dim=prompt_dim[2], prompt_len=len_prompt[2], prompt_size=prompt_size[2], lin_dim=feature_dim[2], text_dim=llm_embd_dim, learnable_input_prompt=learnable_input_prompt)
        self.dec_level3 = UpBlock(feature_dim[2], feature_dim[1], prompt_dim[2], n_dec_cab[2], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)
        self.metacond_level3 = TextConditionBlock(feature_dim[1], kernel_size=kernel_size, reduction=reduction, text_dim=llm_embd_dim)

        self.prompt_level2 = USPromptBlock_lightweight(prompt_dim=prompt_dim[1], prompt_len=len_prompt[1], prompt_size=prompt_size[1], lin_dim=feature_dim[1], text_dim=llm_embd_dim, learnable_input_prompt=learnable_input_prompt)
        self.dec_level2 = UpBlock(feature_dim[1], feature_dim[0], prompt_dim[1], n_dec_cab[1], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)
        self.metacond_level2 = TextConditionBlock(feature_dim[0], kernel_size=kernel_size, reduction=reduction, text_dim=llm_embd_dim)

        self.prompt_level1 = USPromptBlock_lightweight(prompt_dim=prompt_dim[0], prompt_len=len_prompt[0], prompt_size=prompt_size[0], lin_dim=feature_dim[0], text_dim=llm_embd_dim, learnable_input_prompt=learnable_input_prompt)
        self.dec_level1 = UpBlock(feature_dim[0], n_feat0, prompt_dim[0], n_dec_cab[0], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)
        self.metacond_level1 = TextConditionBlock(n_feat0, kernel_size=kernel_size, reduction=reduction, text_dim=llm_embd_dim)

        # OutConv
        self.conv_last = conv(n_feat0, out_chans, 5, bias=bias)

    def forward(self, x, lm_embd_adapt, lm_embd_adapt2):
        """
        Run the U-Net with metadata and undersampling text conditioning.

        Args:
            x: Channelized input tensor shaped ``[batch, channels, height, width]``.
            lm_embd_adapt: Metadata embedding used by ``TextConditionBlock``.
            lm_embd_adapt2: Undersampling embedding used by prompt blocks.

        Returns:
            Output tensor shaped ``[batch, out_chans, height, width]``.
        """
        # 0. featue extraction
        x = self.feat_extract(x)

        # 1. encoder
        x, enc1 = self.enc_level1(x)
        x, enc2 = self.enc_level2(x)
        x, enc3 = self.enc_level3(x)

        # 2. bottleneck
        x = self.bottleneck(x)

        # 3. decoder
        # Each decoder level receives an undersampling prompt and then metadata conditioning.
        dec_usprompt3 = self.prompt_level3(x, lm_embd_adapt2)
        x = self.dec_level3(x, dec_usprompt3, self.skip_attn3(enc3))
        dec_metacond3 = self.metacond_level3(x, lm_embd_adapt)

        dec_usprompt2 = self.prompt_level2(dec_metacond3, lm_embd_adapt2)
        x = self.dec_level2(dec_metacond3, dec_usprompt2, self.skip_attn2(enc2))
        dec_metacond2 = self.metacond_level2(x, lm_embd_adapt)

        dec_usprompt1 = self.prompt_level1(dec_metacond2, lm_embd_adapt2)
        x = self.dec_level1(dec_metacond2, dec_usprompt1, self.skip_attn1(enc1))
        dec_metacond1 = self.metacond_level1(x, lm_embd_adapt)

        # 4. last conv
        return self.conv_last(dec_metacond1)


class NormCardioMMUnet(nn.Module):
    """
    Normalization and complex-channel wrapper around ``CardioMMUnet``.

    fastMRI tensors store complex numbers in the final dimension of size 2.
    This wrapper converts that representation to channel dimensions for the
    U-Net, normalizes per example, pads spatial dimensions to multiples of 8,
    and converts back to complex-last representation afterward.
    """

    def __init__(
        self,
        in_chans: int = 10,
        out_chans: int = 10,
        n_feat0: int = 48,
        feature_dim: List[int] = [72, 96, 120],
        prompt_dim: List[int] = [24, 48, 72],
        len_prompt: List[int] = [3, 3, 3],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        learnable_input_prompt=False,
        llm_embd_dim: int = 256,
    ):
        """
        Args:
            in_chans: Number of channelized input channels.
            out_chans: Number of channelized output channels.
            n_feat0: Number of top-level feature channels.
            feature_dim: Encoder/decoder feature dimensions.
            prompt_dim: Prompt feature dimensions at decoder levels.
            len_prompt: Number of prompt components per decoder level.
            prompt_size: Prompt spatial sizes per decoder level.
            n_enc_cab: Number of CABs in encoder levels.
            n_dec_cab: Number of CABs in decoder levels.
            n_skip_cab: Number of CABs on skip paths.
            n_bottleneck_cab: Number of CABs in the bottleneck.
            no_use_ca: If true, disable channel attention in CABs.
            learnable_input_prompt: Whether prompt-bank tensors are trainable.
            llm_embd_dim: Dimension of projected text embeddings.
        """

        super().__init__()
        self.unet = CardioMMUnet(in_chans=in_chans,
                                out_chans = out_chans,
                                n_feat0=n_feat0,
                                feature_dim = feature_dim,
                                prompt_dim = prompt_dim,
                                len_prompt = len_prompt,
                                prompt_size = prompt_size,
                                n_enc_cab = n_enc_cab,
                                n_dec_cab = n_dec_cab,
                                n_skip_cab = n_skip_cab,
                                n_bottleneck_cab = n_bottleneck_cab,
                                no_use_ca = no_use_ca,
                                learnable_input_prompt=learnable_input_prompt,
                                llm_embd_dim = llm_embd_dim,
                                )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Move complex real/imag components from the last dimension into channels.

        Args:
            x: Tensor shaped ``[batch, coils_or_slices, height, width, 2]``.

        Returns:
            Tensor shaped ``[batch, 2 * coils_or_slices, height, width]``.
        """
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Move channelized real/imag components back to the final complex dimension.

        Args:
            x: Tensor shaped ``[batch, 2 * channels, height, width]``.

        Returns:
            Tensor shaped ``[batch, channels, height, width, 2]``.
        """
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize each batch element by its global mean and standard deviation.

        Args:
            x: Tensor shaped ``[batch, channels, height, width]``.

        Returns:
            Tuple ``(x_norm, mean, std)`` for later unnormalization.
        """
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Undo the per-example normalization performed by ``norm``."""
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        """
        Symmetrically pad spatial dimensions to multiples of 8.

        Args:
            x: Tensor shaped ``[batch, channels, height, width]``.

        Returns:
            Padded tensor and padding metadata used by ``unpad``.
        """
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 7) + 1
        h_mult = ((h - 1) | 7) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        """Remove spatial padding added by ``pad``."""
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor, lm_embd_adapt: torch.Tensor, lm_embd_adapt2: torch.Tensor) -> torch.Tensor:
        """
        Normalize, run the U-Net, and restore complex-last tensor layout.

        Args:
            x: Complex-last tensor shaped ``[batch, channels, height, width, 2]``.
            lm_embd_adapt: Metadata text embedding.
            lm_embd_adapt2: Undersampling text embedding.

        Returns:
            Complex-last tensor with the same spatial shape as ``x``.
        """
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x, lm_embd_adapt, lm_embd_adapt2)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class CardioMMBlock(nn.Module):
    """
    Model block for end-to-end CardioMM model.

    This model applies a combination of modified data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full CardioMM model.
    """

    def __init__(self, model: nn.Module, num_adj_slices=5):
        """
        Args:
            model: Module for "regularization" component of CardioMM model.
            num_adj_slices: Number of adjacent slices grouped in the channel dimension.
        """
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Expand slice images to multi-coil k-space using sensitivity maps.

        Args:
            x: Image tensor in complex-last format.
            sens_maps: Coil sensitivity maps in complex-last format.

        Returns:
            Multi-coil k-space tensor in complex-last format.
        """
        _, c, _, _, _ = sens_maps.shape
        # Repeat adjacent-slice images across coils before multiplying by sensitivities.
        return fastmri.fft2c(fastmri.complex_mul(x.repeat_interleave(c // self.num_adj_slices, dim=1), sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Reduce multi-coil k-space to slice images using sensitivity maps.

        Args:
            x: Multi-coil k-space tensor in complex-last format.
            sens_maps: Coil sensitivity maps in complex-last format.

        Returns:
            Coil-combined image tensor with adjacent slices in the channel dimension.
        """
        b, c, h, w, _ = x.shape
        x = fastmri.ifft2c(x)
        # Multiply by conjugate sensitivities and sum over coils within each adjacent slice.
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).view(b, self.num_adj_slices, c // self.num_adj_slices, h, w, 2).sum(
            dim=2, keepdim=False
        )

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        lm_embd_adapt: torch.Tensor,
        lm_embd_adapt2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply one model regularization and data-consistency cascade.

        Args:
            current_kspace: Current cascade estimate in multi-coil k-space.
            ref_kspace: Originally sampled k-space used for data consistency.
            mask: Boolean sampling mask.
            sens_maps: Coil sensitivity maps.
            lm_embd_adapt: Metadata text embedding.
            lm_embd_adapt2: Undersampling text embedding.

        Returns:
            Updated multi-coil k-space tensor.
        """

        # Data_consistency_PISFC: https://doi.org/10.1016/j.media.2025.103616
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps), lm_embd_adapt, lm_embd_adapt2), sens_maps
        )  # model_term: multi-coil kspace
        # Sampled locations blend the measured k-space and model prediction with a learned weight.
        k_sample = mask * (torch.divide((ref_kspace + torch.abs(self.dc_weight) * model_term), (1 + torch.abs(self.dc_weight))))
        # Unsampled locations rely entirely on the model-predicted k-space.
        k_no_sample = (~mask) * model_term
        k_dc = k_sample + k_no_sample

        return k_dc  # model_term: multi-coil kspace


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end CardioMM model. The optional ACS extractor restricts the input to
    central k-space before sensitivity estimation.
    """

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        num_adj_slices: int = 1,
        n_feat0: int = 24,
        feature_dim: List[int] = [36, 48, 60],
        prompt_dim: List[int] = [12, 24, 36],
        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        mask_center: bool = True,
        low_mem: bool = False,
        learnable_input_prompt=False,
        llm_embd_dim: int = 256,
    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for CardioMMUnet.
            feature_dim: feature dim for each level in PromptUnet.
            prompt_dim: prompt dim for each level in PromptUnet.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in
                BottleneckBlock.
            no_use_ca: not using channel attention.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
            low_mem: Whether to run the sensitivity U-Net one coil image at a
                time to reduce memory use.
            learnable_input_prompt: Whether sensitivity prompt-bank tensors are trainable.
            llm_embd_dim: Dimension of projected text embeddings.
        """
        super().__init__()
        self.mask_center = mask_center
        self.num_adj_slices = num_adj_slices
        self.low_mem = low_mem
        self.norm_unet = NormCardioMMUnet(in_chans=in_chans,
                                out_chans = out_chans,
                                n_feat0=n_feat0,
                                feature_dim = feature_dim,
                                prompt_dim = prompt_dim,
                                len_prompt = len_prompt,
                                prompt_size = prompt_size,
                                n_enc_cab = n_enc_cab,
                                n_dec_cab = n_dec_cab,
                                n_skip_cab = n_skip_cab,
                                n_bottleneck_cab = n_bottleneck_cab,
                                no_use_ca = no_use_ca,
                                learnable_input_prompt = learnable_input_prompt,
                                llm_embd_dim = llm_embd_dim,
                                )
        self.kspace_acs_extractor = KspaceACSExtractor(mask_center)

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Fold coil channels into the batch dimension for per-coil processing.

        Args:
            x: Tensor shaped ``[batch, coils, height, width, 2]``.

        Returns:
            Tuple ``(x_reshaped, batch_size)`` where ``x_reshaped`` is shaped
            ``[batch * coils, 1, height, width, 2]``.
        """
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Restore coil channels from the folded batch dimension.

        Args:
            x: Tensor shaped ``[batch * coils, 1, height, width, 2]``.
            batch_size: Original batch size returned by ``chans_to_batch_dim``.

        Returns:
            Tensor shaped ``[batch, coils, height, width, 2]``.
        """
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize sensitivity maps by root-sum-of-squares coil magnitude.

        Args:
            x: Coil sensitivity tensor shaped
                ``[batch, adjacent_slices * coils, height, width, 2]``.

        Returns:
            Sensitivity tensor with the same shape as ``x``.
        """
        b, adj_coil, h, w, two = x.shape
        coil = adj_coil//self.num_adj_slices
        x = x.view(b, self.num_adj_slices, coil, h, w, two)
        # Normalize over the coil dimension separately for each adjacent slice.
        x = x / fastmri.rss_complex(x, dim=2).unsqueeze(-1).unsqueeze(2)

        return x.view(b, adj_coil, h, w, two)

    def compute_sens(self, model: nn.Module, images: torch.Tensor, compute_per_coil: bool,
                     lm_embd_adapt: torch.Tensor, lm_embd_adapt2: torch.Tensor) -> torch.Tensor:
        """
        Run the sensitivity U-Net on coil images.

        Args:
            model: Normalized CardioMM U-Net used for sensitivity estimation.
            images: Coil images with coils folded into the batch dimension.
            compute_per_coil: If true, process one folded coil image at a time.
            lm_embd_adapt: Metadata text embedding.
            lm_embd_adapt2: Undersampling text embedding.

        Returns:
            Estimated sensitivity images with the same folded batch layout.
        """
        # batch_size * n_coils
        bc = images.shape[0]
        if compute_per_coil:
            # Low-memory mode avoids sending all coil images through the U-Net at once.
            output = []
            for i in range(bc):
                output.append(model(images[i].unsqueeze(0), lm_embd_adapt, lm_embd_adapt2))
            output = torch.cat(output, dim=0)
        else:
            output = model(images, lm_embd_adapt, lm_embd_adapt2)
        return output

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
        lm_embd_adapt: Optional[torch.Tensor] = None,
        lm_embd_adapt2: Optional[torch.Tensor] = None,
        mask_type: Optional[str] = "random",
    ) -> torch.Tensor:
        """
        Estimate normalized coil sensitivity maps from masked k-space.

        Args:
            masked_kspace: Undersampled multi-coil k-space in complex-last format.
            mask: Sampling mask broadcastable to ``masked_kspace``.
            num_low_frequencies: Number of ACS samples used by center masking.
            lm_embd_adapt: Metadata text embedding.
            lm_embd_adapt2: Undersampling text embedding.
            mask_type: Undersampling family, e.g. ``"uniform"``, ``"random"``,
                or ``"radial"``.

        Returns:
            Normalized sensitivity maps in complex-last format.
        """
        # Restrict to ACS samples when center masking is enabled.
        masked_kspace_acs = self.kspace_acs_extractor(masked_kspace, mask, num_low_frequencies, mask_type)
        # convert to image space
        images, batches = self.chans_to_batch_dim(
            fastmri.ifft2c(masked_kspace_acs))

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.compute_sens(self.norm_unet, images, self.low_mem, lm_embd_adapt, lm_embd_adapt2), batches)
        )


class CardioMM_SEM(nn.Module):
    """
    Unrolled multi-coil MR reconstruction model with sensitivity estimation.

    The model first estimates coil sensitivity maps, then applies a stack of
    CardioMM data-consistency cascades conditioned by metadata and undersampling
    text embeddings. The final output is the RSS image for the center adjacent
    slice.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        num_adj_slices: int = 5,
        n_feat0: int = 48,
        feature_dim: List[int] = [72, 96, 120],
        prompt_dim: List[int] = [24, 48, 72],
        sens_n_feat0: int = 24,
        sens_feature_dim: List[int] = [36, 48, 60],
        sens_prompt_dim: List[int] = [12, 24, 36],
        len_prompt: List[int] = [3, 3, 3],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        sens_len_prompt: Optional[List[int]] = None,
        sens_prompt_size: Optional[List[int]] = None,
        sens_n_enc_cab: Optional[List[int]] = None,
        sens_n_dec_cab: Optional[List[int]] = None,
        sens_n_skip_cab: Optional[List[int]] = None,
        sens_n_bottleneck_cab: Optional[List[int]] = None,
        sens_no_use_ca: Optional[bool] = None,
        use_checkpoint: bool = False,
        mask_center: bool = True,
        low_mem: bool = False,
        llm_embd_dim: int = 256,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for CardioMM model.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for CardioMMUnet.
            feature_dim: feature dim for each level in CardioMMUnet.
            prompt_dim: prompt dim for each level in CardioMMUnet.
            sens_n_feat0: Number of top-level feature channels for sense map estimation.
            sens_feature_dim: feature dim for each level in CardioMMUnet for sensitivity map estimation (SME) network.
            sens_prompt_dim: prompt dim for each level in CardioMMUnet in sensitivity map estimation (SME) network.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in BottleneckBlock.
            no_use_ca: not using channel attention.
            mask_center: Whether to mask center of k-space for sensitivity map calculation.
            use_checkpoint: Whether to use checkpointing to trade compute for GPU memory.
            low_mem: Whether to compute sensitivity map coil by coil to save GPU memory.
            llm_embd_dim: Dimension of projected metadata and undersampling text embeddings.
        """
        super().__init__()
        assert num_adj_slices % 2 == 1, "num_adj_slices must be odd"
        self.num_adj_slices = num_adj_slices
        self.center_slice = num_adj_slices//2
        # Sensitivity network can use sensitivity-specific hyperparameters when supplied.
        self.sens_net = SensitivityModel(
            num_adj_slices=num_adj_slices,
            n_feat0=sens_n_feat0,
            feature_dim=sens_feature_dim,
            prompt_dim=sens_prompt_dim,
            len_prompt=sens_len_prompt if sens_len_prompt is not None else len_prompt,
            prompt_size=sens_prompt_size if sens_prompt_size is not None else prompt_size,
            n_enc_cab=sens_n_enc_cab if sens_n_enc_cab is not None else n_enc_cab,
            n_dec_cab=sens_n_dec_cab if sens_n_dec_cab is not None else n_dec_cab,
            n_skip_cab=sens_n_skip_cab if sens_n_skip_cab is not None else n_skip_cab,
            n_bottleneck_cab=sens_n_bottleneck_cab if sens_n_bottleneck_cab is not None else n_bottleneck_cab,
            no_use_ca=sens_no_use_ca if sens_no_use_ca is not None else no_use_ca,
            mask_center=mask_center,
            low_mem=low_mem,
            learnable_input_prompt=False,
            llm_embd_dim=llm_embd_dim,
        )
        # Each cascade contains a normalized U-Net regularizer and data consistency step.
        self.cascades = nn.ModuleList(
            [CardioMMBlock(NormCardioMMUnet(2*num_adj_slices, 2*num_adj_slices, n_feat0, feature_dim, prompt_dim, len_prompt, prompt_size, n_enc_cab, n_dec_cab, n_skip_cab, n_bottleneck_cab, no_use_ca, False, llm_embd_dim), num_adj_slices) for _ in range(num_cascades)]
        )
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
        lm_embd_adapt: Optional[torch.Tensor] = None,
        lm_embd_adapt2: Optional[torch.Tensor] = None,
        mask_type: Optional[str] = "random",
    ) -> torch.Tensor:
        """
        Reconstruct an RSS image from masked multi-coil k-space.

        Args:
            masked_kspace: Input undersampled k-space in complex-last format.
            mask: Boolean sampling mask.
            num_low_frequencies: ACS width/count used for sensitivity estimation.
            lm_embd_adapt: Metadata text embedding.
            lm_embd_adapt2: Undersampling text embedding.
            mask_type: Undersampling family passed to the ACS extractor.

        Returns:
            RSS magnitude reconstruction for the center adjacent slice.
        """

        if isinstance(num_low_frequencies, int):
            pass
        else:
            # If num_low_frequencies is a tensor, convert it to CPU if necessary and get the minimum value as an integer
            if num_low_frequencies.is_cuda:
                num_low_frequencies = num_low_frequencies.cpu().min().item()
            else:
                num_low_frequencies = num_low_frequencies.min().item()

        # Estimate coil sensitivity maps once and share them across cascades.
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies, lm_embd_adapt, lm_embd_adapt2, mask_type)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps, lm_embd_adapt, lm_embd_adapt2)

        # Keep only the center slice from the adjacent-slice stack before image-domain RSS.
        kspace_pred = torch.chunk(kspace_pred, self.num_adj_slices, dim=1)[self.center_slice]

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
