# Copyright contributors to the Terratorch project

"""Swin transformer implementation. Mix of MMSegmentation implementation and timm implementation.

We use this implementation instead of the original implementation or timm's.
This is because it offers a few advantages, namely being able to handle
a dynamic input size through padding.

Please note the original timm implementation can still be used as a backbone
via `timm.create_model("swin_...")`. You can see the available models with `timm.list_models("swin*")`.
"""

import collections
import math
from collections.abc import Callable
from copy import deepcopy
from itertools import repeat

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint as cp
from timm.layers import DropPath, Mlp, trunc_normal_
from timm.layers.classifier import ClassifierHead
from timm.models._manipulate import named_apply
from timm.models.vision_transformer import get_init_weights_vit
from torch import nn
from torch.nn import ModuleList


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.0,
        dropout_layer=None,
        dropout_arg=0,
        act_layer=nn.ReLU,
        add_identity=True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = act_layer

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(nn.Linear(in_channels, feedforward_channels), self.activate(), nn.Dropout(ffn_drop))
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = dropout_layer(dropout_arg) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class PatchMerging(nn.Module):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map. Our implementation uses `nn.Unfold` to
    merge patch, which is about 25% faster than original implementation.
    Instead, we need to modify pretrained models for compatibility.

    Args:
        in_chans (int): The num of input channels.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
    """

    def __init__(
        self,
        in_chans,
        out_channels,
        kernel_size=2,
        stride=None,
        padding="corner",
        dilation=1,
        bias=False,  # noqa: FBT002
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_channels = out_channels
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding
            )
            # disable the padding of unfold
            padding = 0
        else:
            self.adap_padding = None

        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_chans

        if norm_layer is not None:
            self.norm = norm_layer(sample_dim)
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape  # noqa: N806

        H, W = input_size  # noqa: N806
        if not L == H * W:
            msg = "input feature has wrong size"
            raise Exception(msg)

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility

        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]  # noqa: N806

        x = self.sampler(x)
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)

        out_h = (
            H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1) - 1
        ) // self.sampler.stride[0] + 1
        out_w = (
            W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1) - 1
        ) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size


class AdaptivePadding(nn.Module):
    r"""Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
    ```
        >>> import torch
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    ```
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding="corner", mode="constant"):
        super().__init__()

        if padding not in ("same", "corner"):
            msg = "padding must be same or corner"
            raise Exception(msg)

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.mode = mode

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, [0, pad_w, 0, pad_h], mode=self.mode)
            elif self.padding == "same":
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_chans (int): The num of input channels. Default: 3
        embed_dim (int): The dimensions of embedding. Default: 768
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int, optional): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        padding_mode (string): The padding mode to use. Default "constant".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
    """

    def __init__(
        self,
        in_chans=3,
        embed_dim=768,
        kernel_size=16,
        stride=None,
        padding="corner",
        padding_mode="constant",
        dilation=1,
        bias=True,  # noqa: FBT002
        norm_layer=nn.LayerNorm,
        input_size=None,
        drop_rate=0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        if stride is None:
            stride = kernel_size
        self.drop = nn.Dropout(drop_rate)

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, mode=padding_mode
            )
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.projection = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dim)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        x = self.drop(x)
        return x, out_size


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


class WindowMSA(nn.Module):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        window_size,
        qkv_bias=True,  # noqa: FBT002
        qk_scale=None,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dim = embed_dim // num_heads
        self.scale = qk_scale or head_embed_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size  # noqa: N806
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer("relative_position_index", rel_position_index)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape  # noqa: N806
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]  # noqa: N806
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(nn.Module):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        drop_path_rate (float, optional): Dropout ratio of layer used before output.
            Defaults: 0.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        window_size,
        shift_size=0,
        qkv_bias=True,  # noqa: FBT002
        qk_scale=None,
        attn_drop_rate=0,
        proj_drop_rate=0,
        drop_path_rate=0,
    ):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        if not 0 <= self.shift_size < self.window_size:
            msg = "0 <= self.shift_size < self.window_size condition must be true"
            raise Exception(msg)

        self.w_msa = WindowMSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
        )

        self.drop = DropPath(drop_path_rate)

    def forward(self, query, hw_shape):
        B, L, C = query.shape  # noqa: N806
        H, W = hw_shape  # noqa: N806
        if not L == H * W:
            msg = "input feature has wrong size"
            raise Exception(msg)
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]  # noqa: N806

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device, dtype=query.dtype)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(attn_mask == 0, 0.0)
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):  # noqa: N803
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image

        Returns:
            tuple: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))  # noqa: N806
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)

        Returns:
            tuple: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape  # noqa: N806
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(nn.Module):
    """
    Args:
        embed_dim (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for Mlps.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        feedforward_channels,
        window_size=7,
        shift=False,  # noqa: FBT002
        qkv_bias=True,  # noqa: FBT002
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        with_cp=False,  # noqa: FBT002
    ):
        super().__init__()

        self.with_cp = with_cp

        self.norm1 = norm_layer(embed_dim)
        self.attn = ShiftWindowMSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        self.norm2 = norm_layer(embed_dim)
        self.ffn = FFN(
            embed_dim,
            feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=DropPath,
            dropout_arg=drop_path_rate,
            act_layer=act_layer,
        )

    def forward(self, x, hw_shape):
        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinBlockSequence(nn.Module):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dim (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for Mlps.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        feedforward_channels,
        depth,
        window_size=7,
        qkv_bias=True,  # noqa: FBT002
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        downsample=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        with_cp=False,  # noqa: FBT002
    ):
        super().__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            if not len(drop_path_rates) == depth:
                msg = "drop_path_rates must have same len as depth"
                raise Exception(msg)
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                with_cp=with_cp,
            )
            self.blocks.append(block)

        self.downsample = downsample
        self.norm = norm_layer(embed_dim)

    def forward(self, x_hw_shape):
        if len(x_hw_shape) == 2:  # noqa: PLR2004
            x, hw_shape = x_hw_shape
        else:
            _, _, x, hw_shape = x_hw_shape
        for block in self.blocks:
            x = block(x, hw_shape)
        # RETURN ORDER IS IMPORTANT!! features_only=True from timm will pick the first element of the tuple to return
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return self.norm(x).view(x.shape[0], *hw_shape, -1), hw_shape, x_down, down_hw_shape
        else:
            return self.norm(x).view(x.shape[0], *hw_shape, -1), hw_shape, x, hw_shape


class MMSegSwinTransformer(nn.Module):
    def __init__(
        self,
        pretrain_img_size=224,
        in_chans=3,
        embed_dim=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        strides=(4, 2, 2, 2),
        num_classes: int = 1000,
        global_pool: str = "avg",
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,  # noqa: FBT002
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        with_cp=False,  # noqa: FBT002
        frozen_stages=-1,
    ):
        """MMSeg Swin Transformer backbone.

        This backbone is the implementation of `Swin Transformer:
        Hierarchical Vision Transformer using Shifted
        Windows <https://arxiv.org/abs/2103.14030>`_.
        Inspiration from https://github.com/microsoft/Swin-Transformer.

        Args:
            pretrain_img_size (int | tuple[int]): The size of input image when
                pretrain. Defaults: 224.
            in_chans (int): The num of input channels.
                Defaults: 3.
            embed_dim (int): The feature dimension. Default: 96.
            patch_size (int | tuple[int]): Patch size. Default: 4.
            window_size (int): Window size. Default: 7.
            mlp_ratio (int | float): Ratio of mlp hidden dim to embedding dim.
                Default: 4.
            depths (tuple[int]): Depths of each Swin Transformer stage.
                Default: (2, 2, 6, 2).
            num_heads (tuple[int]): Parallel attention heads of each Swin
                Transformer stage. Default: (3, 6, 12, 24).
            strides (tuple[int]): The patch merging or patch embedding stride of
                each Swin Transformer stage. (In swin, we set kernel size equal to
                stride.) Default: (4, 2, 2, 2).
            out_indices (tuple[int]): Output from which stages.
                Default: (0, 1, 2, 3).
            qkv_bias (bool, optional): If True, add a learnable bias to query, key,
                value. Default: True
            qk_scale (float | None, optional): Override default qk scale of
                head_dim ** -0.5 if set. Default: None.
            patch_norm (bool): If add a norm layer for patch embed and patch
                merging. Default: True.
            drop_rate (float): Dropout rate. Defaults: 0.
            attn_drop_rate (float): Attention dropout rate. Default: 0.
            drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
            act_layer (dict): activation layer.
                Default: nn.GELU.
            norm_layer (dict): normalization layer at
                output of backone. Defaults: nn.LayerNorm.
            with_cp (bool, optional): Use checkpoint or not. Using checkpoint
                will save some memory while slowing down the training speed.
                Default: False.
            frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
                -1 means not freezing any parameters.
        """

        self.frozen_stages = frozen_stages
        self.output_fmt = "NHWC"
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            if not len(pretrain_img_size) == 2:  # noqa: PLR2004
                msg = f"The size of image should have length 1 or 2, but got {len(pretrain_img_size)}"
                raise Exception(msg)

        super().__init__()

        self.num_layers = len(depths)
        self.out_indices = out_indices
        self.feature_info = []

        if not strides[0] == patch_size:
            msg = "Use non-overlapping patch embed."
            raise Exception(msg)

        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            kernel_size=patch_size,
            stride=strides[0],
            padding="corner",
            norm_layer=norm_layer,
            padding_mode="replicate",
            drop_rate=drop_rate,
        )

        # self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        stages = []
        in_chans = embed_dim
        scale = 1
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                downsample = PatchMerging(
                    in_chans=in_chans,
                    out_channels=2 * in_chans,
                    stride=strides[i + 1],
                    norm_layer=norm_layer,
                )
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dim=in_chans,
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_chans),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=downsample,
                act_layer=act_layer,
                norm_layer=norm_layer,
                with_cp=with_cp,
            )
            stages.append(stage)
            if i > 0:
                scale *= 2
            self.feature_info += [{"num_chs": in_chans, "reduction": 4 * scale, "module": f"stages.{i}"}]
            if downsample:
                in_chans = downsample.out_channels
        self.stages = nn.Sequential(*stages)
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        # Add a norm layer for each output

        self.head = ClassifierHead(
            self.num_features[-1],
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            input_fmt=self.output_fmt,
        )

    def train(self, mode=True):  # noqa: FBT002
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):
            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f"norm{i - 1}")
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @torch.jit.ignore
    def init_weights(self, mode=""):
        modes = ("jax", "jax_nlhb", "moco", "")
        if mode not in modes:
            msg = f"mode must be one of {modes}"
            raise Exception(msg)
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if "relative_position_bias_table" in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def group_matcher(self, coarse=False):  # noqa: FBT002
        return {
            "stem": r"^patch_embed",  # stem and embed
            "blocks": r"^layers\.(\d+)"
            if coarse
            else [
                (r"^layers\.(\d+).downsample", (0,)),
                (r"^layers\.(\d+)\.\w+\.(\d+)", None),
                (r"^norm", (99999,)),
            ],
        }

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):  # noqa: FBT002, FBT001
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        features = self.forward_features(x)
        x = self.forward_head(features[0])
        return x
