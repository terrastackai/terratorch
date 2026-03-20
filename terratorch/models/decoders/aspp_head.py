import numpy as np
import torch
from torch import nn

from terratorch.registry import TERRATORCH_DECODER_REGISTRY

from .utils import ConvModule, resize


@TERRATORCH_DECODER_REGISTRY.register
class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg, act_cfg):

        super().__init__()

        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        modules_list = list()

        for dilation in dilations:
            layer_module = ConvModule(
                self.in_channels,
                self.channels,
                1 if dilation == 1 else 3,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation,
            )

            # TODO Extend it to support more possible configurations
            # for convolution, normalization and activation.
            # conv_cfg=self.conv_cfg,
            # norm_cfg=self.norm_cfg,
            # act_cfg=self.act_cfg)

            modules_list.append(layer_module)

        self.model_sequence = torch.nn.ModuleList(modules_list)

    def forward(self, x):
        """Forward function."""
        outs = []
        for module in self.model_sequence:
            outs.append(module(x))

        return outs


@TERRATORCH_DECODER_REGISTRY.register
class ASPPHead(nn.Module):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(
        self,
        dilations: list | tuple = (1, 6, 12, 18),
        in_channels: int = None,
        channels: int = None,
        out_dim: int = 3,
        align_corners=False,
        head_dropout_ratio: float = 0.3,
        input_transform: str = None,
        in_index: int = -1,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.out_dim = out_dim

        self.align_corners = align_corners
        self.input_transform = input_transform
        self.in_index = in_index

        if "conv_cfg" not in kwargs:
            self.conv_cfg = self._default_conv_cfg

        if "norm_cfg" not in kwargs:
            self.norm_cfg = self._default_norm_cfg

        if "act_cfg" not in kwargs:
            self.act_cfg = self._default_act_cfg

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                # TODO Extend it to support more possible configurations
                # for convolution, normalization and activation.
                # self.conv_cfg,
                # norm_cfg=self.norm_cfg,
                # act_cfg=self.act_cfg))
            ),
        )

        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            padding=1,
        )
        # TODO Extend it to support more possible configurations
        # for convolution, normalization and activation.
        # conv_cfg=self.conv_cfg,
        # norm_cfg=self.norm_cfg,
        # act_cfg=self.act_cfg)

        if head_dropout_ratio > 0:
            self.dropout = nn.Dropout2d(head_dropout_ratio)

    @property
    def _default_conv_cfg(self):
        return {"kernel_size": 3, "padding": 0, "bias": False}

    @property
    def _default_norm_cfg(self):
        return {}

    @property
    def _default_act_cfg(self):
        return {}

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _forward_feature(self, inputs):
        """Forward function.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        aspp_outs = [
            resize(self.image_pool(inputs), size=inputs.size()[2:], mode="bilinear", align_corners=self.align_corners)
        ]

        aspp_outs.extend(self.aspp_modules(inputs))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        feats = self.bottleneck(aspp_outs)

        return feats

    def forward(self, inputs):

        output = self._forward_feature(inputs)

        return output


@TERRATORCH_DECODER_REGISTRY.register
class ASPPSegmentationHead(ASPPHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(
        self,
        channel_list,
        dilations: list | tuple = (1, 6, 12, 18),
        in_channels: int = None,
        channels: int = None,
        num_classes: int = 2,
        align_corners=False,
        head_dropout_ratio: float = 0.3,
        input_transform: str = None,
        in_index: int = -1,
        **kwargs,
    ):

        super().__init__(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            align_corners=align_corners,
            head_dropout_ratio=head_dropout_ratio,
            input_transform=input_transform,
            in_index=in_index,
            **kwargs,
        )

        self.num_classes = num_classes
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

        if head_dropout_ratio > 0:
            self.dropout = nn.Dropout2d(head_dropout_ratio)

    def segmentation_head(self, features):
        """PixelWise classification"""

        if self.dropout is not None:
            features = self.dropout(features)
        output = self.conv_seg(features)
        return output

    def forward(self, inputs):

        output = self._forward_feature(inputs)
        output = self.segmentation_head(output)

        return output


@TERRATORCH_DECODER_REGISTRY.register
class ASPPRegressionHead(ASPPHead):
    """Rethinking Atrous Convolution for regression.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(
        self,
        channel_list,
        dilations: list | tuple = (1, 6, 12, 18),
        in_channels: int = None,
        channels: int = None,
        out_channels: int = 1,
        align_corners=False,
        head_dropout_ratio: float = 0.3,
        input_transform: str = None,
        in_index: int = -1,
        **kwargs,
    ):

        super().__init__(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            out_dim=out_channels,
            align_corners=align_corners,
            head_dropout_ratio=head_dropout_ratio,
            input_transform=input_transform,
            in_index=in_index,
            **kwargs,
        )

        self.out_channels = out_channels
        self.conv_reg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)

    def regression_head(self, features):
        """PixelWise regression"""
        if self.dropout is not None:
            features = self.dropout(features)
        output = self.conv_reg(features)
        return output

    def forward(self, inputs):

        output = self._forward_feature(inputs)
        output = self.regression_head(output)

        return output
