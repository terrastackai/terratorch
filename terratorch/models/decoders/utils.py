import warnings

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

"""
Adapted from https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py
"""


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=0,
        dilation=1,
        stride=1,
        inplace=False,
        transpose=False,
        scale_factor=None,
    ) -> None:

        super().__init__()

        if transpose:
            kind = "Transpose"
        else:
            kind = ""

        conv_name = f"Conv{kind}2d"

        if transpose:
            stride = scale_factor
            padding = (kernel_size - scale_factor) // 2

        conv_template = getattr(nn, conv_name)
        self.conv = conv_template(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, stride=stride, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x):

        return self.act(self.norm(self.conv(x)))


"""
It cames from https://github.com/open-mmlab/mmsegmentation
"""


def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )

    return F.interpolate(input, size, scale_factor, mode, align_corners)
