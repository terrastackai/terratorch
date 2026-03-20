# Copyright contributors to the Terratorch project

import importlib

from torch import nn


class PixelShuffleUpscale(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(self.channels, self.channels * 2 * 2, kernel_size=3, stride=1, padding=1)
        self.upscale = nn.PixelShuffle(2)
        self.bn = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        post_conv = self.conv(x)
        upscaled = self.upscale(post_conv)
        return self.relu(self.bn(upscaled))


class RegressionHead(nn.Module):
    """Regression head"""

    def __init__(
        self,
        in_channels: int,
        num_outputs: int = 1,
        final_act: nn.Module | str | None = None,
        learned_upscale_layers: int = 0,
        channel_list: list[int] | None = None,
        batch_norm: bool = True,
        dropout: float = 0,
    ) -> None:
        """Constructor

        Args:
            in_channels (int): Number of input channels
            num_outputs (int): Number of predicted regression variables. Defaults to single regression.
            final_act (nn.Module | None, optional): Final activation to be applied. Defaults to None.
            learned_upscale_layers (int, optional): Number of Pixelshuffle layers to create. Each upscales 2x.
                Defaults to 0.
            channel_list (list[int] | None, optional): List with number of channels for each Conv
                layer to be created. Defaults to None.
            batch_norm (bool, optional): Whether to apply batch norm. Defaults to True.
            dropout (float, optional): Dropout value to apply. Defaults to 0.

        """
        super().__init__()

        self.learned_upscale_layers = learned_upscale_layers
        self.final_act = final_act if final_act else nn.Identity()

        if isinstance(final_act, str):
            module_name, class_name = final_act.rsplit(".", 1)
            target_class = getattr(importlib.import_module(module_name), class_name)
            self.final_act = target_class()
        pre_layers = []

        if learned_upscale_layers != 0:
            learned_upscale = nn.Sequential(
                *[PixelShuffleUpscale(in_channels) for _ in range(self.learned_upscale_layers)]
            )
            pre_layers.append(learned_upscale)

        if channel_list is None:
            pre_head = nn.Identity()

        else:

            def block(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )

            channel_list = [in_channels, *channel_list]
            pre_head = nn.Sequential(
                *[block(channel_list[i], channel_list[i + 1]) for i in range(len(channel_list) - 1)]
            )
            in_channels = channel_list[-1]
            pre_layers.append(pre_head)

        dropout = nn.Dropout2d(dropout)

        final_layer = nn.Conv2d(in_channels=in_channels, out_channels=num_outputs, kernel_size=1)
        self.head = nn.Sequential(*[*pre_layers, dropout, final_layer])

    def forward(self, x):
        output = self.head(x)

        return self.final_act(output)
