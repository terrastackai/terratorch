import torch
from torch import nn

from terratorch.models.model import Model


class Encoder(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, hidden_channels_multiplier: list = [1, 2, 4, 8], num_encoder_blocks=4
    ):
        if len(hidden_channels_multiplier) != num_encoder_blocks:
            raise ValueError(
                f"hidden channels multiplier lenght {len(hidden_channels_multiplier)} not matching encoder blocks {num_encoder_blocks}"
            )
        super().__init__()

        self.encoders = [None] * num_encoder_blocks

        for index in range(num_encoder_blocks):
            if index == 0:
                self.encoders[index] = nn.Sequential(
                    nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                self.encoders[index] = nn.Sequential(
                    nn.Conv2d(
                        hidden_channels * hidden_channels_multiplier[index - 1],
                        hidden_channels * hidden_channels_multiplier[index],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_channels * hidden_channels_multiplier[index]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        hidden_channels * hidden_channels_multiplier[index],
                        hidden_channels * hidden_channels_multiplier[index],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_channels * hidden_channels_multiplier[index]),
                    nn.ReLU(inplace=True),
                )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.encoders[index] = self.encoders[index].to(device)

    def forward(self, x):
        encoder_values = [None] * len(self.encoders)
        for index in range(len(encoder_values)):
            if index == 0:
                encoder_values[index] = self.encoders[index](x)
            else:
                encoder_values[index] = self.encoders[index](encoder_values[index - 1])

        return tuple(encoder_values)


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        out_channels,
        hidden_channels_multiplier: list = [(16, 8), (12, 4), (6, 2), (3, 1)],
        num_decoder_blocks=4,
        skip_connection=True,
    ):
        if len(hidden_channels_multiplier) != num_decoder_blocks:
            raise ValueError(
                f"hidden channels multiplier lenght {len(hidden_channels_multiplier)} not matching encoder blocks {num_decoder_blocks}"
            )
        super().__init__()

        self.decoders = [None] * num_decoder_blocks

        for index in range(num_decoder_blocks):
            self.decoders[index] = nn.Sequential(
                nn.Conv2d(
                    hidden_channels * hidden_channels_multiplier[index][0],
                    hidden_channels * hidden_channels_multiplier[index][1],
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(hidden_channels * hidden_channels_multiplier[index][1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_channels * hidden_channels_multiplier[index][1],
                    hidden_channels * hidden_channels_multiplier[index][1],
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(hidden_channels * hidden_channels_multiplier[index][1]),
                nn.ReLU(inplace=True),
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.decoders[index] = self.decoders[index].to(device)

        # Final output layer
        self.final_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        self.skip_connection = skip_connection

    def forward(self, encoders_values: tuple, backbone_values: torch.Tensor):
        if len(encoders_values) != len(self.decoders):
            raise ValueError(
                f"asymetric UNets not (yet) supported, encoders {len(encoders_values)} not matching decoders {len(self.decoders)}"
            )

        pass_on = backbone_values

        if self.skip_connection:
            for index, encoder in enumerate(reversed(encoders_values)):
                pass_on = self.decoders[index](torch.cat((pass_on, encoder), dim=1))

        output = self.final_conv(pass_on)
        return output


class UNetPincer(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        lr: float = 1e-3,
        in_channels: int = 488,
        hidden_channels: int = 160,
        out_channels: int = 366,
        patch_size_px: list[int] = [2, 2],
        encoder_hidden_channels_multiplier: list = [1, 2, 4, 8],
        encoder_num_encoder_blocks=4,
        decoder_hidden_channels_multiplier: list = [(16, 8), (12, 4), (6, 2), (3, 1)],
        decoder_num_decoder_blocks=4,
        skip_connection=True,
    ):
        super().__init__()

        self.lr: float = lr
        self.patch_size_px: list[int] = patch_size_px
        self.out_channels: int = out_channels

        self.encoder = Encoder(
            in_channels, hidden_channels, encoder_hidden_channels_multiplier, encoder_num_encoder_blocks
        )
        self.decoder = Decoder(
            hidden_channels, out_channels, decoder_hidden_channels_multiplier, decoder_num_decoder_blocks
        )

        self.backbone = backbone

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.backbone = backbone.to(device)
        self.skip_connection = skip_connection

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]
        lead_time = batch["lead_time"]
        static = batch["static"]
        x = x.squeeze(1)

        encoder_values: tuple = self.encoder(x)

        # Reshape encoded data for the transformer on last encoder value
        *_, last_encoder_value = encoder_values
        batch_size, c, h, w = last_encoder_value.size()
        last_encoder_value_reshaped = last_encoder_value.unsqueeze(1)

        # Prepare input for transformer model
        batch_dict = {
            "x": last_encoder_value_reshaped,
            "y": last_encoder_value,
            "lead_time": lead_time,
            "static": static,
            "input_time": torch.zeros_like(lead_time),
        }

        # Transformer forward pass
        transformer_output = self.backbone(batch_dict)
        transformer_output_reshaped = transformer_output.view(batch_size, c, h, w)

        # Decode the transformer output
        output = self.decoder(encoder_values, transformer_output_reshaped)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int = None) -> torch.Tensor:
        y_hat: torch.Tensor = self(batch)

        loss: torch.Tensor = torch.nn.functional.mse_loss(input=y_hat, target=batch["target"])
        return loss

    def get_model(self):
        return self.backbone, self.decoder, self.encoder
