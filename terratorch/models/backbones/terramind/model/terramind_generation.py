# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import warnings
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .decoder_embeddings import ImageTokenDecoderEmbedding
from .encoder_embeddings import ImageEncoderEmbedding, ImageTokenEncoderEmbedding
from .generate import (
    GenerationSampler,
    build_chained_generation_schedules,
    init_conditioned_target_modality,
    init_empty_target_modality,
    init_full_input_modality,
)
from .modality_info import MODALITY_INFO
from .terramind import (
    TerraMind,
    build_modality_embeddings,
    build_output_modality_embeddings,
    build_tokenizer,
)
from .tm_utils import LayerNorm


class TerraMindGeneration(nn.Module):
    """Modified TerraMind model for a "Thinking in Modalities" approach.

    Args:
        img_size (int): Input image size.
        modalities (list, dict, optional): List of modality keys and dicts, or dict with modality keys and values being
            ints (num_channels of modality) or nn.Module (patch embedding layer).
        output_modalities (list, optional): List of tokenized modalities used for the TiM approach. The TiM outputs are
            generated in the same order as specified in the given list. Defaults to [tbd].
        decoding_steps (list, int): Number of decoding steps for each TiM modality. Defaults to 1.
        temps (list, float): Sampling temperatures for each TiM modality. Defaults to 1.0.
        top_p (float): Top-p sampling threshold for TiM modalities. Ignored if set to 0.0. Defaults to 0.8.
        top_k (int): Top-k sampling threshold for TiM modalities. Ignored if set to 0. Defaults to 0.
        patch_size (int): Patch size.
        dim (int): Patch embedding dimension.
        encoder_depth (int): Depth of ViT / number of encoder blocks.
        num_heads (int): Number of attention heads in each ViT block.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        proj_bias (bool): If True, adds a bias to the attention out proj layer.
        mlp_bias (bool): If True, adds a learnable bias for the feedforward.
        num_register_tokens (int): Number of register tokens.
        act_layer (nn.Module): Activation layer.
        norm_layer (nn.Module): Normalization layer.
        gated_mlp (bool): If True, makes the feedforward gated (e.g., for SwiGLU)
        qk_norm (bool): If True, normalizes the query and keys (as in ViT-22B)
        tokenizer_dict (dict): Dictionary of tokenizers.
        pretrained (bool): If True, loads pretrained tokenizers.
    """

    def __init__(
        self,
        img_size: int = 224,
        modalities: list[str] | dict[str, int | nn.Module] | None = None,
        output_modalities: list[str] | None = None,
        decoding_steps: list[int] | int = 1,
        temps: list[float] | float = 1.0,
        top_p: float = 0.8,
        top_k: int = 0,
        timesteps: int = 50,
        patch_size: int = 16,
        dim: int = 768,
        encoder_depth: int = 12,
        decoder_depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        num_register_tokens: int = 0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: partial | nn.Module = partial(LayerNorm, eps=1e-6),
        gated_mlp: bool = False,
        qk_norm: bool = False,
        standardize: bool = False,
        offset: dict[str, float] | None = None,
        pretraining_mean: dict[str, list] | None = None,
        pretraining_std: dict[str, list] | None = None,
        tokenizer_dict: dict = None,
        pretrained: bool = False,
    ):
        super().__init__()

        if modalities is None or len(modalities) == 0:
            raise ValueError("Input modalities not provided.")
        elif isinstance(modalities, dict):
            modalities = [modalities]
        elif not isinstance(modalities, list):
            raise ValueError("Modalities must be a list of modality keys or a dict with embedding layers.")

        if output_modalities is None or len(output_modalities) == 0:
            raise ValueError("Output modalities not provided.")
        elif not isinstance(output_modalities, list):
            raise ValueError("Output modalities must be a list of modality keys.")

        # Parameters for generation schedule
        self.top_p = top_p
        self.top_k = top_k
        self.timesteps = timesteps
        self.decoding_steps = decoding_steps
        self.temps = temps

        # Init embeddings
        self.encoder_embeddings, mod_name_mapping = build_modality_embeddings(
            MODALITY_INFO, modalities, img_size=img_size, dim=dim, patch_size=patch_size
        )
        self.decoder_embeddings, decoder_name_mapping = build_output_modality_embeddings(
            MODALITY_INFO, output_modalities, img_size=img_size, dim=dim, patch_size=patch_size
        )
        self.output_modalities = list(decoder_name_mapping.values())  # Update output modality names
        self.mod_name_mapping = decoder_name_mapping | mod_name_mapping  # Merging dicts
        self.modalities = list(mod_name_mapping.keys())  # Further code expects list
        self.image_modalities = [
            key
            for key, value in self.encoder_embeddings.items()
            if isinstance(value, ImageEncoderEmbedding) or isinstance(value, ImageTokenEncoderEmbedding)
        ]
        self.output_image_modalities = [
            key for key, value in self.decoder_embeddings.items() if isinstance(value, ImageTokenDecoderEmbedding)
        ]
        self.output_mod_name_mapping = {v: k for k, v in decoder_name_mapping.items()}
        self.standardize = standardize

        if len(modalities) == 1 and self.mod_name_mapping[modalities[0]] == "caption":
            # TODO Debug why text to image generations don't work.
            raise NotImplementedError("TerraMind v0.1 generations with only text input don't work yet.")

        if offset is not None:
            for mod, o in offset.items():
                # Add offset to mean values
                if mod not in self.mod_name_mapping:
                    warnings.warn(f"offset {mod} not defined in input or output modalities, ignoring offset.")
                    continue
                pretraining_mean[self.mod_name_mapping[mod]] = (
                    np.array(pretraining_mean[self.mod_name_mapping[mod]], dtype=float) + o
                )

        self.pretraining_mean = (
            {
                mod: torch.tensor(mean, dtype=torch.get_default_dtype())[None, :, None, None]
                for mod, mean in pretraining_mean.items()
            }
            if pretraining_mean
            else {}
        )
        self.pretraining_std = (
            {
                mod: torch.tensor(std, dtype=torch.get_default_dtype())[None, :, None, None]
                for mod, std in pretraining_std.items()
            }
            if pretraining_std
            else {}
        )

        # Build MAE model
        mae_model = TerraMind(
            encoder_embeddings=self.encoder_embeddings,
            decoder_embeddings=self.decoder_embeddings,
            modality_info=MODALITY_INFO,
            dim=dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mlp_bias=mlp_bias,
            num_register_tokens=num_register_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
            gated_mlp=gated_mlp,
            qk_norm=qk_norm,
        )

        self.sampler = GenerationSampler(mae_model)

        self.tokenizer = build_tokenizer(
            tokenizer_dict=tokenizer_dict,
            input_modalities=list(self.encoder_embeddings.keys()),
            output_modalities=list(self.decoder_embeddings.keys()),
            pretrained=pretrained,
        )

    def to(self, device):
        super().to(device)
        # Move standardization values to device
        for mod, mean in self.pretraining_mean.items():
            self.pretraining_mean[mod] = mean.to(device)
        for mod, std in self.pretraining_std.items():
            self.pretraining_std[mod] = std.to(device)
        return self

    def forward(
        self,
        d: dict[str, torch.Tensor] | torch.Tensor | None = None,
        standardize: bool | None = None,
        timesteps: int = None,
        verbose: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            d (dict, torch.Tensor): Dict of inputs or input tensor with shape (B, C, H, W)

            Alternatively, keyword arguments with modality=tensor.

        Returns:
            dict[str, torch.Tensor]: Dict of generated images
        """
        # Handle single image modality
        if not isinstance(d, dict):
            # Assuming first modality
            d = {self.modalities[0]: d}
        elif d is None or len(d) == 0:
            d = {}
            if len(kwargs) == 0:
                raise ValueError("No inputs provided.")

        # Add additional keyword args to input dict
        for key, value in kwargs.items():
            d[key] = value

        # Check for unknown modalities in input
        for mod in list(d.keys()):
            if mod not in self.mod_name_mapping:
                warnings.warn(f"Unknown input modality: {mod}. Ignoring input.")
                del d[mod]
        if len(d) == 0:
            raise ValueError("No valid inputs provided.")

        # Get batch size and device
        batch_size = len(list(d.values())[0])
        device = next(self.parameters()).device

        standardize = standardize if standardize is not None else self.standardize
        if standardize:
            for mod, value in d.items():
                if self.mod_name_mapping[mod] in self.pretraining_mean:
                    d[mod] = (value - self.pretraining_mean[self.mod_name_mapping[mod]]) / self.pretraining_std[
                        self.mod_name_mapping[mod]
                    ]

        # Define the initial input
        input_dict = {}
        # Default values if no images are provided
        img_num_tokens, image_size = 196, (224, 224)
        for mod, value in d.items():
            if self.mod_name_mapping[mod] in self.image_modalities:
                input_shape = value.shape
            if self.mod_name_mapping[mod] in self.tokenizer:
                # Tokenize
                value = self.tokenizer[self.mod_name_mapping[mod]].encode(value, device)
                if not isinstance(value, dict):
                    value = value[-1]  # Select tokens from img tokenizer

            if self.mod_name_mapping[mod] in self.image_modalities:
                # Get image size and num tokens
                patch_size = self.encoder_embeddings[self.mod_name_mapping[mod]].patch_size
                img_num_tokens = int((input_shape[-1] / patch_size[-1]) * (input_shape[-2] / patch_size[-2]))
                image_size = (input_shape[-1], input_shape[-2])

                # Init raw image input masks
                value = {
                    "tensor": value,
                    "input_mask": torch.zeros(batch_size, img_num_tokens, dtype=torch.bool, device=device),
                    "target_mask": torch.ones(batch_size, img_num_tokens, dtype=torch.bool, device=device),
                    "decoder_attention_mask": torch.zeros(batch_size, img_num_tokens, dtype=torch.bool, device=device),
                }

            # Encode input and provide expected format
            input_dict[self.mod_name_mapping[mod]] = init_full_input_modality(
                value, MODALITY_INFO, self.mod_name_mapping[mod], device
            )

        # Initialize output modalities
        tokens_per_target = []
        autoregression_schemes = []
        token_decoding_schedules = []
        token_decoding_steps = []

        for mod in self.output_modalities:
            if mod in self.output_image_modalities:
                mod_num_tokens = img_num_tokens
                autoregression_schemes.append("roar")
                token_decoding_schedules.append("linear")
                token_decoding_steps.append(self.decoding_steps)
            else:
                # Get max length from modality info for sequence data
                mod_num_tokens = self.decoder_embeddings[mod].max_length
                autoregression_schemes.append("autoregressive")
                token_decoding_schedules.append(None)
                token_decoding_steps.append(None)
            tokens_per_target.append(mod_num_tokens)

            if mod in input_dict:
                # Modality in input and target
                input_dict[mod] = init_conditioned_target_modality(input_dict[mod], MODALITY_INFO, mod, mod_num_tokens)
            else:
                input_dict[mod] = init_empty_target_modality(MODALITY_INFO, mod, batch_size, mod_num_tokens, device)

        # Predict tokens of output modalities
        schedule = build_chained_generation_schedules(
            cond_domains=[self.mod_name_mapping[m] for m in d.keys()],
            target_domains=self.output_modalities,
            tokens_per_target=tokens_per_target,
            autoregression_schemes=autoregression_schemes,
            decoding_steps=token_decoding_steps,
            token_decoding_schedules=token_decoding_schedules,
            temps=[self.temps] * len(self.output_modalities),
            temp_schedules=["constant"] * len(self.output_modalities),
            cfg_scales=[1.0] * len(self.output_modalities),
            cfg_schedules=["constant"] * len(self.output_modalities),
            cfg_grow_conditioning=True,
        )

        out_dict = self.sampler.generate(
            input_dict,
            schedule,
            verbose=False,
            seed=random.randint(-(2**31), 2**31 - 1),
            top_p=self.top_p,
            top_k=self.top_k,
            num_tokens=sum(tokens_per_target),
            tokenizer=self.tokenizer,
        )

        # TODO Vary timesteps based on codebook diversity
        timesteps = timesteps or self.timesteps
        out = {}
        for mod in self.output_modalities:
            tok = out_dict[mod]["tensor"]
            if mod in self.output_image_modalities:
                patch_size = self.tokenizer[mod].patch_size
                tok = rearrange(
                    tok, "b (nh nw) -> b nh nw", nh=image_size[0] // patch_size, nw=image_size[1] // patch_size
                )

                out[self.output_mod_name_mapping[mod]] = self.tokenizer[mod].decode_tokens(
                    tok, image_size=image_size, timesteps=timesteps, verbose=verbose
                )

            elif mod in self.output_modalities and mod in ["caption", "coords"]:
                out[self.output_mod_name_mapping[mod]] = self.tokenizer[mod].decode_text(out_dict)

        if standardize:
            for mod, value in out.items():
                if self.mod_name_mapping[mod] in self.pretraining_mean:
                    out[mod] = (
                        value * self.pretraining_std[self.mod_name_mapping[mod]]
                        + self.pretraining_mean[self.mod_name_mapping[mod]]
                    )

        return out
