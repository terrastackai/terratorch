# Copyright 2024 EPFL and Apple Inc.
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

import math
import random
import warnings
from functools import partial

import torch
from torch import nn

from .encoder_embeddings import ImageEncoderEmbedding, ImageTokenEncoderEmbedding
from .generate import (
    GenerationSampler,
    build_chained_generation_schedules,
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
from .tm_utils import Block, LayerNorm


def build_tim_modality_embeddings(modalities, tim_modalities, img_size=None, dim=None, patch_size=None):
    mod_embeddings, mod_name_mapping = build_modality_embeddings(MODALITY_INFO, modalities, img_size, dim, patch_size)

    for modality in tim_modalities:
        # Cover multiple naming conventions
        modality_renamed = (
            modality.lower()
            .replace("s2", "sen2")
            .replace("s1", "sen1")
            .replace("text", "caption")
            .replace("location", "coords")
        )

        # Get modality key in MODALITY_INFO
        if modality in MODALITY_INFO.keys():
            key = modality
        elif "sen2" in modality_renamed:
            key = "tok_sen2l2a@224"
        elif "sen1rtc" in modality_renamed:
            key = "tok_sen1rtc@224"
        elif "sen1" in modality_renamed:  # Default to S1GRD if not specified
            key = "tok_sen1grd@224"
        elif "dem" in modality_renamed:
            key = "tok_dem@224"
        elif "lulc" in modality_renamed:
            key = "tok_lulc@224"
        elif "ndvi" in modality_renamed:
            key = "tok_ndvi@224"
        elif "caption" in modality_renamed:
            key = "caption"
        elif "coords" in modality_renamed:
            key = "coords"
        else:
            raise NotImplementedError(f"Could not find modality {modality} in default modality info.")

        if modality in mod_name_mapping:
            # Modality is defined for input and TiM
            if key != mod_name_mapping[modality]:
                raise NotImplementedError(
                    f"Fallback TiM modalities are currently only supported for tokenized "
                    f"modalities, found {modality} ({mod_name_mapping[modality]})."
                )
                # TODO: Handle TiM modalities based on missing modalities in the input.
                #  E.g., if untok S2 input is missing, predict tok S2.
            continue

        mod_name_mapping[modality] = key  # Requires manual mapping for loading model weights
        mod_info = MODALITY_INFO[key]
        mod_embeddings[key] = mod_info["encoder_embedding"](image_size=img_size, dim_tokens=dim, **mod_info)

    return mod_embeddings, mod_name_mapping


class TerraMindTiM(nn.Module):
    """Modified TerraMind model for a "Thinking in Modalities" approach.

    Args:
        img_size (int): Input image size.
        modalities (list, dict, optional): List of modality keys and dicts, or dict with modality keys and values being
            ints (num_channels of modality) or nn.Module (patch embedding layer).
        tim_modalities (list, optional): List of tokenized modalities used for the TiM approach. The TiM outputs are
            generated in the same order as specified in the given list. Defaults to [LULC].
        tim_chained_generations (bool): If True, uses generated modalities as conditions for following TiM modalities.
            If False, generates all modalities only based on inputs. Defaults to True.
        tim_decoding_steps (int): Number of decoding steps for each TiM modality. Defaults to 1.
        tim_temps (float): Sampling temperature for each TiM modality. Defaults to 1.0.
        tim_top_p (float): Top-p sampling threshold for TiM modalities. Ignored if set to 0.0. Defaults to 0.8.
        tim_top_k (int): Top-k sampling threshold for TiM modalities. Ignored if set to 0. Defaults to 0.
        merge_method (str, optional): Specify how the output is merged for further processing. One of 'mean', 'max',
            'concat', 'dict', or None. 'mean', 'max', and 'concat' are dropping all sequence modality tokens, split all
            image modality tokens and reduce the by applying the appropriate method. 'dict' splits all tokens into a
            dictionary {'modality': torch.Tensor}. Defaults to 'mean'.
        patch_size (int): Patch size.
        in_chans (int): Number of input image channels.
        dim (int): Patch embedding dimension.
        encoder_depth (int): Depth of ViT / number of encoder blocks.
        num_heads (int): Number of attention heads in each ViT block.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        proj_bias (bool): If True, adds a bias to the attention out proj layer.
        mlp_bias (bool): If True, adds a learnable bias for the feedforward.
        num_register_tokens (int): Number of register tokens.
        drop_path_rate (float): Stochastic depth rate.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        modality_drop_rate (float): Drop modality inputs during training.
        act_layer (nn.Module): Activation layer.
        norm_layer (nn.Module): Normalization layer.
        gated_mlp (bool): If True, makes the feedforward gated (e.g., for SwiGLU)
        qk_norm (bool): If True, normalizes the query and keys (as in ViT-22B)
        encoder_norm (bool): If True, adds a norm layer after the last encoder block.
        tokenizer_dict (dict): Dictionary of tokenizers.
        pretrained (bool): If True, loads pretrained tokenizers.
    """

    def __init__(
        self,
        img_size: int = 224,
        modalities: list[str] | dict[str, int | nn.Module] | None = None,
        tim_modalities: list[str] | None = None,
        tim_chained_generations: bool = True,
        tim_decoding_steps: int = 1,
        tim_temps: float = 1.0,
        tim_top_p: float = 0.8,
        tim_top_k: int = 0,
        merge_method: str | None = "mean",
        patch_size: int = 16,
        in_chans: int = 3,
        dim: int = 768,
        encoder_depth: int = 12,
        decoder_depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        num_register_tokens: int = 0,
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        modality_drop_rate: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: partial | nn.Module = partial(LayerNorm, eps=1e-6),
        gated_mlp: bool = False,  # Make the feedforward gated for e.g. SwiGLU
        qk_norm: bool = False,
        encoder_norm: bool = True,
        tokenizer_dict: dict | None = None,
        pretrained: bool = False,
    ):
        super().__init__()

        if modalities is None or len(modalities) == 0:
            # Init new image modality
            modalities = [{"image": in_chans}]
        elif isinstance(modalities, dict):
            modalities = [modalities]
        elif not isinstance(modalities, list):
            raise ValueError("Modalities must be None, a list of modality keys or a dict with ints/embedding layers.")

        self.tim_modalities = tim_modalities or ["LULC"]
        self.tim_decoding_steps = tim_decoding_steps
        self.tim_temps = tim_temps
        self.tim_top_p = tim_top_p
        self.tim_top_k = tim_top_k

        # Build embedding layers for encoder (modalities and tim_modalities as inputs)
        mod_embeddings, mod_name_mapping = build_tim_modality_embeddings(
            modalities, self.tim_modalities, img_size=img_size, dim=dim, patch_size=patch_size
        )

        # Init embeddings for TiM model
        if tim_chained_generations:
            tim_encoder_embeddings = mod_embeddings
        else:
            tim_encoder_embeddings, _ = build_modality_embeddings(
                MODALITY_INFO, modalities, img_size=img_size, dim=dim, patch_size=patch_size
            )
        tim_decoder_embeddings, tim_decoder_name_mapping = build_output_modality_embeddings(
            MODALITY_INFO, self.tim_modalities, img_size=img_size, dim=dim, patch_size=patch_size
        )

        self.encoder_embeddings = nn.ModuleDict(mod_embeddings)
        self.mod_name_mapping = mod_name_mapping
        self.modalities = list(mod_name_mapping.keys())  # Further code expects list
        self.output_mod_name_mapping = {k: k for k in mod_name_mapping.keys()}
        self.output_mod_name_mapping.update({v: k for k, v in tim_decoder_name_mapping.items()})

        # Build TiM model
        enc_dec_model = TerraMind(
            encoder_embeddings=tim_encoder_embeddings,
            decoder_embeddings=tim_decoder_embeddings,
            modality_info=MODALITY_INFO,
            dim=dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mlp_bias=mlp_bias,
            act_layer=act_layer,
            norm_layer=norm_layer,
            gated_mlp=gated_mlp,
            qk_norm=qk_norm,
            num_register_tokens=num_register_tokens,
        )
        # No fine-tuning of the encoder decoder model
        enc_dec_model = enc_dec_model.requires_grad_(False)

        self.sampler = GenerationSampler(enc_dec_model)

        self.img_size = img_size
        self.merge_method = merge_method
        self.image_modalities = [
            key
            for key, value in self.encoder_embeddings.items()
            if isinstance(value, ImageEncoderEmbedding) or isinstance(value, ImageTokenEncoderEmbedding)
        ]
        self.modality_drop_rate = modality_drop_rate
        assert 0 <= self.modality_drop_rate <= 1, "modality_drop_rate must be in [0, 1]"
        # New learned parameter for handling missing modalities
        if self.merge_method == "concat":
            self.missing_mod_token = nn.Parameter(torch.Tensor(dim))

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_depth)]

        self.encoder = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_bias=mlp_bias,
                    drop_path=dpr[i],
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    gated_mlp=gated_mlp,
                    qk_norm=qk_norm,
                )
                for i in range(encoder_depth)
            ]
        )

        # Needed for terratorch decoders
        if merge_method == "concat":
            self.out_channels = [dim * len(self.image_modalities) for i in range(encoder_depth)]
        else:
            self.out_channels = [dim for i in range(encoder_depth)]

        self.encoder_norm = norm_layer(dim) if encoder_norm else nn.Identity()

        # Additional register tokens that can be used by the encoder during fine-tuning
        self.num_register_tokens = num_register_tokens
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, dim))
            nn.init.normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None

        if tokenizer_dict is not None:
            self.tokenizer = build_tokenizer(
                tokenizer_dict=tokenizer_dict,
                input_modalities=list(self.encoder_embeddings.keys()),
                pretrained=pretrained,
            )
        else:
            self.tokenizer = {}

        # Weight init
        self.init_weights()

    def init_weights(self):
        """Weight initialization following MAE's initialization scheme"""

        for name, m in self.named_modules():
            # Skipping tokenizers to avoid reinitializing them
            if "tokenizer" in name:
                continue
            # Linear
            elif isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif "kv" in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

            # Embedding
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            # Conv2d
            elif isinstance(m, nn.Conv2d):
                if ".proj" in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = set()

        for mod, emb_module in self.encoder_embeddings.items():
            if hasattr(emb_module, "no_weight_decay"):
                to_skip = emb_module.no_weight_decay()
                to_skip = set([f"encoder_embeddings.{mod}.{name}" for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def forward(self, d: dict[str, torch.Tensor] | torch.Tensor | None = None, **kwargs) -> list[torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            d (dict, torch.Tensor): Dict of inputs or input tensor with shape (B, C, H, W)

            Alternatively, keyword arguments with modality=tensor.

        Returns:
            list[torch.Tensor]: List of transformer layer outputs. Shape (B, L, D).
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
        d_mod = list(d.keys())
        batch_size = d[d_mod[0]].shape[0]
        device = d[d_mod[0]].device

        # Define the initial TiM input
        tim_dict = {}
        img_num_tokens = 196  # Default value if no images are provided
        for mod, value in list(d.items()):
            if self.mod_name_mapping[mod] in self.image_modalities:
                input_size = value.shape
            if self.mod_name_mapping[mod] in self.tokenizer:
                # Tokenize
                with torch.no_grad():
                    value = self.tokenizer[self.mod_name_mapping[mod]].encode(value, device)
                if isinstance(value, dict):
                    # Save tokenized input in d to avoid running the tokenizer twice
                    d[mod] = value
                else:
                    value = d[mod] = value[-1]  # Select tokens from img tokenizer

            if self.mod_name_mapping[mod] in self.image_modalities:
                # Get image size and num tokens
                patch_size = self.encoder_embeddings[self.mod_name_mapping[mod]].patch_size
                img_num_tokens = int((input_size[-1] / patch_size[-1]) * (input_size[-2] / patch_size[-2]))

                # Init raw image input masks
                value = {
                    "tensor": value,
                    "input_mask": torch.zeros(batch_size, img_num_tokens, dtype=torch.bool, device=device),
                    "target_mask": torch.ones(batch_size, img_num_tokens, dtype=torch.bool, device=device),
                    "decoder_attention_mask": torch.zeros(batch_size, img_num_tokens, dtype=torch.bool, device=device),
                }

            # Encode input and provide expected format
            tim_dict[self.mod_name_mapping[mod]] = init_full_input_modality(
                value, MODALITY_INFO, self.mod_name_mapping[mod], device
            )

        # Initialize TiM modalities
        target_domains = []
        tokens_per_target = []
        autoregression_schemes = []
        token_decoding_schedules = []
        token_decoding_steps = []

        for mod in self.tim_modalities:
            if self.mod_name_mapping[mod] in tim_dict:
                # TiM modality already in input, skipping TiM step
                continue
            if self.mod_name_mapping[mod] in self.image_modalities:
                mod_num_tokens = img_num_tokens
                autoregression_schemes.append("roar")
                token_decoding_schedules.append("linear")
                token_decoding_steps.append(self.tim_decoding_steps)
            else:
                # Get max length from modality info for sequence data
                mod_num_tokens = 50
                autoregression_schemes.append("autoregressive")
                token_decoding_schedules.append(None)
                token_decoding_steps.append(None)
            tokens_per_target.append(mod_num_tokens)
            target_domains.append(self.mod_name_mapping[mod])
            tim_dict[self.mod_name_mapping[mod]] = init_empty_target_modality(
                MODALITY_INFO, self.mod_name_mapping[mod], batch_size, mod_num_tokens, device
            )
        num_tim_mod = len(target_domains)

        # Predict tokens for TiM modalities
        schedule = build_chained_generation_schedules(
            cond_domains=[self.mod_name_mapping[m] for m in d_mod],
            target_domains=target_domains,
            tokens_per_target=tokens_per_target,
            autoregression_schemes=autoregression_schemes,
            decoding_steps=token_decoding_steps,
            token_decoding_schedules=token_decoding_schedules,
            temps=[self.tim_temps] * num_tim_mod,
            temp_schedules=["constant"] * num_tim_mod,
            cfg_scales=[1.0] * num_tim_mod,
            cfg_schedules=["constant"] * num_tim_mod,
            cfg_grow_conditioning=True,
        )

        with torch.no_grad():
            out_dict = self.sampler.generate(
                tim_dict,
                schedule,
                verbose=False,
                seed=random.randint(-(2**31), 2**31 - 1),
                top_p=self.tim_top_p,
                top_k=self.tim_top_k,
                num_tokens=sum(tokens_per_target),
                tokenizer=self.tokenizer,
            )

        # Add TiM outputs to input dict
        for mod in target_domains:
            d[self.output_mod_name_mapping[mod]] = out_dict[mod]["tensor"]

        if self.training and self.modality_drop_rate:
            # Drop random modalities during training
            for key in random.sample(d_mod, k=len(d) - 1):
                if random.random() < self.modality_drop_rate:
                    _ = d.pop(key)

        x = []
        num_tokens = []
        image_mod = []
        for mod, tensor in d.items():
            mod_dict = self.encoder_embeddings[self.mod_name_mapping[mod]](tensor)
            # Add embeddings to patchified data
            x.append(mod_dict["x"] + mod_dict["emb"])
            num_tokens.append(mod_dict["x"].shape[-2])
            image_mod.append(self.mod_name_mapping[mod] in self.image_modalities)

        # Concatenate along token dim
        x = torch.cat(x, dim=1)  # Shape: (B, N, D)

        if self.num_register_tokens > 0:
            register_tokens = self.register_tokens.repeat((x.shape[0], 1, 1))
            # We add register tokens at the beginning of the sequence
            x = torch.cat([register_tokens, x], dim=1)
            if self.merge_method == "dict":
                # Return register tokens as additional modality
                d_mod.insert(0, "register_tokens")
                num_tokens.insert(0, self.num_register_tokens)

        out = []
        for block in self.encoder:
            x = block(x)
            out.append(x.clone())

        out[-1] = self.encoder_norm(x)  # Shape: (B, N, D)

        def _unstack_image_modalities(x):
            if self.num_register_tokens:
                # Remove register tokens
                x = x[:, self.num_register_tokens :]
            x = torch.split(x, num_tokens, dim=1)  # Split tokens by modality
            x = [m for m, keep in zip(x, image_mod) if keep]  # Drop sequence modalities
            x = torch.stack(x, dim=1)  # (B, M, N, D)
            return x

        # Merge tokens from different modalities
        if self.merge_method == "mean":
            out = [_unstack_image_modalities(x) for x in out]
            out = [x.mean(dim=1) for x in out]

        elif self.merge_method == "max":
            out = [_unstack_image_modalities(x) for x in out]
            out = [x.max(dim=1)[0] for x in out]

        elif self.merge_method == "concat":
            out = [_unstack_image_modalities(x) for x in out]
            if len(d) < len(self.image_modalities):
                # Handle missing modalities with missing_mod_token
                num_missing = len(self.image_modalities) - len(d)
                missing_tokens = self.missing_mod_token.repeat(out[-1].shape[0], num_missing, out[-1].shape[2], 1)
                out = [torch.cat([x, missing_tokens], dim=1) for x in out]
            # Concat along embedding dim
            out = [torch.cat(x.unbind(dim=1), dim=-1) for x in out]

        elif self.merge_method == "dict":
            out = [torch.split(x, num_tokens, dim=1) for x in out]
            out = [{mod: x[i] for i, mod in enumerate(d_mod)} for x in out]

        elif self.merge_method is None:
            pass  # Do nothing
        else:
            raise NotImplementedError(
                f"Merging method {self.merge_method} is not implemented. Select one of mean, max or concat."
            )

        return out
