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

import torch

from .vqvae import VQ


class VQBackbone(VQ):
    def __init__(self, return_type: str = "embedding", **kwargs):
        """
        Backbone class for VQ models.

        return_type: "embedding", "latent", or "quantized"
        """
        super().__init__(**kwargs)
        self.return_type = return_type
        if self.return_type == "embedding":
            self.out_channels = [self.enc_dim] * len(self.encoder.blocks)
        elif self.return_type in ["latent", "quantized"]:
            self.out_channels = [self.latent_dim]
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the encoder and quantizer.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation

        Returns:
            Embedding, latent, or quantized latent code of shape B D_Q H_Q W_Q
        """
        x = self.prepare_input(x)

        embedding = self.encoder(x, return_intermediates=True)
        latent = self.quant_proj(embedding[-1])

        if self.return_type == "embedding":
            return embedding
        elif self.return_type == "latent":
            return latent
        elif self.return_type == "quantized":
            quant, code_loss, tokens = self.quantize(latent)
            return quant
