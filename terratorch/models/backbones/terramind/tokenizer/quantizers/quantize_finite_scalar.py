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
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from torch import nn


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def round_ste(x):
    """Round with straight through gradients."""
    xhat = x.round()
    return x + (xhat - x).detach()


class FiniteScalarQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: str,  # example: "8-8-8-6-5"
    ):
        super().__init__()

        levels = [int(level) for level in codebook_size.split("-")]

        _levels = torch.tensor(levels, dtype=torch.int)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int)
        self.register_buffer("_basis", _basis, persistent=False)

        # initialize codebook
        self.codebook_size = self._levels.prod().item()
        codebook = self.indice_to_code(torch.arange(self.codebook_size))
        self.register_buffer("codebook", codebook, persistent=False)

    def latent_to_code_and_indice(self, latent):
        d = self._levels - 1
        number = round_ste(F.sigmoid(latent) * d)
        code = number / d
        indice = (number * self._basis).sum(dim=-1).to(torch.int)

        return code, indice

    def indice_to_code(self, indice):
        # (..., d)
        code = (indice.unsqueeze(-1) // self._basis) % self._levels
        # convert to [0, 1]
        code = code / (self._levels - 1)

        return code

    def indices_to_embedding(self, indices):
        B, H, W = indices.shape
        indices = rearrange(indices, "b h w -> b (h w)")
        embeddings = self.indice_to_code(indices)
        embeddings = rearrange(embeddings, "b (h w) c -> b c h w", h=H)
        return embeddings

    def forward(self, x):
        height, width = x.shape[-2:]
        x = rearrange(x, "b c h w -> b (h w) c")

        quantize, embed_ind = self.latent_to_code_and_indice(x)

        quantize = rearrange(quantize, "b (h w) c -> b c h w", h=height, w=width)
        embed_ind = rearrange(embed_ind, "b (h w) ... -> b h w ...", h=height, w=width)

        # no auxiliary losses needed for FSQ
        loss = torch.tensor([0.0], device=x.device, requires_grad=self.training)

        return quantize, loss, embed_ind
