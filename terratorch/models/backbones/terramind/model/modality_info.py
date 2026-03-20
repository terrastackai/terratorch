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
#
# ---
#
# This project includes code adapted from the original work by EPFL and Apple Inc.,
# licensed under the Apache License, Version 2.0.
# Source: https://github.com/apple/ml-4m/

from functools import partial

from terratorch.models.backbones.terramind.utils import generate_uint15_hash

from .decoder_embeddings import ImageTokenDecoderEmbedding, SequenceDecoderEmbedding
from .encoder_embeddings import ImageEncoderEmbedding, ImageTokenEncoderEmbedding, SequenceEncoderEmbedding

MODALITY_INFO = {
    "sen1grd@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        # 'decoder_embedding': None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 2,
        "id": generate_uint15_hash("sen1grd@264"),
        "path": "S1GRD/",
    },
    "sen1rtc@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        # 'decoder_embedding': None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 2,
        "id": generate_uint15_hash("sen1rtc@264"),
        "path": "S1RTC/",
    },
    "sen2l2a@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=12),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=12),
        # 'decoder_embedding': None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 12,
        "id": generate_uint15_hash("sen2l2a@264"),
        "path": "S2L2A/",
    },
    "sen2l1c@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=12),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=12),
        # 'decoder_embedding': None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 13,
        "id": generate_uint15_hash("sen2l1c@264"),
        "path": "S2L1C/",
    },
    "sen2rgb@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=3),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=3),
        # 'decoder_embedding': None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 3,
        "id": generate_uint15_hash("sen2rgb@264"),
        "path": "S2RGB/",
    },
    "lulc@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=9),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=9),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 9,
        "id": generate_uint15_hash("lulc@264"),
        "path": "LULC/",
    },
    "dem@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=1),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=1),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("dem@264"),
        "path": "DEM/",
    },
    "ndvi@264": {
        "input_size": 264,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=1),
        "decoder_embedding": partial(ImageEncoderEmbedding, num_channels=1),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("ndvi@264"),
        "path": "NDVI/",
    },
    "untok_sen2l2a@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=12),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 12,
        "id": generate_uint15_hash("untok_sen2l2a@224"),
        "path": "S2L2A_untokenized",
    },
    "untok_sen2l1c@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=13),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 13,
        "id": generate_uint15_hash("untok_sen2l1c@224"),
        "path": "S2L1C_untokenized",
    },
    "untok_sen2rgb@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=13),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 3,
        "id": generate_uint15_hash("untok_sen2rgb@224"),
        "path": "S2RGB_untokenized",
    },
    "untok_sen1grd@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 2,
        "id": generate_uint15_hash("untok_sen1grd@224"),
        "path": "S1GRD_untokenized",
    },
    "untok_sen1rtc@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 2,
        "id": generate_uint15_hash("untok_sen1rtc@224"),
        "path": "S1RTC_untokenized",
    },
    "tok_sen1grd@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 15360,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=15360),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=15360),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_sen1grd@224"),
        "pretokenized": True,
        "path": "S1GRD_tokens",
    },
    "tok_sen1rtc@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 15360,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=15360),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=15360),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_sen1rtc@224"),
        "pretokenized": True,
        "path": "S1RTC_tokens",
    },
    "tok_sen2l2a@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 15360,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=15360),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=15360),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_sen2l2a@224"),
        "pretokenized": True,
        "path": "S2L2A_tokens",
    },
    "tok_lulc@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 4375,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=4375),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=4375),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_lulc@224"),
        "pretokenized": True,
        "path": "LULC_tokens",
    },
    "untok_dem@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=1),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("untok_dem@224"),
        "path": "DEM_untokenized",
    },
    "tok_dem@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 15360,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=15360),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=15360),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_dem@224"),
        "pretokenized": True,
        "path": "DEM_tokens",
    },
    "tok_ndvi@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 15360,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=15360),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=15360),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_ndvi@224"),
        "pretokenized": True,
        "path": "NDVI_tokens",
    },
    ### Natural image/text domains
    "rgb@224": {
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=3),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 3,
        "id": generate_uint15_hash("rgb@224"),
        "path": "rgb",
    },
    "caption": {
        "vocab_size": 30_000,
        "encoder_embedding": partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        "decoder_embedding": partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        "min_tokens": 0,
        "max_tokens": 256,
        "type": "seq",
        "id": generate_uint15_hash("caption"),
        "path": "captions_txt",
    },
    "coords": {
        "vocab_size": 30_000,
        "encoder_embedding": partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=4, padding_idx=0),
        "decoder_embedding": partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=4, padding_idx=0),
        "min_tokens": 0,
        "max_tokens": 4,  # buffer for EOD token
        "type": "seq",
        "id": generate_uint15_hash("coords"),
        "path": "coords",
    },
    "det": {
        "vocab_size": 30_000,
        "encoder_embedding": partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        "decoder_embedding": partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        "min_tokens": 0,
        "max_tokens": 256,
        "type": "seq",
        "id": generate_uint15_hash("det"),
    },
    "tok_rgb@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 16384,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=16384),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=16384),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_rgb@224"),
        "pretokenized": True,
    },
    "tok_depth@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 8192,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_depth@224"),
        "pretokenized": True,
    },
    "tok_normal@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 8192,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_normal@224"),
        "pretokenized": True,
    },
    "tok_semseg@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 4096,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=4096),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=4096),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_semseg@224"),
        "pretokenized": True,
    },
    "tok_clip@224": {
        "input_size": 224,
        "patch_size": 16,
        "vocab_size": 8192,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "id": generate_uint15_hash("tok_clip@224"),
        "pretokenized": True,
    },
    ### 224->448 super resolution modalities
    "rgb@448": {
        "input_size": 448,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=3),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 784
        "type": "img",
        "num_channels": 3,
        "id": generate_uint15_hash("rgb@448"),
        "path": "rgb",
    },
    "tok_rgb@448": {
        "input_size": 448,
        "patch_size": 16,
        "vocab_size": 16384,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=16384),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=16384),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 784
        "type": "img",
        "id": generate_uint15_hash("tok_rgb@448"),
        "pretokenized": True,
    },
    "tok_depth@448": {
        "input_size": 448,
        "patch_size": 16,
        "vocab_size": 8192,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 784
        "type": "img",
        "id": generate_uint15_hash("tok_depth@448"),
        "pretokenized": True,
    },
    "tok_normal@448": {
        "input_size": 448,
        "patch_size": 16,
        "vocab_size": 8192,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 784
        "type": "img",
        "id": generate_uint15_hash("tok_normal@448"),
        "pretokenized": True,
    },
    "tok_semseg@448": {
        "input_size": 448,
        "patch_size": 16,
        "vocab_size": 4096,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=4096),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=4096),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 784
        "type": "img",
        "id": generate_uint15_hash("tok_semseg@448"),
        "pretokenized": True,
    },
    "tok_clip@448": {
        "input_size": 448,
        "patch_size": 16,
        "vocab_size": 8192,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 784
        "type": "img",
        "id": generate_uint15_hash("tok_clip@448"),
        "pretokenized": True,
    },
}
