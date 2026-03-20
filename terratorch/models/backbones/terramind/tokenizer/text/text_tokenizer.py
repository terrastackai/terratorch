import re
import warnings

import torch
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from torch import nn


def build_blank_wordpiece():
    # Minimal tokens needed for testing
    vocab = {"[PAD]": 0, "[UNK]": 1, "[SOS]": 2, "[EOS]": 3, "[S_0]": 4, "[S_1]": 5, "[S_2]": 6}
    added_tokens = [AddedToken(w, normalized=False, special=True) for w in vocab.keys()]
    tok = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tok.normalizer = BertNormalizer()
    tok.pre_tokenizer = BertPreTokenizer()
    tok.decoder = WordPieceDecoder()
    tok.add_tokens(added_tokens)
    return tok


def capitalize_sentences(text):
    # Split text into sentences using a regex that looks for sentence end punctuation
    sentences = re.split("([.!?] *)", text)
    capitalized = "".join([s.capitalize() for s in sentences])
    return capitalized


class CaptionTokenizer(nn.Module):
    def __init__(self, tokenizer_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tokenizer_file is not None:
            self.text_tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            self.text_tokenizer = build_blank_wordpiece()  # un-trained
        self.text_tokenizer.enable_padding()

    def encode(self, text: list[str], device: torch.device, eos_id=3, *args, **kwargs) -> dict[str, torch.Tensor]:
        """
        Args:
            text list[str]: Text to be tokenized
            device: torch.device
        Returns:
            dict for generation sampler input
        """
        # Add start token
        text = [t + " [S_1]" for t in text]

        # Tokenize
        tok_ids = [t.ids for t in self.text_tokenizer.encode_batch(text, add_special_tokens=True)]

        # Add EOS token
        tok_ids = [t + [eos_id] for t in tok_ids]

        tok_ids = torch.tensor(tok_ids, device=device)

        text_dict = {
            "tensor": tok_ids,
            "input_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
            "target_mask": torch.ones_like(tok_ids, dtype=torch.bool, device=device),
            "decoder_attention_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
        }

        return text_dict

    def decode_text(self, mod_dict, key="caption"):
        """
        Decodes a text sequence from a model dictionary.

        Args:
            mod_dict (dict): Model output dictionary.
            key (str): Key of the text modality to decode.
        """
        decoded_texts = []

        for i in range(mod_dict[key]["tensor"].shape[0]):
            seq = mod_dict[key]["tensor"][i]
            seq = seq[mod_dict[key]["input_mask"][i] == 0]
            seq = seq.tolist()

            merged_text = self.text_tokenizer.decode(seq, skip_special_tokens=False)

            decoded_texts.append(merged_text.replace(" [EOS]", ""))

        decoded_texts = list(map(capitalize_sentences, decoded_texts))

        return decoded_texts


class CoordsTokenizer(nn.Module):
    def __init__(self, tokenizer_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tokenizer_file is not None:
            self.text_tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            self.text_tokenizer = build_blank_wordpiece()  # un-trained

    def encode(self, coords: torch.Tensor, *args, **kwargs) -> dict[str, torch.Tensor]:
        """
        Encodes coords to token ids. Returns tuple to be compatible with image tokenizers.

        Args:
            coords (torch.Tensor): Center coordinates of image with shape [B, 2] with [lon, lat] values in second dim.

        Returns:
            tok_ids(tuple[torch.Tensor]): Token ids with shape [B, 2]
        """
        if coords.shape[1] != 2:
            raise ValueError(
                f"Expect coords data in shape [batch, 2] with [lon, lat] values, got coords with shape {coords.shape}."
            )

        # Align coords with 0.25 degree grid
        coords = (coords * 4).round() / 4
        device = coords.device

        coords = [f"lat={c[1].item():.2f} lon={c[0].item():.2f} [EOS]" for c in coords]

        # Tokenize
        tok_ids = [t.ids for t in self.text_tokenizer.encode_batch(coords, add_special_tokens=True)]
        tok_ids = torch.tensor(tok_ids, device=device)

        coords_dict = {
            "tensor": tok_ids,
            "input_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
            "target_mask": torch.ones_like(tok_ids, dtype=torch.bool, device=device),
            "decoder_attention_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
        }

        return coords_dict

    def decode_text(self, mod_dict, key="coords"):
        """
        Decodes a coordinate sequence from a modality dictionary.

        Args:
            mod_dict (dict): Model output dictionary.
            key (str): Key of the coords modality to decode.
        """
        coords = []

        B = mod_dict[key]["tensor"].shape[0]

        for i in range(B):
            seq = mod_dict[key]["tensor"][i].tolist()[:2]

            text = self.text_tokenizer.decode(seq, skip_special_tokens=False)

            try:
                lat, lon = text.split(" ")
                coords.append([float(lon.strip("lon=")), float(lat.strip("lat="))])
            except Exception as e:
                warnings.warn(
                    f"Coordinate generation did not work correctly, generated text: {text} (Error: {e}). Returning NaN."
                )
                coords.append([torch.nan, torch.nan])

        return coords
