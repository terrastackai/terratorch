"""This module contains logic for tiled inference.
This does some additional things over the obvious fold -> predict -> unfold logic,
e.g. cropping out areas around model prediction to reduce artifacts

It additionally rebatches after the fold operation to gain speed up.
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import torch
import tqdm

from terratorch.models.utils import pad_images


# TODO: Remove TiledInferenceParameters in version 1.3.
@dataclass
class TiledInferenceParameters:
    """
    Parameters to be used for inference. Deprecated, please us directly pass the parameters to tiled_inference.
    """

    h_crop: int = (224,)
    h_stride: int = (200,)
    w_crop: int = (224,)
    w_stride: int = (200,)
    delta: int = (4,)
    average_patches: bool = (True,)
    blend_overlaps: bool = (True,)
    batch_size: int = (16,)
    verbose: bool = (False,)


def get_blend_mask(
    h_crop: int = 224,
    h_stride: int = 200,
    w_crop: int = 224,
    w_stride: int = 200,
    delta: int = 0,
) -> torch.Tensor:
    overlap_w = min(w_crop // 2, w_crop - w_stride) - delta
    overlap_h = min(h_crop // 2, h_crop - h_stride) - delta

    # Vertical window
    y_pos = torch.arange(h_crop - 2 * delta, device="cpu")
    y = torch.ones_like(y_pos, dtype=torch.float)
    if overlap_h:
        # ramp = (torch.cos(math.pi * (y_pos[:overlap_w] + 1) / (overlap_w + 1) / 2))
        ramp = torch.cos(math.pi * (y_pos[:overlap_h] + 1) / (overlap_h + 1)) / 2 + 0.5
        y[:overlap_h] = ramp.flip(0)  # top edge
        y[-overlap_h:] = ramp  # bottom edge

    # Horizontal window
    x_pos = torch.arange(w_crop - 2 * delta, device="cpu")
    x = torch.ones_like(x_pos, dtype=torch.float)
    if overlap_w:
        # ramp = (torch.cos(math.pi * (x_pos[:overlap_w] + 1) / (overlap_w + 1) / 2))
        ramp = torch.cos(math.pi * (x_pos[:overlap_w] + 1) / (overlap_w + 1)) / 2 + 0.5
        x[:overlap_w] = ramp.flip(0)  # left edge
        x[-overlap_w:] = ramp  # right edge

    # Get outer product (2D mask)
    mask = y[:, None] * x[None, :]

    # Add buffer to ensure every pixel gets a generation
    mask += 1e-6

    return mask


@dataclass
class InferenceInput:
    batch: int
    input_coords: tuple[slice, slice]
    input_data: torch.Tensor
    blend_mask: torch.Tensor
    output_crop: None | tuple[slice, slice]


def get_input_chips(
    input_batch, h_crop, h_stride, w_crop, w_stride, delta, blend_overlaps, padding
) -> list[InferenceInput]:
    """
    Create input chips of type InferenceInput for tiled inference. These contain:
      0. batch
      1. Coordinates where this should end up in the preds
      2. output/input
      3. Blend mask for weighting the edges of the chips
      4. Optionally, for inputs, how to crop the output
    """
    if padding:
        w_pad, h_pad = delta, delta

        if len(input_batch.shape) > 4:
            # Ignore additional during padding (e.g. with multi-temporal input)
            add_dim = [0, 0] * (len(input_batch.shape) - 4)
        else:
            add_dim = []

        input_batch = torch.nn.functional.pad(input_batch, (w_pad, w_pad, h_pad, h_pad, *add_dim), mode=padding)

        border_output_crop = (slice(delta, h_crop - delta), slice(delta, w_crop - delta))
    else:
        border_output_crop = None
    inner_output_crop = (slice(delta, h_crop - delta), slice(delta, w_crop - delta))

    # Blend overlapping areas using weighted masks
    if blend_overlaps:
        inner_blend_mask = get_blend_mask(h_crop, h_stride, w_crop, w_stride, delta)
        border_blend_mask = inner_blend_mask if padding else get_blend_mask(h_crop, h_stride, w_crop, w_stride)
    else:
        inner_blend_mask = torch.ones((h_crop - 2 * delta, w_crop - 2 * delta), device="cpu", dtype=torch.float)
        border_blend_mask = (
            inner_blend_mask if padding else torch.ones((h_crop, w_crop), device="cpu", dtype=torch.float)
        )

    input_batch_size = input_batch.shape[0]
    h_img, w_img = input_batch.shape[-2:]

    # Stage 1: deal with border patches (using border settings and subtract delta from coords only if padding is used)
    # Deal with patches near the right border
    coordinates_and_inputs: list[InferenceInput] = []
    for i in range(0, h_img - h_crop - 1, h_stride):
        patch = input_batch[..., i : i + h_crop, w_img - w_crop : w_img]
        coordinates_and_inputs += [
            InferenceInput(
                b,
                (slice(i + delta, i + h_crop - delta), slice(w_img - w_crop + delta, w_img - delta))
                if padding
                else (slice(i, i + h_crop), slice(w_img - w_crop, w_img)),
                patch[b],
                border_blend_mask,
                border_output_crop,
            )
            for b in range(input_batch_size)
        ]

    # Deal with patches near the bottom of the image
    for i in range(0, w_img - w_crop - 1, w_stride):
        patch = input_batch[..., h_img - h_crop : h_img, i : i + w_crop]
        coordinates_and_inputs += [
            InferenceInput(
                b,
                (slice(h_img - h_crop + delta, h_img - delta), slice(i + delta, i + w_crop - delta))
                if padding
                else (slice(h_img - h_crop, h_img), slice(i, i + w_crop)),
                patch[b],
                border_blend_mask,
                border_output_crop,
            )
            for b in range(input_batch_size)
        ]

    # Deal with last patches at the right bottom of the image
    patch = input_batch[..., h_img - h_crop : h_img, w_img - w_crop : w_img]
    coordinates_and_inputs += [
        InferenceInput(
            b,
            (slice(h_img - h_crop + delta, h_img - delta), slice(w_img - w_crop + delta, w_img - delta))
            if padding
            else (slice(h_img - h_crop, h_img), slice(w_img - w_crop, w_img)),
            patch[b],
            border_blend_mask,
            border_output_crop,
        )
        for b in range(input_batch_size)
    ]

    for row in range(0, h_img - h_crop - 1, h_stride):
        for col in range(0, w_img - w_crop - 1, w_stride):
            patch = input_batch[..., row : row + h_crop, col : col + w_crop]
            if row == 0 or col == 0:
                # Add patches along the left and top of the image
                coordinates_and_inputs += [
                    InferenceInput(
                        b,
                        (slice(row + delta, row + h_crop - delta), slice(col + delta, col + w_crop - delta))
                        if padding
                        else (slice(row, row + h_crop), slice(col, col + w_crop)),
                        patch[b],
                        border_blend_mask,
                        border_output_crop,
                    )
                    for b in range(input_batch_size)
                ]
            else:
                # Stage 2: process internally with patch overlap
                coordinates_and_inputs += [
                    InferenceInput(
                        b,
                        (slice(row + delta, row + h_crop - delta), slice(col + delta, col + w_crop - delta)),
                        patch[b],
                        inner_blend_mask,
                        inner_output_crop,
                    )
                    for b in range(input_batch_size)
                ]

    return coordinates_and_inputs


def prepare_tiled_inference_input(
    input_batch: torch.Tensor,
    out_channels: int | None = None,
    inference_parameters: TiledInferenceParameters | None = None,
    crop: int = 224,
    stride: int = 192,
    delta: int = 8,
    blend_overlaps: bool = True,
    h_crop: int | None = None,
    w_crop: int | None = None,
    h_stride: int | None = None,
    w_stride: int | None = None,
    padding: str | bool = "reflect",
    device: str | None = None,
) -> tuple[list[InferenceInput], Callable[[torch.Tensor], dict | torch.Tensor], int, int, int, str | torch.device]:

    if inference_parameters is not None:
        # TODO: Remove inference_parameters in later version 1.3.
        warnings.warn(
            "Using inference_parameters and ignoring other parameters."
            "The parameter `inference_parameters` is deprecated and is removed in version 1.3, "
            "please pass the parameters directly to `tiled_inference`. ",
            DeprecationWarning,
        )
        h_crop = inference_parameters.h_crop
        h_stride = inference_parameters.h_stride
        w_crop = inference_parameters.w_crop
        w_stride = inference_parameters.w_stride
        delta = inference_parameters.delta
        blend_overlaps = inference_parameters.blend_overlaps

    if isinstance(input_batch, dict):
        # Handle dict inputs for tiled inference
        modalities, tensors = list(input_batch.keys()), list(input_batch.values())

        # Check that all values in dict are tensors and have a same image shape
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise ValueError("input for tiled inference must be either a torch.Tensor or a dict of torch.Tensors")
        img_shapes = [t.shape[-2:] for t in tensors]
        if len(set(img_shapes)) != 1:
            raise ValueError(
                f"Tensors in input dict must have the same height and width for tiled inference, "
                f"found {dict(zip(modalities, img_shapes))}"
            )
        t_dims = [len(t.shape) for t in tensors]
        if len(set(t_dims)) != 1:
            raise ValueError(
                f"Tensors in input dict must have the same number of dimensions for tiled inference, "
                f"found {dict(zip(modalities, t_dims, strict=False))}"
            )

        # Tiled inference is implemented for single tensors.
        # We concatenate all tensors and reshape them before the model forward
        t_dims = t_dims[0]
        if t_dims == 4:  # B, C, H, W
            channel_length = [t.shape[-3] for t in tensors]
            channel_start = torch.tensor([0] + channel_length).cumsum(0)
            input_batch = torch.concat(tensors, dim=-3)
        elif t_dims == 5:  # B, C, T, H, W
            channel_length = [t.shape[-4] for t in tensors]
            channel_start = torch.tensor([0] + channel_length).cumsum(0)
            input_batch = torch.concat(tensors, dim=-4)
        else:
            raise ValueError("Tensors must have 4 or 5 dimensions")

        def tensor_reshape(t):
            # Convert tensor back to dict of tensors
            if t_dims == 4:  # B, C, H, W
                out = {m: t[..., s : s + l, :, :] for m, s, l in zip(modalities, channel_start, channel_length)}
            elif t_dims == 5:  # B, C, T, H, W
                out = {m: t[..., s : s + l, :, :, :] for m, s, l in zip(modalities, channel_start, channel_length)}
            return out

    elif isinstance(input_batch, torch.Tensor):
        # Dummy function if input is a tensor
        tensor_reshape = lambda x: x
    else:
        raise ValueError("input for tiled inference must be either a torch.Tensor or a dict of torch.Tensors")

    input_batch_size = input_batch.shape[0]
    h_img, w_img = input_batch.shape[-2:]
    ret_device = device or input_batch.device

    # Move inputs to CPU to avoid out-of-memory errors
    input_batch = input_batch.cpu()

    h_crop = h_crop or crop
    w_crop = w_crop or crop
    h_stride = h_stride or stride
    w_stride = w_stride or stride

    if (h_crop - h_stride) // 2 < delta or (w_crop - w_stride) // 2 < delta:
        # Ensure that every pixel is covered
        delta = min((h_crop - h_stride) // 2, (w_crop - w_stride) // 2)
        warnings.warn(f"Tiled inference: delta is higher than overlap, reducing delta to {delta}.")

    if out_channels is not None:
        warnings.warn(
            "out_channels is deprecated and automatically selected after first forward pass.", DeprecationWarning
        )

    # Get smaller inputs
    coordinates_and_inputs = get_input_chips(
        input_batch, h_crop, h_stride, w_crop, w_stride, delta, blend_overlaps, padding
    )

    return coordinates_and_inputs, tensor_reshape, input_batch_size, h_img, w_img, ret_device, delta


def generate_tiled_inference_output(
    outputs,
    input_batch_size: int,
    h_img: int,
    w_img: int,
    delta: int,
    padding: str | bool = "reflect",
    average_patches: bool = True,
) -> torch.Tensor:
    preds: torch.Tensor | None = None
    preds_count: torch.Tensor | None = None

    for batch_input, predicted in outputs:
        if preds is None:
            # Initialize preds based on first output
            out_channels = 1 if len(predicted.shape) == 2 else predicted.shape[0]
            if padding:
                # Add padding areas to align with input indexes
                preds = torch.zeros((input_batch_size, out_channels, h_img + (2 * delta), w_img + (2 * delta)))
                preds_count = torch.zeros(input_batch_size, h_img + (2 * delta), w_img + (2 * delta))
            else:
                preds = torch.zeros((input_batch_size, out_channels, h_img, w_img))
                preds_count = torch.zeros(input_batch_size, h_img, w_img)
        if batch_input.output_crop is not None:
            predicted = predicted[..., batch_input.output_crop[0], batch_input.output_crop[1]]
        if average_patches:
            preds[
                batch_input.batch,
                :,
                batch_input.input_coords[0],
                batch_input.input_coords[1],
            ] += predicted * batch_input.blend_mask
        else:
            preds[
                batch_input.batch,
                :,
                batch_input.input_coords[0],
                batch_input.input_coords[1],
            ] = predicted

        preds_count[
            batch_input.batch,
            batch_input.input_coords[0],
            batch_input.input_coords[1],
        ] += batch_input.blend_mask

    if padding:
        # Remove padded areas
        preds = preds[..., delta:-delta, delta:-delta]
        preds_count = preds_count[..., delta:-delta, delta:-delta]
    if (preds_count == 0).sum() != 0:
        msg = "Some pixels did not receive a classification!"
        raise RuntimeError(msg)

    output = preds

    if average_patches:
        output = output / preds_count.unsqueeze(1)

    return output


def tiled_inference(
    model_forward: Callable,
    input_batch: torch.Tensor,
    out_channels: int | None = None,
    inference_parameters: TiledInferenceParameters | None = None,
    crop: int = 224,
    stride: int = 192,
    delta: int = 8,
    h_crop: int | None = None,
    w_crop: int | None = None,
    h_stride: int | None = None,
    w_stride: int | None = None,
    average_patches: bool = True,
    blend_overlaps: bool = True,
    batch_size: int = 16,
    verbose: bool = False,
    padding: str | bool = "reflect",
    device: str | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Divide an image into (potentially) overlapping chips and perform inference on them.
    Additionally, re-batch for varibale GPU utilization defined by crop size and batch_size.
    The overlap between chips is defined with: crop - stride - 2 * delta.

    Args:
        model_forward (Callable): Callable that return the output of the model.
        input_batch (torch.Tensor): Input batch to be processed
        out_channels (int): Number of output channels
        inference_parameters (TiledInferenceParameters): Parameters to be used for inference.
            Deprecated, please us directly pass the parameters to tiled_inference.
        crop (int): height and width of the smaller chips. Ignored if h_crop or w_crop is provided. Defaults to 224.
        stride (int): size of the stride. Ignored if h_stride or w_stride is provided. Defaults to 192.
        delta (int): size of the border cropped from each chip. Defaults to 8.
        h_crop (int, optional): height of the smaller chips.
        w_crop (int, optional): width of the smaller chips.
        h_stride (int, optional): size of the stride on the y-axis.
        w_stride (int, optional): size of the stride on the x-axis.
        average_patches (bool): Whether to average the overlapping regions. Defaults to True.
        batch_size (int): Number of chips per forward pass. Defaults to 16.
        padding (str | bool): Padding mode for input image to reduce artefacts on edges. Deactivate padding with False.
            Defaults to reflect.
    Returns:
        torch.Tensor: The result of the inference
    """

    coordinates_and_inputs, tensor_reshape, input_batch_size, h_img, w_img, device, delta = (
        prepare_tiled_inference_input(
            input_batch=input_batch,
            out_channels=out_channels,
            inference_parameters=inference_parameters,
            crop=crop,
            stride=stride,
            delta=delta,
            h_crop=h_crop,
            w_crop=w_crop,
            h_stride=h_stride,
            w_stride=w_stride,
            blend_overlaps=blend_overlaps,
            padding=padding,
            device=device,
        )
    )

    outputs: list[tuple[InferenceInput, torch.Tensor]] = []
    # NOTE: the output may be SLIGHTLY different using batched inputs because of layers such as nn.LayerNorm
    # During inference, these layers compute batch statistics that affect the output.
    # However, this should still be correct.
    with torch.no_grad():
        for start in tqdm.tqdm(
            range(0, len(coordinates_and_inputs), batch_size), desc="Tiled inference", disable=not verbose
        ):
            end = min(len(coordinates_and_inputs), start + batch_size)
            batch = coordinates_and_inputs[start:end]
            tensor_input = torch.stack([b.input_data for b in batch], dim=0)
            tensor_input = tensor_input.to(device)
            tensor_input = tensor_reshape(tensor_input)  # Optional reshaping for inputs other than plain tensors
            output = model_forward(tensor_input, **kwargs).cpu()
            output = [output[i] for i in range(len(batch))]

            outputs.extend(list(zip(batch, output, strict=True)))

    output = generate_tiled_inference_output(
        outputs=outputs,
        input_batch_size=input_batch_size,
        h_img=h_img,
        w_img=w_img,
        delta=delta,
        padding=padding,
        average_patches=average_patches,
    )

    return output.to(device)
