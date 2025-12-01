# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Module containing generic dataset classes"""

import glob
import logging
import random
import warnings
import os
import re
import torch
import pandas as pd
from abc import ABC
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib as mpl
import numpy as np
import rioxarray
import xarray as xr
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import HLSBands, default_transform, filter_valid_files, generate_bands_intervals
from terratorch.datasets.transforms import MultimodalTransforms

logger = logging.getLogger("terratorch")


def load_table_data(file_path: str | Path) -> pd.DataFrame:
    file_path = str(file_path)
    if file_path.endswith("parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith("csv"):
        df = pd.read_csv(file_path, index_col=0)
    else:
        raise Exception(f"Unrecognized file type: {file_path}. Only parquet and csv are supported.")
    return df


class MultimodalToTensor:
    def __init__(self, modalities):
        self.modalities = modalities

    def __call__(self, d):
        new_dict = {}
        for k, v in d.items():
            if not isinstance(v, np.ndarray):
                new_dict[k] = v
            else:
                if k in self.modalities and len(v.shape) >= 3:  # Assuming raster modalities with 3+ dimensions
                    if len(v.shape) <= 4:
                        v = np.moveaxis(v, -1, 0)  # C, H, W or C, T, H, W
                    elif len(v.shape) == 5:
                        v = np.moveaxis(v, -1, 1)  # B, C, T, H, W
                    else:
                        raise ValueError(f"Unexpected shape for {k}: {v.shape}")
                new_dict[k] = torch.from_numpy(v)
        return new_dict


class GenericMultimodalDataset(NonGeoDataset, ABC):
    """
    This is a generic dataset class to be used for instantiating datasets from arguments.
    Ideally, one would create a dataset class specific to a dataset.
    """

    def __init__(
        self,
        data_root: dict[str, Path | str],
        label_data_root: Path | str | list[Path | str] | None = None,
        image_grep: dict[str, str] | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        image_modalities: list[str] | None = None,
        rgb_modality: str | None = None,
        rgb_indices: list[int] | None = None,
        allow_missing_modalities: bool = False,
        allow_substring_file_names: bool = True,
        dataset_bands: dict[str, list] | None = None,
        output_bands: dict[str, list] | None = None,
        constant_scale: dict[str, float] = None,
        transform: A.Compose | dict | None = None,
        no_data_replace: float | None = None,
        no_label_replace: float | None = -1,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        channel_position: int = -3,
        scalar_label: bool = False,
        data_with_sample_dim: bool = False,
        concat_bands: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Constructor

        Args:
            data_root (dict[Path]): Dictionary of paths to data root directory or csv/parquet files with image-level
                data, with modalities as keys.
            label_data_root (Path, optional): Path to data root directory with labels or csv/parquet files with
                image-level labels. Needs to be specified for supervised tasks. Set to None for prediction mode.
            image_grep (dict[str], optional): Dictionary with regular expression appended to data_root to find input
                images, with modalities as keys. Defaults to "*". Ignored when allow_substring_file_names is False.
            label_grep (str, optional): Regular expression appended to label_data_root to find labels or mask files.
                Defaults to "*". Ignored when allow_substring_file_names is False.
            split (Path, optional): Path to file containing samples prefixes to be used for this split.
                The file can be a csv/parquet file with the prefixes in the index or a txt file with new-line separated
                sample prefixes. File names must be exact matches if allow_substring_file_names is False. Otherwise,
                files are searched using glob with the form Path(data_root).glob(prefix + [image or label grep]).
                If not specified, search samples based on files in data_root. Defaults to None.
            image_modalities(list[str], optional): List of pixel-level raster modalities. Defaults to data_root.keys().
                The difference between all modalities and image_modalities are non-image modalities which are treated
                differently during the transforms and are not modified but only converted into a tensor if possible.
            rgb_modality (str, optional): Modality used for RGB plots. Defaults to first modality in data_root.keys().
            rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
            allow_missing_modalities (bool, optional): Allow missing modalities during data loading. Defaults to False.
            allow_substring_file_names (bool, optional): Allow substrings during sample identification by adding
                image or label grep to the sample prefixes. If False, treats sample prefixes as full file names.
                If True and no split file is provided, considers the file stem as prefix, otherwise the full file name.
                Defaults to True.
            dataset_bands (dict[list], optional): Bands present in the dataset, provided in a dictionary with modalities
                as keys. This parameter names input channels (bands) using HLSBands, ints, int ranges, or strings, so
                that they can then be referred to by output_bands. Needs to be superset of output_bands. Can be a subset
                of all modalities. Defaults to None.
            output_bands (dict[list], optional): Bands that should be output by the dataset as named by dataset_bands,
                provided as a dictionary with modality keys. Can be subset of all modalities. Defaults to None.
            constant_scale (dict[float]): Factor to multiply data values by, provided as a dictionary with modalities as
                keys. Can be subset of all modalities. Defaults to None.
            transform (Albumentations.Compose | dict | None): Albumentations transform to be applied to all image
                modalities (transformation are shared between image modalities, e.g., similar crop or rotation).
                Should end with ToTensorV2(). If used through the generic_data_module, should not include normalization.
                Not supported for multi-temporal data. The transform is not applied to non-image data, which is only
                converted to tensors if possible. If dict, can include multiple transforms per modality which are
                applied separately (no shared parameters between modalities).
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input data with this value.
                If None, does no replacement. Defaults to None.
            no_label_replace (float | None): Replace nan values in label with this value.
                If none, does no replacement. Defaults to -1.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Only works with image modalities. Is only applied to modalities with defined dataset_bands.
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            channel_position (int): Position of the channel dimension in the image modalities. Defaults to -3.
            scalar_label (bool): Returns a image mask if False or otherwise the raw labels. Defaults to False.
            concat_bands (bool): Concatenate all image modalities along the band dimension into a single "image", so
                that it can be processed by single-modal models. Concatenate in the order of provided modalities.
                Works with image modalities only. Does not work with allow_missing_modalities. Defaults to False.
        """

        super().__init__()

        self.split_file = split
        self.modalities = list(data_root.keys())
        assert "mask" not in self.modalities, "Modality cannot be called 'mask'."
        self.image_modalities = image_modalities or self.modalities
        self.non_image_modalities = list(set(self.modalities) - set(image_modalities))
        self.modalities = self.image_modalities + self.non_image_modalities  # Ensure image modalities to be first

        # Order by modalities and convert path strings to lists as the code expects a list of paths per modality
        data_root = {m: data_root[m] for m in self.modalities}

        self.constant_scale = constant_scale or {}
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.reduce_zero_label = reduce_zero_label
        self.expand_temporal_dimension = expand_temporal_dimension
        self.channel_position = channel_position
        self.scalar_label = scalar_label
        self.data_with_sample_dim = data_with_sample_dim
        self.concat_bands = concat_bands
        assert not self.concat_bands or len(self.non_image_modalities) == 0, (
            f"concat_bands can only be used with image modalities, "
            f"but non-image modalities are given: {self.non_image_modalities}"
        )
        assert (
            not self.concat_bands or not allow_missing_modalities
        ), "concat_bands cannot be used with allow_missing_modalities."

        if self.expand_temporal_dimension and dataset_bands is None:
            msg = "Please provide dataset_bands when expand_temporal_dimension is True"
            raise Exception(msg)

        if scalar_label:
            self.non_image_modalities += ["label"]

        # Load samples based on split file
        if self.split_file is not None:
            if not os.path.isfile(self.split_file):
                raise FileNotFoundError(f"Split file {self.split_file} does not exist.")
            elif str(self.split_file).endswith(".txt"):
                with open(self.split_file) as f:
                    split = f.readlines()
                valid_files = [rf"{substring.strip()}" for substring in split]
            else:
                valid_files = list(load_table_data(self.split_file).index)
            if len(valid_files) == 0:
                raise ValueError(f"No sample candidates (file prefixes) found in split file {self.split_file}.")

        else:
            image_files = {}
            for m, m_paths in data_root.items():
                image_files[m] = sorted(glob.glob(os.path.join(m_paths, image_grep[m])))

            def get_file_id(file_name, mod):
                glob_as_regex = '^' + ''.join('(.*?)' if ch == '*' else re.escape(ch)
                                              for ch in image_grep[mod]) + '$'
                stem = re.match(glob_as_regex, os.path.basename(file_name)).group(1)
                if allow_substring_file_names:
                    # Remove file extensions
                    stem = os.path.splitext(stem)[0]
                return stem

            if allow_missing_modalities:
                valid_files = list(set([get_file_id(file, mod)
                                        for mod, files in image_files.items()
                                        for file in files
                                        ]))
            else:
                valid_files = [get_file_id(file, self.modalities[0]) for file in image_files[self.modalities[0]]]

        self.samples = []
        num_modalities = len(self.modalities) + int(label_data_root is not None)
        if len(valid_files) == 0:
            # Provide additional information if no candidates are found
            image_files = {m: f[:3] for m, f in image_files.items()}
            raise ValueError(f"No sample candidates (file prefixes) found for multimodal dataset. "
                             f"Please review files and parameters.\n"
                             f"data_root: {data_root}\n"
                             f"image_grep: {image_grep}\n"
                             f"allow_missing_modalities: {allow_missing_modalities}\n"
                             f"File examples in data_root: {image_files}\n")

        # Check for parquet and csv files with modality data and read the file
        for m, m_path in data_root.items():
            if os.path.isfile(m_path):
                data_root[m] = load_table_data(m_path)
                # Check for some sample keys
                if not any(f in data_root[m].index for f in valid_files[:100]):
                    warnings.warn(f"Sample key expected in table index (first column) for {m} (file: {m_path}). "
                                  f"{valid_files[:3]+['...']} are not in index {list(data_root[m].index[:3])+['...']}.")
        if label_data_root is not None:
            if os.path.isfile(label_data_root):
                label_data_root = load_table_data(label_data_root)
                # Check for some sample keys
                if not any(f in label_data_root.index for f in valid_files[:100]):
                    warnings.warn(f"Keys expected in table index (first column) for labels (file: {label_data_root}). "
                                  f"The keys {valid_files[:3] + ['...']} are not in the index.")

        # Iterate over all files in split
        failed_candidates = []
        for file in valid_files:
            sample = {}
            # Iterate over all modalities
            for m, m_path in data_root.items():
                if isinstance(m_path, pd.DataFrame):
                    # Add tabular data to sample
                    sample[m] = m_path.loc[file].values
                elif allow_substring_file_names:
                    # Substring match with image_grep
                    m_files = sorted(glob.glob(os.path.join(m_path, file + image_grep[m])))
                    if m_files:
                        sample[m] = m_files[-1]
                        if len(m_files) > 1:
                            warnings.warn(f"Found multiple matching files for sample {file} and grep {image_grep[m]}: "
                                          f"{m_files}. Selecting last one. "
                                          f"Consider changing data structure or parameters for unique selection.")
                else:
                    # Exact match
                    file_path = os.path.join(m_path, file)
                    if os.path.exists(file_path):
                        sample[m] = file_path

            if label_data_root is not None:
                if isinstance(label_data_root, pd.DataFrame):
                    # Add tabular data to sample
                    sample["mask"] = label_data_root.loc[file].values
                elif allow_substring_file_names:
                    # Substring match with label_grep
                    l_files = sorted(glob.glob(os.path.join(label_data_root, file + label_grep)))
                    if l_files:
                        sample["mask"] = l_files[-1]
                else:
                    # Exact match
                    file_path = os.path.join(label_data_root, file)
                    if os.path.exists(file_path):
                        sample["mask"] = file_path
                if "mask" not in sample:
                    # Only add sample if mask is present
                    failed_candidates.append(sample)
                    continue

            if len(sample) == num_modalities or allow_missing_modalities:
                self.samples.append(sample)
            else:
                failed_candidates.append(sample)

        if len(self.samples) == 0:
            # Provide additional information if no multi-modal samples are found
            idx = random.sample(range(len(valid_files)), min(5, len(valid_files)))
            raise ValueError(f"No samples found for multimodal dataset. Please review files, path, and grep params.\n"
                             f"data_root: {data_root}\n"
                             f"image_grep: {image_grep}\n"
                             f"allow_missing_modalities: {allow_missing_modalities}\n"
                             f"Candidate prefixes: {', '.join([valid_files[i] for i in idx])}\n"
                             f"Sample candidate paths: {', '.join([failed_candidates[i] for i in idx])}")

        self.rgb_modality = rgb_modality or self.modalities[0]
        self.rgb_indices = rgb_indices or [0, 1, 2]

        if dataset_bands is not None:
            self.dataset_bands = {m: generate_bands_intervals(m_bands) for m, m_bands in dataset_bands.items()}
        else:
            self.dataset_bands = None
        if output_bands is not None:
            self.output_bands = {m: generate_bands_intervals(m_bands) for m, m_bands in output_bands.items()}
            for modality in self.modalities:
                if modality in self.output_bands and modality not in self.dataset_bands:
                    msg = f"If output bands are provided, dataset_bands must also be provided (modality: {modality})"
                    raise Exception(msg)  # noqa: PLE0101
        else:
            self.output_bands = {}

        self.filter_indices = {}
        if self.output_bands:
            for m in self.output_bands.keys():
                if m not in self.output_bands or self.output_bands[m] == self.dataset_bands[m]:
                    continue
                if len(set(self.output_bands[m]) & set(self.dataset_bands[m])) != len(self.output_bands[m]):
                    msg = f"Output bands must be a subset of dataset bands (Modality: {m})"
                    raise Exception(msg)

                self.filter_indices[m] = [self.dataset_bands[m].index(band) for band in self.output_bands[m]]

            if not self.channel_position:
                logger.warning(
                    "output_bands is defined but no channel_position is provided. "
                    "Channels must be in the last dimension, otherwise provide channel_position."
                )

        # If no transform is given, apply only to transform to torch tensor
        if isinstance(transform, A.Compose):
            self.transform = MultimodalTransforms(transform,
                                                  non_image_modalities=self.non_image_modalities + ['label']
                                                  if scalar_label else self.non_image_modalities)
        elif transform is None:
            self.transform = MultimodalToTensor(self.modalities)
        else:
            # Modality-specific transforms
            transform = {m: transform[m] if m in transform else default_transform for m in self.modalities}
            self.transform = MultimodalTransforms(transform, shared=False)

        # Ignore rasterio warning for not geo-referenced files
        import rasterio

        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
        warnings.filterwarnings("ignore", message="Dataset has no geotransform")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        output = {}
        if isinstance(index, tuple):
            # Load only sampled modalities instead of all modalities
            # (see sample_num_modalities in GenericMultiModalDataModule for details)
            index, modalities = index
            sample = {m: self.samples[index][m] for m in modalities}
        else:
            sample = self.samples[index]

        for modality, file in sample.items():
            data = self._load_file(
                file,
                nan_replace=self.no_label_replace if modality == "mask" else self.no_data_replace,
                modality=modality,
            )

            # Expand temporal dim
            if modality in self.filter_indices and self.expand_temporal_dimension:
                data = rearrange(
                    data, "(channels time) h w -> channels time h w", channels=len(self.dataset_bands[modality])
                )

            if modality == "mask" and not self.scalar_label:
                # tasks expect image masks without channel dim
                data = data[0]

            if modality in self.image_modalities and len(data.shape) >= 3 and self.channel_position:
                # to channels last (required by albumentations)
                data = np.moveaxis(data, self.channel_position, -1)

            if modality in self.filter_indices:
                data = data[..., self.filter_indices[modality]]

            if modality in self.constant_scale:
                data = data.astype(np.float32) * self.constant_scale[modality]

            if data.dtype == np.float64:
                data = data.astype(np.float32)

            output[modality] = data

        if "mask" in output:
            if self.reduce_zero_label:
                output["mask"] -= 1
            if self.scalar_label:
                output["label"] = output.pop("mask")

        if self.transform:
            output = self.transform(output)

        if self.concat_bands:
            # Concatenate bands of all image modalities
            data = [output.pop(m) for m in self.image_modalities if m in output]
            output["image"] = torch.cat(data, dim=1 if self.data_with_sample_dim else 0)
        else:
            # Tasks expect data to be stored in "image", moving modalities to image dict
            output["image"] = {m: output.pop(m) for m in self.modalities if m in output}

        output["filename"] = self.samples[index]

        return output

    def _load_file(self, path, nan_replace: int | float | None = None, modality: str | None = None) -> xr.DataArray:
        if isinstance(path, np.ndarray):
            # data was loaded from table and is saved in memory
            data = path
        elif path.endswith(".zarr") or path.endswith(".zarr.zip"):
            data = xr.open_zarr(path, mask_and_scale=True)
            data_var = modality if modality in data.data_vars else list(data.data_vars)[0]
            data = data[data_var].to_numpy()
        elif path.endswith(".npy"):
            data = np.load(path)
        else:
            data = rioxarray.open_rasterio(path, masked=True).to_numpy()

        if nan_replace is not None:
            data = np.nan_to_num(data, nan=nan_replace)
        return data

    def plot(self, sample: dict[str, torch.Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        image = sample["image"]
        if isinstance(image, dict):
            image = image[self.rgb_modality]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        image = image.take(self.rgb_indices, axis=0)
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)

        if "mask" in sample:
            mask = sample["mask"]
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=-1)
            # Convert masked regions to 0.
            mask = mask * -1 + 1
        else:
            mask = None

        if "prediction" in sample:
            prediction = sample["prediction"]
            if isinstance(image, dict):
                prediction = prediction[self.rgb_modality]
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.numpy()
            # Assuming reconstructed image
            prediction = prediction.take(self.rgb_indices, axis=0)
            prediction = np.transpose(prediction, (1, 2, 0))
            prediction = (prediction - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
            prediction = np.clip(prediction, 0, 1)
        else:
            prediction = None

        return self._plot_sample(
            image,
            mask=mask,
            prediction=prediction,
            suptitle=suptitle,
        )

    @staticmethod
    def _plot_sample(image, mask=None, prediction=None, suptitle=None):
        num_images = 1 + int(mask is not None) + int(prediction is not None)
        fig, ax = plt.subplots(1, num_images, figsize=(5*num_images, 5), layout="compressed")

        ax[0].axis("off")
        ax[0].imshow(image)

        if mask is not None:
            ax[1].axis("off")
            ax[1].imshow(image * mask)

        if prediction is not None:
            ax[num_images-1].axis("off")
            ax[num_images-1].imshow(prediction)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class GenericMultimodalSegmentationDataset(GenericMultimodalDataset):
    """GenericNonGeoSegmentationDataset"""

    def __init__(
        self,
        data_root: Path,
        num_classes: int,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        image_modalities: list[str] | None = None,
        rgb_modality: str | None = None,
        rgb_indices: list[str] | None = None,
        allow_missing_modalities: bool = False,
        allow_substring_file_names: bool = False,
        dataset_bands: dict[list] | None = None,
        output_bands: dict[list] | None = None,
        class_names: list[str] | None = None,
        constant_scale: dict[float] = 1.0,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = -1,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        channel_position: int = -3,
        concat_bands: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Constructor

        Args:
            data_root (dict[Path]): Dictionary of paths to data root directory or csv/parquet files with image-level
                data, with modalities as keys.
            num_classes (int): Number of classes.
            label_data_root (Path): Path to data root directory with mask files. Set to None for prediction mode.
            image_grep (dict[str], optional): Dictionary with regular expression appended to data_root to find input
                images, with modalities as keys. Defaults to "*". Ignored when allow_substring_file_names is False.
            label_grep (str, optional): Regular expression appended to label_data_root to find mask files.
                Defaults to "*". Ignored when allow_substring_file_names is False.
            split (Path, optional): Path to file containing samples prefixes to be used for this split.
                The file can be a csv/parquet file with the prefixes in the index or a txt file with new-line separated
                sample prefixes. File names must be exact matches if allow_substring_file_names is False. Otherwise,
                files are searched using glob with the form Path(data_root).glob(prefix + [image or label grep]).
                If not specified, search samples based on files in data_root. Defaults to None.
            image_modalities(list[str], optional): List of pixel-level raster modalities. Defaults to data_root.keys().
                The difference between all modalities and image_modalities are non-image modalities which are treated
                differently during the transforms and are not modified but only converted into a tensor if possible.
            rgb_modality (str, optional): Modality used for RGB plots. Defaults to first modality in data_root.keys().
            rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
            allow_missing_modalities (bool, optional): Allow missing modalities during data loading. Defaults to False.
            allow_substring_file_names (bool, optional): Allow substrings during sample identification by adding
                image or label grep to the sample prefixes. If False, treats sample prefixes as full file names.
                If True and no split file is provided, considers the file stem as prefix, otherwise the full file name.
                Defaults to True.
            dataset_bands (dict[list], optional): Bands present in the dataset, provided in a dictionary with modalities
                as keys. This parameter names input channels (bands) using HLSBands, ints, int ranges, or strings, so
                that they can then be referred to by output_bands. Needs to be superset of output_bands. Can be a subset
                of all modalities. Defaults to None.
            output_bands (dict[list], optional): Bands that should be output by the dataset as named by dataset_bands,
                provided as a dictionary with modality keys. Can be subset of all modalities. Defaults to None.
            class_names (list[str], optional): Names of the classes. Defaults to None.
            constant_scale (dict[str, float]): Factor to multiply data values by, provided as a dictionary with modalities as
                keys. Can be subset of all modalities. Defaults to None.
            transform (Albumentations.Compose | dict | None): Albumentations transform to be applied to all image
                modalities (transformation are shared between image modalities, e.g., similar crop or rotation).
                Should end with ToTensorV2(). If used through the generic_data_module, should not include normalization.
                Not supported for multi-temporal data. The transform is not applied to non-image data, which is only
                converted to tensors if possible. If dict, can include multiple transforms per modality which are
                applied separately (no shared parameters between modalities).
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input data with this value.
                If None, does no replacement. Defaults to None.
            no_label_replace (float | None): Replace nan values in label with this value.
                If none, does no replacement. Defaults to -1.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Only works with image modalities. Is only applied to modalities with defined dataset_bands.
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            channel_position (int): Position of the channel dimension in the image modalities. Defaults to -3.
            concat_bands (bool): Concatenate all image modalities along the band dimension into a single "image", so
                that it can be processed by single-modal models. Concatenate in the order of provided modalities.
                Works with image modalities only. Does not work with allow_missing_modalities. Defaults to False.
        """

        super().__init__(
            data_root,
            label_data_root=label_data_root,
            image_grep=image_grep,
            label_grep=label_grep,
            split=split,
            image_modalities=image_modalities,
            rgb_modality=rgb_modality,
            rgb_indices=rgb_indices,
            allow_missing_modalities=allow_missing_modalities,
            allow_substring_file_names=allow_substring_file_names,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
            channel_position=channel_position,
            concat_bands=concat_bands,
            *args,
            **kwargs,
        )
        self.num_classes = num_classes
        self.class_names = class_names

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)

        if "mask" in item:
            item["mask"] = item["mask"].long()

        return item

    def plot(
        self, sample: dict[str, torch.Tensor], suptitle: str | None = None, show_axes: bool | None = False
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle
            show_axes: whether to show axes or not

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        image = sample["image"]
        if isinstance(image, dict):
            image = image[self.rgb_modality]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        image = image.take(self.rgb_indices, axis=0)
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)

        label_mask = sample["mask"]
        if isinstance(label_mask, torch.Tensor):
            label_mask = label_mask.numpy()

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"]
            if isinstance(prediction_mask, torch.Tensor):
                prediction_mask = prediction_mask.numpy()

        return self._plot_sample(
            image,
            label_mask,
            self.num_classes,
            prediction=prediction_mask if showing_predictions else None,
            suptitle=suptitle,
            class_names=self.class_names,
            show_axes=show_axes,
        )

    @staticmethod
    def _plot_sample(image, label, num_classes, prediction=None, suptitle=None, class_names=None, show_axes=False):
        num_images = 5 if prediction is not None else 4
        fig, ax = plt.subplots(1, num_images, figsize=(12, 10), layout="compressed")
        axes_visibility = "on" if show_axes else "off"

        # for legend
        ax[0].axis("off")

        norm = mpl.colors.Normalize(vmin=0, vmax=num_classes - 1)
        ax[1].axis(axes_visibility)
        ax[1].title.set_text("Image")
        ax[1].imshow(image)

        ax[2].axis(axes_visibility)
        ax[2].title.set_text("Ground Truth Mask")
        ax[2].imshow(label, cmap="jet", norm=norm)

        ax[3].axis(axes_visibility)
        ax[3].title.set_text("GT Mask on Image")
        ax[3].imshow(image)
        ax[3].imshow(label, cmap="jet", alpha=0.3, norm=norm)

        if prediction is not None:
            ax[4].axis(axes_visibility)
            ax[4].title.set_text("Predicted Mask")
            ax[4].imshow(prediction, cmap="jet", norm=norm)

        cmap = plt.get_cmap("jet")
        legend_data = []
        for i, _ in enumerate(range(num_classes)):
            class_name = class_names[i] if class_names else str(i)
            data = [i, cmap(norm(i)), class_name]
            legend_data.append(data)
        handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
        labels = [n for k, c, n in legend_data]
        ax[0].legend(handles, labels, loc="center")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class GenericMultimodalPixelwiseRegressionDataset(GenericMultimodalDataset):
    """GenericNonGeoPixelwiseRegressionDataset"""

    def __init__(
        self,
        data_root: Path,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        image_modalities: list[str] | None = None,
        rgb_modality: str | None = None,
        rgb_indices: list[int] | None = None,
        allow_missing_modalities: bool = False,
        allow_substring_file_names: bool = False,
        dataset_bands: dict[list] | None = None,
        output_bands: dict[list] | None = None,
        constant_scale: dict[str, float] = 1.0,
        transform: A.Compose | dict | None = None,
        no_data_replace: float | None = None,
        no_label_replace: float | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        channel_position: int = -3,
        concat_bands: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Constructor

        Args:
            data_root (dict[Path]): Dictionary of paths to data root directory or csv/parquet files with image-level
                data, with modalities as keys.
            label_data_root (Path): Path to data root directory with ground truth files. Set to None for predictions.
            image_grep (dict[str], optional): Dictionary with regular expression appended to data_root to find input
                images, with modalities as keys. Defaults to "*". Ignored when allow_substring_file_names is False.
            label_grep (str, optional): Regular expression appended to label_data_root to find ground truth files.
                Defaults to "*". Ignored when allow_substring_file_names is False.
            split (Path, optional): Path to file containing samples prefixes to be used for this split.
                The file can be a csv/parquet file with the prefixes in the index or a txt file with new-line separated
                sample prefixes. File names must be exact matches if allow_substring_file_names is False. Otherwise,
                files are searched using glob with the form Path(data_root).glob(prefix + [image or label grep]).
                If not specified, search samples based on files in data_root. Defaults to None.
            image_modalities(list[str], optional): List of pixel-level raster modalities. Defaults to data_root.keys().
                The difference between all modalities and image_modalities are non-image modalities which are treated
                differently during the transforms and are not modified but only converted into a tensor if possible.
            rgb_modality (str, optional): Modality used for RGB plots. Defaults to first modality in data_root.keys().
            rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
            allow_missing_modalities (bool, optional): Allow missing modalities during data loading. Defaults to False.
            allow_substring_file_names (bool, optional): Allow substrings during sample identification by adding
                image or label grep to the sample prefixes. If False, treats sample prefixes as full file names.
                If True and no split file is provided, considers the file stem as prefix, otherwise the full file name.
                Defaults to True.
            dataset_bands (dict[list], optional): Bands present in the dataset, provided in a dictionary with modalities
                as keys. This parameter names input channels (bands) using HLSBands, ints, int ranges, or strings, so
                that they can then be referred to by output_bands. Needs to be superset of output_bands. Can be a subset
                of all modalities. Defaults to None.
            output_bands (dict[list], optional): Bands that should be output by the dataset as named by dataset_bands,
                provided as a dictionary with modality keys. Can be subset of all modalities. Defaults to None.
            constant_scale (dict[float]): Factor to multiply data values by, provided as a dictionary with modalities as
                keys. Can be subset of all modalities. Defaults to None.
            transform (Albumentations.Compose | dict | None): Albumentations transform to be applied to all image
                modalities. Should end with ToTensorV2() and not include normalization. The transform is not applied to
                non-image data, which is only converted to tensors if possible. If dict, can include separate transforms
                per modality (no shared parameters between modalities).
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input data with this value.
                If None, does no replacement. Defaults to None.
            no_label_replace (float | None): Replace nan values in label with this value.
                If none, does no replacement. Defaults to None.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Only works with image modalities. Is only applied to modalities with defined dataset_bands.
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            channel_position (int): Position of the channel dimension in the image modalities. Defaults to -3.
            concat_bands (bool): Concatenate all image modalities along the band dimension into a single "image", so
                that it can be processed by single-modal models. Concatenate in the order of provided modalities.
                Works with image modalities only. Does not work with allow_missing_modalities. Defaults to False.
        """

        super().__init__(
            data_root,
            label_data_root=label_data_root,
            image_grep=image_grep,
            label_grep=label_grep,
            split=split,
            image_modalities=image_modalities,
            rgb_modality=rgb_modality,
            rgb_indices=rgb_indices,
            allow_missing_modalities=allow_missing_modalities,
            allow_substring_file_names=allow_substring_file_names,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
            channel_position=channel_position,
            concat_bands=concat_bands,
            *args,
            **kwargs,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)

        if "mask" in item:
            item["mask"] = item["mask"].float()

        return item

    def plot(
        self, sample: dict[str, torch.Tensor], suptitle: str | None = None, show_axes: bool | None = False
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample (dict[str, Tensor]): a sample returned by :meth:`__getitem__`
            suptitle (str|None): optional string to use as a suptitle
            show_axes (bool|None): whether to show axes or not

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """

        image = sample["image"]
        if isinstance(image, dict):
            image = image[self.rgb_modality]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        image = image.take(self.rgb_indices, axis=0)
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)

        label_mask = sample["mask"]
        if isinstance(label_mask, torch.Tensor):
            label_mask = label_mask.numpy()

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"]
            if isinstance(prediction_mask, torch.Tensor):
                prediction_mask = prediction_mask.numpy()

        return self._plot_sample(
            image,
            label_mask,
            prediction=prediction_mask if showing_predictions else None,
            suptitle=suptitle,
            show_axes=show_axes,
        )

    @staticmethod
    def _plot_sample(image, label, prediction=None, suptitle=None, show_axes=False):
        num_images = 4 if prediction is not None else 3
        fig, ax = plt.subplots(1, num_images, figsize=(12, 10), layout="compressed")
        axes_visibility = "on" if show_axes else "off"

        norm = mpl.colors.Normalize(vmin=label.min(), vmax=label.max())
        ax[0].axis(axes_visibility)
        ax[0].title.set_text("Image")
        ax[0].imshow(image)

        ax[1].axis(axes_visibility)
        ax[1].title.set_text("Ground Truth Mask")
        ax[1].imshow(label, cmap="Greens", norm=norm)

        ax[2].axis(axes_visibility)
        ax[2].title.set_text("GT Mask on Image")
        ax[2].imshow(image)
        ax[2].imshow(label, cmap="Greens", alpha=0.3, norm=norm)
        # ax[2].legend()

        if prediction is not None:
            ax[3].axis(axes_visibility)
            ax[3].title.set_text("Predicted Mask")
            ax[3].imshow(prediction, cmap="Greens", norm=norm)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class GenericMultimodalScalarDataset(GenericMultimodalDataset):
    """GenericMultimodalClassificationDataset"""

    def __init__(
        self,
        data_root: Path,
        num_classes: int,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        image_modalities: list[str] | None = None,
        rgb_modality: str | None = None,
        rgb_indices: list[int] | None = None,
        allow_missing_modalities: bool = False,
        allow_substring_file_names: bool = False,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        class_names: list[str] | None = None,
        constant_scale: dict[float] = 1.0,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        channel_position: int = -3,
        concat_bands: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Constructor

        Args:
            data_root (dict[Path]): Dictionary of paths to data root directory or csv/parquet files with image-level
                data, with modalities as keys.
            num_classes (int): Number of classes.
            label_data_root (Path, optional): Path to data root directory with labels or csv/parquet files with labels.
                Set to None for prediction mode.
            image_grep (dict[str], optional): Dictionary with regular expression appended to data_root to find input
                images, with modalities as keys. Defaults to "*". Ignored when allow_substring_file_names is False.
            label_grep (str, optional): Regular expression appended to label_data_root to find labels files.
                Defaults to "*". Ignored when allow_substring_file_names is False.
            split (Path, optional): Path to file containing samples prefixes to be used for this split.
                The file can be a csv/parquet file with the prefixes in the index or a txt file with new-line separated
                sample prefixes. File names must be exact matches if allow_substring_file_names is False. Otherwise,
                files are searched using glob with the form Path(data_root).glob(prefix + [image or label grep]).
                If not specified, search samples based on files in data_root. Defaults to None.
            image_modalities(list[str], optional): List of pixel-level raster modalities. Defaults to data_root.keys().
                The difference between all modalities and image_modalities are non-image modalities which are treated
                differently during the transforms and are not modified but only converted into a tensor if possible.
            rgb_modality (str, optional): Modality used for RGB plots. Defaults to first modality in data_root.keys().
            rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
            allow_missing_modalities (bool, optional): Allow missing modalities during data loading. Defaults to False.
            allow_substring_file_names (bool, optional): Allow substrings during sample identification by adding
                image or label grep to the sample prefixes. If False, treats sample prefixes as full file names.
                If True and no split file is provided, considers the file stem as prefix, otherwise the full file name.
                Defaults to True.
            dataset_bands (dict[list], optional): Bands present in the dataset, provided in a dictionary with modalities
                as keys. This parameter names input channels (bands) using HLSBands, ints, int ranges, or strings, so
                that they can then be referred to by output_bands. Needs to be superset of output_bands. Can be a subset
                of all modalities. Defaults to None.
            output_bands (dict[list], optional): Bands that should be output by the dataset as named by dataset_bands,
                provided as a dictionary with modality keys. Can be subset of all modalities. Defaults to None.
            class_names (list[str], optional): Names of the classes. Defaults to None.
            constant_scale (dict[float]): Factor to multiply data values by, provided as a dictionary with modalities as
                keys. Can be subset of all modalities. Defaults to None.
            transform (Albumentations.Compose | dict | None): Albumentations transform to be applied to all image
                modalities (transformation are shared between image modalities, e.g., similar crop or rotation).
                Should end with ToTensorV2(). If used through the generic_data_module, should not include normalization.
                Not supported for multi-temporal data. The transform is not applied to non-image data, which is only
                converted to tensors if possible. If dict, can include multiple transforms per modality which are
                applied separately (no shared parameters between modalities).
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input data with this value.
                If None, does no replacement. Defaults to None.
            no_label_replace (float | None): Replace nan values in label with this value.
                If none, does no replacement. Defaults to -1.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Only works with image modalities. Is only applied to modalities with defined dataset_bands.
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            channel_position (int): Position of the channel dimension in the image modalities. Defaults to -3.
            concat_bands (bool): Concatenate all image modalities along the band dimension into a single "image", so
                that it can be processed by single-modal models. Concatenate in the order of provided modalities.
                Works with image modalities only. Does not work with allow_missing_modalities. Defaults to False.
        """

        super().__init__(
            data_root,
            label_data_root=label_data_root,
            image_grep=image_grep,
            label_grep=label_grep,
            split=split,
            image_modalities=image_modalities,
            rgb_modality=rgb_modality,
            rgb_indices=rgb_indices,
            allow_missing_modalities=allow_missing_modalities,
            allow_substring_file_names=allow_substring_file_names,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
            channel_position=channel_position,
            scalar_label=True,
            concat_bands=concat_bands,
            *args,
            **kwargs,
        )

        self.num_classes = num_classes
        self.class_names = class_names

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        return item

    def plot(
        self, sample: dict[str, torch.Tensor], suptitle: str | None = None, show_axes: bool | None = False
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample (dict[str, Tensor]): a sample returned by :meth:`__getitem__`
            suptitle (str|None): optional string to use as a suptitle
            show_axes (bool|None): whether to show axes or not

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """

        # TODO: Check plotting code for classification tasks and add it to generic classification dataset as well
        raise NotImplementedError

        image = sample["image"]
        if isinstance(image, dict):
            image = image[self.rgb_modality]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        image = image.take(self.rgb_indices, axis=0)
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)

        label_mask = sample["mask"]
        if isinstance(label_mask, torch.Tensor):
            label_mask = label_mask.numpy()

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"]
            if isinstance(prediction_mask, torch.Tensor):
                prediction_mask = prediction_mask.numpy()

        return self._plot_sample(
            image,
            label_mask,
            prediction=prediction_mask if showing_predictions else None,
            suptitle=suptitle,
            show_axes=show_axes,
        )

    @staticmethod
    def _plot_sample(image, label, prediction=None, suptitle=None, show_axes=False):
        num_images = 4 if prediction is not None else 3
        fig, ax = plt.subplots(1, num_images, figsize=(12, 10), layout="compressed")
        axes_visibility = "on" if show_axes else "off"

        norm = mpl.colors.Normalize(vmin=label.min(), vmax=label.max())
        ax[0].axis(axes_visibility)
        ax[0].title.set_text("Image")
        ax[0].imshow(image)

        ax[1].axis(axes_visibility)
        ax[1].title.set_text("Ground Truth Mask")
        ax[1].imshow(label, cmap="Greens", norm=norm)

        ax[2].axis(axes_visibility)
        ax[2].title.set_text("GT Mask on Image")
        ax[2].imshow(image)
        ax[2].imshow(label, cmap="Greens", alpha=0.3, norm=norm)
        # ax[2].legend()

        if prediction is not None:
            ax[3].axis(axes_visibility)
            ax[3].title.set_text("Predicted Mask")
            ax[3].imshow(prediction, cmap="Greens", norm=norm)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
