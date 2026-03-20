# Copyright contributors to the Terratorch project

"""Module containing generic dataset classes"""

import glob
import os
import warnings
from abc import ABC
from pathlib import Path
from typing import Any, Tuple

import albumentations as A  # noqa: N812
import numpy as np
import PIL
import rioxarray
import tifffile
import torch
import xarray as xr
from einops import rearrange
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import rasterio_loader
from torchvision.datasets import ImageFolder

from terratorch.datasets.utils import HLSBands, default_transform, filter_valid_files, generate_bands_intervals


class GenericScalarLabelDataset(NonGeoDataset, ImageFolder, ABC):
    """
    This is a generic dataset class to be used for instantiating datasets from arguments.
    Ideally, one would create a dataset class specific to a dataset.
    """

    def __init__(
        self,
        data_root: Path,
        split: Path | None = None,
        require_label: bool = True,
        ignore_split_file_extensions: bool = True,  # noqa: FBT001, FBT002
        allow_substring_split_file: bool = True,  # noqa: FBT001, FBT002
        rgb_indices: list[int] | None = None,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float = 0,
        expand_temporal_dimension: bool = False,  # noqa: FBT001, FBT002
        temporal_channel_major: bool = False,
    ) -> None:
        """Constructor

        Args:
            data_root (Path): Path to data root directory
            split (Path, optional): Path to file containing files to be used for this split.
                The file should be a new-line separated prefixes contained in the desired files.
                Files will be seached using glob with the form Path(data_root).glob(prefix + [image or label grep])
            ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
                file to determine which files to include in the dataset.
                E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
                actually ".jpg". Defaults to True.
            allow_substring_split_file (bool, optional): Whether the split files contain substrings
                that must be present in file names to be included (as in mmsegmentation), or exact
                matches (e.g. eurosat). Defaults to True.
            rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
            dataset_bands (list[HLSBands | int | tuple[int, int] | str] | None): Bands present in the dataset. This
                parameter gives identifiers to input channels (bands) so that they can then be refered to by
                output_bands. Can use the HLSBands enum, ints, int ranges, or strings. Defaults to None.
            output_bands (list[HLSBands | int | tuple[int, int] | str] | None): Bands that should be output by the
                dataset as named by dataset_bands.
            constant_scale (float): Factor to multiply image values by. Defaults to 1.
            transform (Albumentations.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float): Replace nan values in input images with this value. Defaults to 0.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to False. Assumes bands are grouped by time (all bands of one timestep are stacked together), otherwise set temporal_channel_major=True.
            temporal_channel_major: Used for expand_temporal_dimension, set True if bands are grouped by channel (all timesteps of one band are stacked together).
        """
        self.split_file = split
        self.split = split
        self.require_label = require_label
        self.image_files = sorted(glob.glob(os.path.join(data_root, "**"), recursive=True))
        self.image_files = [f for f in self.image_files if not os.path.isdir(f)]
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.expand_temporal_dimension = expand_temporal_dimension
        if self.expand_temporal_dimension:
            if not self.temporal_channel_major:
                warnings.warn(
                    "expand_temporal_dimension=True assumes bands are grouped by time "
                    "(all bands of one timestep are stacked together). "
                    "If instead bands are grouped by channel "
                    "(all timesteps of one band are stacked together), "
                    "set temporal_channel_major=True."
                )

            if dataset_bands is None:
                raise ValueError("Please provide dataset_bands when expand_temporal_dimension=True.")
        if self.split_file is not None:
            with open(self.split_file) as f:
                split = f.readlines()
            valid_files = {rf"{substring.strip()}" for substring in split}
            self.image_files = filter_valid_files(
                self.image_files,
                valid_files=valid_files,
                ignore_extensions=ignore_split_file_extensions,
                allow_substring=allow_substring_split_file,
            )

            def is_valid_file(x):
                return x in self.image_files

        else:

            def is_valid_file(x):
                return True

        # Checking if it's necessary to overwrite the `find_classes` method
        try:
            _, _ = self.find_classes(data_root)
        except FileNotFoundError:
            self.find_classes = self.find_classes_

        super().__init__(
            root=data_root, transform=None, target_transform=None, loader=rasterio_loader, is_valid_file=is_valid_file
        )

        self.rgb_indices = [0, 1, 2] if rgb_indices is None else rgb_indices

        self.dataset_bands = generate_bands_intervals(dataset_bands)
        self.output_bands = generate_bands_intervals(output_bands)

        if self.output_bands and not self.dataset_bands:
            msg = "If output bands provided, dataset_bands must also be provided"
            return Exception(msg)  # noqa: PLE0101

        # There is a special condition if the bands are defined as simple strings.
        if self.output_bands:
            if len(set(self.output_bands) & set(self.dataset_bands)) != len(self.output_bands):
                msg = "Output bands must be a subset of dataset bands"
                raise Exception(msg)

            self.filter_indices = [self.dataset_bands.index(band) for band in self.output_bands]

        else:
            self.filter_indices = None
        # If no transform is given, apply only to transform to torch tensor
        self.transforms = transform if transform else default_transform
        # self.transform = transform if transform else ToTensorV2()

        import warnings

        import rasterio

        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    def __len__(self) -> int:
        return len(self.image_files)

    def _loader(self, path: str | Path) -> Image.Image:
        try:
            with open(path, "rb") as f:
                img = np.asarray(Image.open(f))
        except PIL.UnidentifiedImageError:
            # TIFF files containing floating-point values should be handled in
            # another way.
            if path.endswith(".tif") or path.endswith(".tiff"):
                img = tifffile.imread(path)
            else:
                raise OSError(f"Could not open {path}. Unsupported format or configuration.")
        return img

    def __base_getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self._loader(path)

        return sample, target

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, label = self.__base_getitem__(index)

        if self.expand_temporal_dimension:
            if self.temporal_channel_major:
                image = rearrange(image, "h w (channels time) -> time h w channels", channels=len(self.dataset_bands))
            else:
                image = rearrange(image, "h w (time channels) -> time h w channels", channels=len(self.dataset_bands))
        if self.filter_indices:
            image = image[..., self.filter_indices]

        image = image.astype(np.float32) * self.constant_scale

        if self.transforms:
            image = self.transforms(image=image)["image"]  # albumentations returns dict

        if self.require_label:
            output = {
                "image": image,
                "label": label,  # samples is an attribute of ImageFolder. Contains a tuple of (Path, Target)
                "filename": self.image_files[index],
            }
        else:
            output = {
                "image": image,
                "filename": self.image_files[index],
            }

        return output

    def _generate_bands_intervals(self, bands_intervals: list[int | str | HLSBands | tuple[int]] | None = None):
        if bands_intervals is None:
            return None
        bands = []
        for element in bands_intervals:
            # if its an interval
            if isinstance(element, tuple):
                if len(element) != 2:  # noqa: PLR2004
                    msg = "When defining an interval, a tuple of two integers should be passed,\
                    defining start and end indices inclusive"
                    raise Exception(msg)
                expanded_element = list(range(element[0], element[1] + 1))
                bands.extend(expanded_element)
            else:
                bands.append(element)
        return bands

    def _load_file(self, path) -> xr.DataArray:
        data = rioxarray.open_rasterio(path, masked=True)
        data = data.fillna(self.no_data_replace)
        return data

    def find_classes_(self, directory: str | Path) -> tuple[list[str], dict[str, int]]:
        return ["./"], {"./": 0}


class GenericNonGeoClassificationDataset(GenericScalarLabelDataset):
    """GenericNonGeoClassificationDataset"""

    def __init__(
        self,
        data_root: Path,
        num_classes: int,
        split: Path | None = None,
        require_label: bool = True,
        ignore_split_file_extensions: bool = True,  # noqa: FBT001, FBT002
        allow_substring_split_file: bool = True,  # noqa: FBT001, FBT002
        rgb_indices: list[str] | None = None,
        dataset_bands: list[HLSBands | int] | None = None,
        output_bands: list[HLSBands | int] | None = None,
        class_names: list[str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float = 0,
        expand_temporal_dimension: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """A generic Non-Geo dataset for classification.

        Args:
            data_root (Path): Path to data root directory
            num_classes (int): Number of classes in the dataset
            split (Path, optional): Path to file containing files to be used for this split.
                The file should be a new-line separated prefixes contained in the desired files.
                Files will be seached using glob with the form Path(data_root).glob(prefix + [image or label grep])
            ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
                file to determine which files to include in the dataset.
                E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
                actually ".jpg". Defaults to True.
            allow_substring_split_file (bool, optional): Whether the split files contain substrings
                that must be present in file names to be included (as in mmsegmentation), or exact
                matches (e.g. eurosat). Defaults to True.
            rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
            dataset_bands (list[HLSBands | int] | None): Bands present in the dataset.
            output_bands (list[HLSBands | int] | None): Bands that should be output by the dataset.
            class_names (list[str], optional): Class names. Defaults to None.
            constant_scale (float): Factor to multiply image values by. Defaults to 1.
            transform (Albumentations.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float): Replace nan values in input images with this value. Defaults to 0.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to False.
        """
        super().__init__(
            data_root,
            split=split,
            require_label=require_label,
            ignore_split_file_extensions=ignore_split_file_extensions,
            allow_substring_split_file=allow_substring_split_file,
            rgb_indices=rgb_indices,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            expand_temporal_dimension=expand_temporal_dimension,
        )
        self.num_classes = num_classes
        self.class_names = class_names

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        if "label" in item:
            item["label"] = torch.tensor(item["label"]).long()
        return item

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        pass
