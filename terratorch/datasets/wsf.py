# Copyright contributors to the Terratorch project


"""World Settlement Footprint datasets."""

import abc
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from rasterio.crs import CRS
from torchgeo.datasets import RasterDataset


class WSF(RasterDataset, abc.ABC):
    """Abstract base class for all World Settlement Footprint datasets."""

    filename_regex = r"""
        ^WSF.*v1_
        (?P<longitude>[-]?\d+)_
        (?P<latitude>[-]?\d+)
        \..*$
    """
    is_image = False

    @property
    @abc.abstractmethod
    def cmap(self) -> str:
        """Color map."""

    def init(
        self,
        paths: str | Iterable[str] = "data",
        crs: CRS | None = None,
        res: float | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            cmap: a valid Matplotlib colormap name

        Raises:
            FileNotFoundError: if no files are found in ``paths``
        """

        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.
        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
        Returns:
            a matplotlib Figure with the rendered sample
        """

        mask = sample["mask"].squeeze().numpy()
        mask = np.where(mask == 0, np.nan, mask)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.imshow(mask, self.cmap)
        ax.axis("off")

        if show_titles:
            ax.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class WSF2019(WSF):
    """World Settlement Footprint 2019 dataset.

    The `World Settlement Footprint 2019 dataset
    <https://download.geoservice.dlr.de/WSF2019/>`__, created and hosted by
    `DLR <https://dlr.de/>`_, provides a raster, geo-referenced, single-layer
    global built settlement land cover map. The data was created with both
    Sentinel 1 and 2 satellite imagery.
    """

    filename_glob = "WSF2019_v1_*.tif"
    cmap = "binary_r"


class WSFEvolution(WSF):
    """World Settlement Footprint Evolution dataset.

    The `World Settlement Footprint Evolution Version 1 dataset
    <https://download.geoservice.dlr.de/WSF_EVO/>`__, created and hosted by
    `DLR <https://dlr.de/>`_, provides a raster, geo-referenced, single-layer
    global built settlement land cover map. The WSF Evolution mask defines one
    value per pixel corresponding to the year between 1985 and 2015 when built
    settlements were identified, or zero otherwise. The data was created with a
    mix of Landsat and Sentinel satellite imagery.
    """

    filename_glob = "WSFevolution_v1_*.tif"
    cmap = "RdYlGn"
