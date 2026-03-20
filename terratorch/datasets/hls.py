# Copyright contributors to the Terratorch project


"""Harmonized Landsat and Sentinel-2 datasets."""

import abc
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional, Union

from rasterio.crs import CRS
from torchgeo.datasets import Landsat


class HLS(Landsat, abc.ABC):
    """Abstract base class for all HLS datasets.

    The `Harmonized Landsat and Sentinel-2 (HLS)
     <https://hls.gsfc.nasa.gov/>`_ project is a NASA initiative
    aiming to produce a seamless surface reflectance record from the
    Operational Land Imager (OLI) and Multi-Spectral Instrument (MSI) aboard
    Landsat-8/9 and Sentinel-2A/B remote sensing satellites, respectively.
    The HLS products are hosted at `LP DAAC
    <https://hls.gsfc.nasa.gov/hls-data/>`_, and created from a set of
    algorithms: atmospheric correction, cloud and cloud-shadow masking,
    geographic co-registration and common gridding, bidirectional reflectance
    distribution function (BRDF) normalization, and bandpass adjustment. The
    HLS Version-2.0 data are available now with global coverage (except for
    Antarctica). With four sensors currently in this virtual constellation,
    HLS provides observations once every three days at the equator and more
    frequently with increasing latitude. The HLS project is a collaboration
    between NASA and USGS, with the science team at NASA Goddard Space Flight
    Center supported by USGS Earth Resources Observation and Science (EROS)
    Center on atmospheric correction, the production team at NASA Marshall
    Space Flight Center, and product archive at USGS.
    """

    filename_regex = r"""
        ^HLS
        \.(?P<sensor>[LS])30
        \.T(?P<tile>\d{2}[A-Z]{3})
        \.(?P<date>\d{7})
        T(?P<time>\d{6})
        \.v2\.0
        \.(?P<band>B\d[0-9A])
        .*$
    """
    date_format = "%Y%j"
    is_image = True
    separate_files = True

    def init(
        self,
        paths: str | Iterable[str] = "data",
        crs: CRS | None = None,
        res: float | None = None,
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new dataset instance.
        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            cmap: a valid Matplotlib colormap name
        Raises:
            FileNotFoundError: if no files are found in ``root``
        """

        super().__init__(paths, crs, res, bands, transforms, cache)


class HLSS30(HLS):
    """HLS Sentinel S30."""

    filename_glob = "HLS.S30.*.v2.0.B??.tif"

    default_bands = ["B02", "B03", "B04", "B8A", "B11", "B12"]
    rgb_bands = ["B04", "B03", "B02"]


class HLSL30(HLS):
    """HLS Landsat L30."""

    filename_glob = "HLS.L30.*.v2.0.B??.tif"

    default_bands = ["B02", "B03", "B04", "B05", "B06", "B07"]
    rgb_bands = ["B04", "B03", "B02"]
