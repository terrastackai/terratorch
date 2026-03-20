import importlib
from typing import Any, Dict

from torch import nn


def get_module_and_class(path: str) -> (str, str):
    """Splits a dotted path string into module and class name."""
    class_name = path.rsplit(".", maxsplit=1)[-1]
    module_name = path.replace("." + class_name, "")
    return module_name, class_name


def _instantiate_from_path(dotted_path: str, **kwargs: Any) -> Any:
    """
    Instantiates a class from a dotted path string (e.g., "my_module.MyClass").

    Args:
        dotted_path: The complete dotted path to the class.
        **kwargs: Keyword arguments to pass to the class constructor.

    Returns:
        An instance of the specified class.
    """
    module_name, class_name = get_module_and_class(dotted_path)

    # Import the module
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}' from path '{dotted_path}'") from e

    # Get the class from the module
    try:
        class_ = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"Module '{module_name}' has no class named '{class_name}'") from e

    # Instantiate and return the class
    return class_(**kwargs)


def bounds_from_transform(transform, width: int, height: int) -> tuple[float, float, float, float]:
    """
    Compute spatial bounds (left, bottom, right, top) from an affine transform
    and raster size.

    Assumes pixel-edge coordinates with corners at (0, 0) and (width, height).
    """
    corners = [(0, 0), (width, 0), (0, height), (width, height)]
    xs, ys = zip(*(transform * xy for xy in corners))

    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)

    return left, bottom, right, top


def infer_transform_and_bounds(
    center_y: float,
    center_x: float,
    height: int,
    width: int,
    pixelsize: float,
) -> tuple[float, float, float, float]:
    """
    Infer spatial bounds (left, bottom, right, top) for an image given its center
    coordinate and pixel size heuristically.
    If (center_y, center_x) looks like lat/lon (y in [-90, 90], x in [-180, 180]),
    bounds are computed geodesically on WGS84.
    Otherwise, bounds are computed with a simple planar offset.
    """

    px = float(pixelsize)
    half_w = (width * px) / 2.0
    half_h = (height * px) / 2.0

    if -90.0 <= center_y <= 90.0 and -180.0 <= center_x <= 180.0:
        from pyproj import Geod

        geod = Geod(ellps="WGS84")

        lon0, lat0 = float(center_x), float(center_y)

        lon_w, lat_w, _ = geod.fwd(lon0, lat0, 270, half_w)
        lon_e, lat_e, _ = geod.fwd(lon0, lat0, 90, half_w)

        lon_s, lat_s, _ = geod.fwd(lon0, lat0, 180, half_h)
        lon_n, lat_n, _ = geod.fwd(lon0, lat0, 0, half_h)

        left = min(lon_w, lon_e)
        right = max(lon_w, lon_e)
        bottom = min(lat_s, lat_n)
        top = max(lat_s, lat_n)

        return left, bottom, right, top

    left = center_x - half_w
    right = center_x + half_w
    bottom = center_y - half_h
    top = center_y + half_h

    return left, bottom, right, top
