# Copyright contributors to the Terratorch project

import importlib
import logging
import re
from collections.abc import Callable, Iterable

import albumentations as A
import numpy as np
import torch

logger = logging.getLogger("terratorch")


def instantiate_transform_from_dict(transform_dict):
    """
    Convert a YAML transform configuration dict into an actual transform object.

    Args:
        transform_dict: Dict with "class_path" and "init_args" keys
    Returns:
        Instantiated transform object
    Example:
        >>> config = {
        ...     "class_path": "albumentations.pytorch.ToTensorV2",
        ...     "init_args": {"always_apply": True, "p": 1.0}
        ... }
        >>> transform = instantiate_transform_from_dict(config)
    """
    if not isinstance(transform_dict, dict) or "class_path" not in transform_dict:
        # Already an instantiated object or not a config dict
        return transform_dict

    class_path = transform_dict["class_path"]
    init_args = transform_dict.get("init_args", {})
    if "always_apply" in init_args:
        # Always apply is a special case that should not be passed to the transform
        init_args.pop("always_apply")

    # Handle both full paths (albumentations.pytorch.ToTensorV2) and short names (ToTensorV2)
    if "." in class_path:
        # Full path: split into module and class name
        module_path, class_name = class_path.rsplit(".", 1)
    else:
        # Short name: try common albumentations locations
        class_name = class_path
        # Try albumentations.pytorch first (most common for ToTensorV2)
        for module_path in ["albumentations.pytorch", "albumentations", "albumentations.augmentations"]:
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, class_name):
                    break
            except (ImportError, AttributeError):
                continue
        else:
            raise ImportError(f"Could not find transform class '{class_name}' in common albumentations modules")

    # Import the module and get the class
    module = importlib.import_module(module_path)
    transform_class = getattr(module, class_name)

    # Instantiate with init_args
    return transform_class(**init_args)


def wrap_in_compose_is_list(transform_list):
    """
    Wrap a list of transforms in an Albumentations Compose object.

    Handles both:
    - Already instantiated transform objects
    - YAML config dicts with "class_path" and "init_args"
    - Already composed transforms (returns as-is)
    - Mixed lists containing both instantiated transforms and dict configs

    Args:
        transform_list: List of transforms, config dicts, or a single transform/Compose object

    Returns:
        Albumentations Compose object or the original transform if not a list
    """
    # If it"s already a Compose object (has available_keys attribute), return as-is
    if hasattr(transform_list, "available_keys"):
        return transform_list

    # If it's a list or tuple, process each item
    if isinstance(transform_list, (list, tuple)):
        # Convert any dict configs to actual transform objects
        instantiated_transforms = []
        for transform in transform_list:
            if isinstance(transform, dict):
                # This could be a YAML config dict or a plain dict
                if "class_path" in transform:
                    # YAML config dict with class_path, instantiate it
                    instantiated_transforms.append(instantiate_transform_from_dict(transform))
                else:
                    # Plain dict without class_path - this is unexpected and likely an error
                    # We require explicit class_path for safety and clarity
                    msg = f"Unexpected dict in transform list without 'class_path': {transform}"
                    raise ValueError(msg)
            elif isinstance(transform, A.Compose):
                # Already a Compose object, unwrap it
                # This handles cases where a Compose is nested in a list
                instantiated_transforms.extend(transform.transforms)
            else:
                # Already an instantiated transform object (BasicTransform subclass)
                instantiated_transforms.append(transform)

        # Wrap in Compose with check_shapes=False for multitemporal case
        return A.Compose(instantiated_transforms, is_check_shapes=False)

    # Single transform object (not a list), return as-is
    return transform_list


def check_dataset_stackability(dataset, batch_size: int, max_checks: int | None = 100) -> bool:
    if max_checks is None or len(dataset) <= max_checks:
        random_indexes = np.arange(len(dataset))
    else:
        random_indexes = np.random.randint(low=0, high=len(dataset), size=max_checks)
    shapes = np.array([dataset[idx]["image"].shape for idx in random_indexes])

    if len(shapes) > 1:
        if np.array_equal(np.max(shapes, 0), np.min(shapes, 0)):
            return batch_size
        else:
            logger.warning(
                "The batch samples can't be stacked, since they don't have the same dimensions. Setting batch_size=1."
            )
            return 1
    else:
        return batch_size


def check_dataset_stackability_dict(dataset, batch_size: int, max_checks: int | None = 100) -> bool:
    """Check stackability with item['image'] being a dict."""
    if max_checks is None or len(dataset) <= max_checks:
        random_indexes = np.arange(len(dataset))
    else:
        random_indexes = np.random.randint(low=0, high=len(dataset), size=max_checks)

    shapes = {}
    for idx in random_indexes:
        for mod, value in dataset[idx]["image"].items():
            if mod in shapes:
                shapes[mod].append(value.shape)
            else:
                shapes[mod] = [value.shape]

    if all(np.array_equal(np.max(s, 0), np.min(s, 0)) for s in shapes.values()):
        return batch_size
    else:
        logger.warning(
            "The batch samples can't be stacked, since they don't have the same dimensions. Setting batch_size=1."
        )
        return 1


class Normalize(Callable):
    """
    Unified normalization class for both regular and temporal images.

    Handles normalization for images with shapes:
    - (B, C, H, W): Regular 4D images
    - (B, C, T, H, W): Temporal 5D images

    Means and stds can be:
    - Shape (C,): For regular images or to average over temporal dimension
    - Shape (C, T): For temporal statistics applied to 5D images

    Args:
        means: Mean values. Can be list, numpy array, or torch tensor.
               Shape (C,) or (C, T).
        stds: Standard deviation values. Same format as means.
        denormalize: If True, reverses normalization (image * stds + means).
                    If False, applies normalization ((image - means) / stds).
                    Defaults to False.

    Examples:
        >>> # Regular 4D image
        >>> means = [123.5, 128.2, 129.1]
        >>> stds = [50.0, 51.2, 52.3]
        >>> norm = Normalize(means, stds)
        >>> batch = {"image": torch.randn(2, 3, 256, 256)}  # (B,C,H,W)
        >>> normalized = norm(batch)

        >>> # Temporal 5D image
        >>> means = [[100, 101], [200, 201], [300, 301]]  # (C,T) = (3,2)
        >>> stds = [[10, 11], [20, 21], [30, 31]]
        >>> norm = Normalize(means, stds)
        >>> batch = {"image": torch.randn(2, 3, 2, 256, 256)}  # (B,C,T,H,W)
        >>> normalized = norm(batch)

        >>> # Denormalization
        >>> norm_denorm = Normalize(means, stds, denormalize=True)
        >>> restored = norm_denorm({"image": normalized["image"]})
    """

    def __init__(self, means, stds, denormalize: bool = False):
        super().__init__()

        # Convert to torch tensors for consistent handling
        self.means = torch.tensor(means) if not isinstance(means, torch.Tensor) else means.clone()
        self.stds = torch.tensor(stds) if not isinstance(stds, torch.Tensor) else stds.clone()
        self.denormalize = denormalize

    def __call__(self, batch):
        """
        Apply normalization to batch images.

        Args:
            batch: Dictionary with "image" key containing tensor to normalize.

        Returns:
            Dictionary with normalized "image" tensor.
        """
        image = batch["image"]
        device = image.device

        means_tensor = self.means.to(device)
        stds_tensor = self.stds.to(device)

        if len(image.shape) == 5:
            # Image shape: (B, C, T, H, W)
            if len(self.means.shape) == 2:
                # Means shape: (C, T) - use full temporal statistics
                # Reshape to (1, C, T, 1, 1) for broadcasting
                c, t = self.means.shape
                means = means_tensor.view(1, c, t, 1, 1)
                stds = stds_tensor.view(1, c, t, 1, 1)
            else:
                # Means shape: (C,) - replicate across temporal dimension
                # Reshape to (1, C, 1, 1, 1) for broadcasting
                means = means_tensor.view(1, -1, 1, 1, 1)
                stds = stds_tensor.view(1, -1, 1, 1, 1)

        elif len(image.shape) == 4:
            # Image shape: (B, C, H, W)
            if len(self.means.shape) == 2:
                # Means shape: (C, T) - average over temporal dimension
                # Reshape to (1, C, 1, 1) for broadcasting
                means = means_tensor.mean(dim=1).view(1, -1, 1, 1)
                stds = stds_tensor.mean(dim=1).view(1, -1, 1, 1)
            else:
                # Means shape: (C,)
                # Reshape to (1, C, 1, 1) for broadcasting
                means = means_tensor.view(1, -1, 1, 1)
                stds = stds_tensor.view(1, -1, 1, 1)

        else:
            msg = f"Expected image with 4 or 5 dimensions, got {len(image.shape)}"
            raise ValueError(msg)

        # Apply normalization or denormalization
        if self.denormalize:
            batch["image"] = image * stds + means
        else:
            batch["image"] = (image - means) / stds

        return batch
