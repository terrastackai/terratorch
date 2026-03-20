import random
import warnings
from pathlib import Path

import albumentations as A  # noqa: N812
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import pad_numpy, to_tensor

CAT_TILES = ["31TBF", "31TCF", "31TCG", "31TDF", "31TDG"]
FR_TILES = ["31TCJ", "31TDK", "31TCL", "31TDM", "31UCP", "31UDR"]

MAX_TEMPORAL_IMAGE_SIZE = (366, 366)

SELECTED_CLASSES = [
    110,  # 'Wheat'
    120,  # 'Maize'
    140,  # 'Sorghum'
    150,  # 'Barley'
    160,  # 'Rye'
    170,  # 'Oats'
    330,  # 'Grapes'
    435,  # 'Rapeseed'
    438,  # 'Sunflower'
    510,  # 'Potatoes'
    770,  # 'Peas'
]


class Sen4AgriNet(NonGeoDataset):
    def __init__(
        self,
        data_root: str,
        bands: list[str] | None = None,
        scenario: str = "random",
        split: str = "train",
        transform: A.Compose = None,
        truncate_image: int | None = 4,
        pad_image: int | None = 4,
        spatial_interpolate_and_stack_temporally: bool = True,  # noqa: FBT001, FBT002
        seed: int = 42,
    ):
        """
        Pytorch Dataset class to load samples from the [Sen4AgriNet](https://github.com/Orion-AI-Lab/S4A) dataset, supporting
        multiple scenarios for splitting the data.

        Args:
            data_root (str): Root directory of the dataset.
            bands (list of str, optional): List of band names to load. Defaults to all available bands.
            scenario (str): Defines the splitting scenario to use. Options are:
                - 'random': Random split of the data.
                - 'spatial': Split by geographical regions (Catalonia and France).
                - 'spatio-temporal': Split by region and year (France 2019 and Catalonia 2020).
            split (str): Specifies the dataset split. Options are 'train', 'val', or 'test'.
            transform (albumentations.Compose, optional): Albumentations transformations to apply to the data.
            truncate_image (int, optional): Number of timesteps to truncate the time dimension of the image.
                If None, no truncation is applied. Default is 4.
            pad_image (int, optional): Number of timesteps to pad the time dimension of the image.
                If None, no padding is applied. Default is 4.
            spatial_interpolate_and_stack_temporally (bool): Whether to interpolate bands and concatenate them over time
            seed (int): Random seed used for data splitting.
        """
        self.data_root = Path(data_root) / "data"
        self.transform = transform if transform else lambda **batch: to_tensor(batch)
        self.scenario = scenario
        self.seed = seed
        self.truncate_image = truncate_image
        self.pad_image = pad_image
        self.spatial_interpolate_and_stack_temporally = spatial_interpolate_and_stack_temporally

        if bands is None:
            bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
        self.bands = bands

        self.image_files = list(self.data_root.glob("**/*.nc"))

        self.train_files, self.val_files, self.test_files = self.split_data()

        if split == "train":
            self.image_files = self.train_files
        elif split == "val":
            self.image_files = self.val_files
        elif split == "test":
            self.image_files = self.test_files

    def __len__(self):
        return len(self.image_files)

    def split_data(self):
        random.seed(self.seed)

        if self.scenario == "random":
            random.shuffle(self.image_files)
            total_files = len(self.image_files)
            train_split = int(0.6 * total_files)
            val_split = int(0.8 * total_files)

            train_files = self.image_files[:train_split]
            val_files = self.image_files[train_split:val_split]
            test_files = self.image_files[val_split:]

        elif self.scenario == "spatial":
            catalonia_files = [f for f in self.image_files if any(tile in f.stem for tile in CAT_TILES)]
            france_files = [f for f in self.image_files if any(tile in f.stem for tile in FR_TILES)]

            val_split_cat = int(0.2 * len(catalonia_files))
            train_files = catalonia_files[val_split_cat:]
            val_files = catalonia_files[:val_split_cat]
            test_files = france_files

        elif self.scenario == "spatio-temporal":
            france_files = [f for f in self.image_files if any(tile in f.stem for tile in FR_TILES)]
            catalonia_files = [f for f in self.image_files if any(tile in f.stem for tile in CAT_TILES)]

            france_2019_files = [f for f in france_files if "2019" in f.stem]
            catalonia_2020_files = [f for f in catalonia_files if "2020" in f.stem]

            val_split_france_2019 = int(0.2 * len(france_2019_files))
            train_files = france_2019_files[val_split_france_2019:]
            val_files = france_2019_files[:val_split_france_2019]
            test_files = catalonia_2020_files

        return train_files, val_files, test_files

    def __getitem__(self, index: int):
        patch_file = self.image_files[index]

        with h5py.File(patch_file, "r") as patch_data:
            output = {}
            images_over_time = []
            for band in self.bands:
                band_group = patch_data[band]
                band_data = band_group[f"{band}"][:]
                time_vector = band_group["time"][:]

                sorted_indices = np.argsort(time_vector)
                band_data = band_data[sorted_indices].astype(np.float32)

                if self.truncate_image:
                    band_data = band_data[-self.truncate_image :]
                if self.pad_image:
                    band_data = pad_numpy(band_data, self.pad_image)

                if self.spatial_interpolate_and_stack_temporally:
                    band_data = torch.from_numpy(band_data)
                    band_data = band_data.clone().detach()

                    interpolated = F.interpolate(
                        band_data.unsqueeze(0), size=MAX_TEMPORAL_IMAGE_SIZE, mode="bilinear", align_corners=False
                    ).squeeze(0)
                    images_over_time.append(interpolated)
                else:
                    output[band] = band_data

            if self.spatial_interpolate_and_stack_temporally:
                images = torch.stack(images_over_time, dim=0).numpy()
                output["image"] = images

            labels = patch_data["labels"]["labels"][:].astype(int)
            parcels = patch_data["parcels"]["parcels"][:].astype(int)

        output["mask"] = labels

        image_shape = output["image"].shape[-2:]
        mask_shape = output["mask"].shape

        if image_shape != mask_shape:
            diff_h = mask_shape[0] - image_shape[0]
            diff_w = mask_shape[1] - image_shape[1]

            output["image"] = np.pad(
                output["image"],
                [(0, 0), (0, 0), (diff_h // 2, diff_h - diff_h // 2), (diff_w // 2, diff_w - diff_w // 2)],
                mode="constant",
                constant_values=0,
            )

        linear_encoder = {val: i + 1 for i, val in enumerate(sorted(SELECTED_CLASSES))}
        linear_encoder[0] = 0

        output["image"] = output["image"].transpose(0, 2, 3, 1)
        output["mask"] = self.map_mask_to_discrete_classes(output["mask"], linear_encoder)

        if self.transform:
            output = self.transform(**output)

        output["parcels"] = parcels

        return output

    def plot(self, sample, suptitle=None, show_axes=False):
        rgb_bands = ["B04", "B03", "B02"]

        if not all(band in sample for band in rgb_bands):
            warnings.warn("No RGB image.")  # noqa: B028
            return None

        rgb_images = []
        for t in range(sample["B04"].shape[0]):
            rgb_image = torch.stack([sample[band][t] for band in rgb_bands])

            # Normalization
            rgb_min = rgb_image.min(dim=1, keepdim=True).values.min(dim=2, keepdim=True).values
            rgb_max = rgb_image.max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values
            denom = rgb_max - rgb_min
            denom[denom == 0] = 1
            rgb_image = (rgb_image - rgb_min) / denom

            rgb_image = rgb_image.permute(1, 2, 0).numpy()
            rgb_images.append(np.clip(rgb_image, 0, 1))

        dates = torch.arange(sample["B04"].shape[0])

        return self._plot_sample(rgb_images, dates, sample.get("labels"), suptitle=suptitle, show_axes=show_axes)

    def _plot_sample(self, images, dates, labels=None, suptitle=None, show_axes=False):
        num_images = len(images)
        cols = 5
        rows = (num_images + cols - 1) // cols

        fig, ax = plt.subplots(rows, cols, figsize=(20, 4 * rows))
        axes_visibility = "on" if show_axes else "off"

        for i, image in enumerate(images):
            ax[i // cols, i % cols].imshow(image)
            ax[i // cols, i % cols].set_title(f"T{i + 1} - Day {dates[i].item()}")
            ax[i // cols, i % cols].axis(axes_visibility)

        if labels is not None:
            if rows * cols > num_images:
                target_ax = ax[(num_images) // cols, (num_images) % cols]
            else:
                fig.add_subplot(rows + 1, 1, 1)
                target_ax = fig.gca()

            target_ax.imshow(labels.numpy(), cmap="tab20")
            target_ax.set_title("Labels")
            target_ax.axis(axes_visibility)

        for k in range(num_images, rows * cols):
            ax[k // cols, k % cols].axis(axes_visibility)

        if suptitle:
            plt.suptitle(suptitle)

        plt.tight_layout()
        plt.show()

    def map_mask_to_discrete_classes(self, mask, encoder):
        map_func = np.vectorize(lambda x: encoder.get(x, 0))
        return map_func(mask)
