"""
LightningDataModule to load Earth Observation data from GeoTIFF files using
rasterio.
"""

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Literal

import lightning as L
import numpy as np
import torch

# import torchdata
import yaml
from box import Box
from einops import rearrange
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torchvision.transforms import v2


class EODataset(Dataset):
    """Reads different Earth Observation data sources from a directory."""

    def __init__(self, chips_path: list[Path], size: int, platforms: list, metadata: Box) -> None:
        super().__init__()
        self.chips_path = chips_path
        self.size = size
        self.transforms = {}

        # Generate transforms for each platform using a helper function
        for platform in platforms:
            mean = list(metadata[platform].bands.mean.values())
            std = list(metadata[platform].bands.std.values())
            self.transforms[platform] = self.create_transforms(mean, std)

    def create_transforms(self, mean, std):
        return v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                # v2.RandomCrop(size=(self.size, self.size)),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self):
        return len(self.chips_path)

    def __getitem__(self, idx):
        chip_path = self.chips_path[idx]
        with np.load(chip_path, allow_pickle=False) as chip:
            platform = chip_path.parent.name
            if platform == "sentinel-1-rtc":
                pixels = chip["pixels"].astype(np.float32)
                pixels[pixels <= 0] = 1e-10  # replace corrupted pixels in sentinel-1-rtc with small value
                pixels = 10 * np.log10(pixels)  # convert to dB scale, more interpretable pixels
            else:
                pixels = chip["pixels"].astype(np.float32)

            pixels = torch.from_numpy(pixels)
            pixels = self.transforms[platform](pixels)

            time_tensor = torch.tensor(np.hstack((chip["week_norm"], chip["hour_norm"]), dtype=np.float32))
            latlon_tensor = torch.tensor(np.hstack((chip["lat_norm"], chip["lon_norm"]), dtype=np.float32))

            # Randomly set time & latlon to zero for 20% of the chips
            if random.random() < 0.2:  # noqa: PLR2004
                time_tensor.zero_()
                latlon_tensor.zero_()

            # Prepare additional information
            additional_info = {
                "platform": platform,
                "time": time_tensor,
                "latlon": latlon_tensor,
            }

        return {"pixels": pixels, **additional_info}


class ClaySampler(Sampler):
    def __init__(self, dataset, platforms, batch_size):
        self.dataset = dataset
        self.platforms = platforms
        self.batch_size = batch_size

        self.cubes_per_platform = {platform: [] for platform in platforms}
        for idx, chip_path in enumerate(self.dataset.chips_path):
            platform = chip_path.parent.name
            self.cubes_per_platform[platform].append(idx)

    def __iter__(self):
        cubes_per_platform_per_epoch = {}
        rng = np.random.default_rng()
        # Shuffle and adjust sizes
        max_len = max(len(indices) for indices in self.cubes_per_platform.values())
        for platform in self.platforms:
            indices = self.cubes_per_platform[platform]
            rng.shuffle(indices)
            repeated_indices = np.tile(indices, (max_len // len(indices) + 1))[:max_len]
            cubes_per_platform_per_epoch[platform] = repeated_indices

        # Create batches such that we return one platform per batch in cycle
        # Ignore the last batch if it is incomplete
        for i in range(0, max_len, self.batch_size):
            for platform in self.platforms:
                batch = cubes_per_platform_per_epoch[platform][i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    yield batch

    def __len__(self):
        return len(self.dataset.chips_path) // self.batch_size


class ClayDistributedSampler(Sampler):
    def __init__(
        self,
        dataset,
        platforms,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        self.dataset = dataset
        self.platforms = platforms
        self.batch_size = batch_size
        self.num_replicas = num_replicas if num_replicas is not None else torch.distributed.get_world_size()
        self.rank = rank if rank is not None else torch.distributed.get_rank()
        self.shuffle = shuffle
        self.epoch = 0

        self.platform_indices = {platform: [] for platform in platforms}
        for idx, chip_path in enumerate(self.dataset.chips_path):
            platform = chip_path.parent.name
            self.platform_indices[platform].append(idx)

        self.max_len = max(len(indices) for indices in self.platform_indices.values())
        self.adjusted_indices = {}
        # Normalize the length of indices for each platform by replicating the indices
        # to match the max_len
        for platform, indices in self.platform_indices.items():
            if len(indices) < self.max_len:
                extended_indices = np.tile(indices, (self.max_len // len(indices) + 1))[: self.max_len]
                self.adjusted_indices[platform] = extended_indices
            else:
                self.adjusted_indices[platform] = indices

        self.num_samples = math.ceil(((self.max_len * len(self.platforms)) - self.num_replicas) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.num_samples_per_platform = self.max_len // self.num_replicas

    def __iter__(self):
        rng = np.random.default_rng(self.epoch)
        platform_batches = {}
        for platform, indices in self.adjusted_indices.items():
            if self.shuffle:
                rng.shuffle(indices)
            # Distribute the indices to each process
            start_idx = self.rank * self.num_samples_per_platform
            end_idx = start_idx + self.num_samples_per_platform
            platform_batches[platform] = indices[start_idx:end_idx]

        for i in range(0, self.num_samples_per_platform, self.batch_size):
            for platform in self.platforms:
                batch = platform_batches[platform][i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    yield batch

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def batch_collate(batch):
    """Collate function for DataLoader.

    Merge the first two dimensions of the input tensors.
    """
    d = defaultdict(list)
    for item in batch:
        d["pixels"].append(item["pixels"])
        d["time"].append(item["time"])
        d["latlon"].append(item["latlon"])
        d["platform"].append(item["platform"])
    return {
        "pixels": rearrange(d["pixels"], "b1 b2 c h w -> (b1 b2) c h w"),
        "time": rearrange(d["time"], "b1 b2 t -> (b1 b2) t"),
        "latlon": rearrange(d["latlon"], "b1 b2 ll -> (b1 b2) ll"),
        "platform": d["platform"],
    }


class ClayDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        size: int = 224,
        metadata_path: str = "configs/metadata.yaml",
        platforms: list = [
            "landsat-c2l1",
            "landsat-c2l2-sr",
            "linz",
            "modis",
            "naip",
            "sentinel-1-rtc",
            "sentinel-2-l2a",
        ],
        batch_size: int = 10,
        num_workers: int = 8,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.size = size
        self.platforms = platforms
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.split_ratio = 0.8

    def setup(self, stage: Literal["fit", "predict"] | None = None) -> None:
        # Get list of GeoTIFF filepaths from s3 bucket or data/ folder
        # if self.data_dir.startswith("s3://"):
        #     dp = torchdata.datapipes.iter.IterableWrapper(iterable=[self.data_dir])
        #     chips_path = list(dp.list_files_by_s3(masks="*.npz"))
        # else:  # if self.data_dir is a local data path
        chips_path = sorted(list(Path(self.data_dir).glob("**/*.npz")))
        chips_platform = [chip.parent.name for chip in chips_path]
        # chips_platform = [chip.parent.parent.name for chip in chips_path]
        print(f"Total number of chips: {len(chips_path)}")

        if stage == "fit":
            trn_paths, val_paths = train_test_split(
                chips_path,
                test_size=(1 - self.split_ratio),
                stratify=chips_platform,
                shuffle=True,
            )

            self.trn_ds = EODataset(
                chips_path=trn_paths,
                size=self.size,
                platforms=self.platforms,
                metadata=self.metadata,
            )
            self.trn_sampler = ClayDistributedSampler(
                dataset=self.trn_ds,
                platforms=self.platforms,
                batch_size=self.batch_size,
            )
            self.val_ds = EODataset(
                chips_path=val_paths,
                size=self.size,
                platforms=self.platforms,
                metadata=self.metadata,
            )
            self.val_sampler = ClayDistributedSampler(
                dataset=self.val_ds,
                platforms=self.platforms,
                batch_size=self.batch_size,
            )

        elif stage == "predict":
            self.prd_ds = EODataset(
                chips_path=chips_path,
                platform=self.platform,
                metadata_path=self.metadata_path,
            )

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            num_workers=self.num_workers,
            batch_sampler=self.trn_sampler,
            collate_fn=batch_collate,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            num_workers=self.num_workers,
            batch_sampler=self.val_sampler,
            collate_fn=batch_collate,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.prd_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
