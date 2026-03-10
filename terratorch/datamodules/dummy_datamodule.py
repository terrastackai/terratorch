"""
Dummy Data Module for Testing
Generates random data on-the-fly without requiring actual data files.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


class DummyDataset(Dataset):
    """
    Dataset that generates random images and masks on-the-fly.
    
    Returns dict with keys:
        - "image": Random tensor of shape (channels, height, width)
        - "mask": Random tensor of shape (height, width) for regression
                  or (height, width) with integer values for segmentation
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        channels: int = 3,
        height: int = 256,
        width: int = 256,
        image_mean: float = 0.0,
        image_std: float = 1.0,
        mask_mean: float = 0.0,
        mask_std: float = 1.0,
        segmentation: bool = False,
        num_classes: int = 10,
    ):
        """
        Args:
            num_samples: Number of samples in the dataset
            channels: Number of image channels (default: 3 for RGB)
            height: Image height in pixels
            width: Image width in pixels
            image_mean: Mean for random image generation
            image_std: Standard deviation for random image generation
            mask_mean: Mean for random mask generation (regression only)
            mask_std: Standard deviation for random mask generation (regression only)
            segmentation: If True, generates integer masks for segmentation
            num_classes: Number of classes for segmentation masks
        """
        self.num_samples = num_samples
        self.channels = channels
        self.height = height
        self.width = width
        self.image_mean = image_mean
        self.image_std = image_std
        self.mask_mean = mask_mean
        self.mask_std = mask_std
        self.segmentation = segmentation
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random image (channels, height, width)
        image = torch.randn(self.channels, self.height, self.width)
        image = image * self.image_std + self.image_mean
        
        # Generate random mask (height, width)
        if self.segmentation:
            # For segmentation: random class labels
            mask = torch.randint(0, self.num_classes, (self.height, self.width))
        else:
            # For regression: random continuous values
            mask = torch.randn(self.height, self.width)
            mask = mask * self.mask_std + self.mask_mean
        
        return {
            "image": image,
            "mask": mask
        }
    
    def plot(dummy_sample,dummy_mask):
        None


class DummyDataModule(pl.LightningDataModule):
    """
    Lightning DataModule that provides random data for testing.
    
    Perfect for testing custom modules without needing actual data files.
    Generates data on-the-fly with configurable dimensions and properties.
    """
    
    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 2,
        train_samples: int = 100,
        val_samples: int = 20,
        test_samples: int = 20,
        channels: int = 3,
        height: int = 256,
        width: int = 256,
        image_mean: float = 0.0,
        image_std: float = 1.0,
        mask_mean: float = 0.0,
        mask_std: float = 1.0,
        segmentation: bool = False,
        num_classes: int = 10,
    ):
        """
        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            train_samples: Number of training samples
            val_samples: Number of validation samples
            test_samples: Number of test samples
            channels: Number of image channels (default: 3 for RGB)
            height: Image height in pixels
            width: Image width in pixels
            image_mean: Mean for random image generation
            image_std: Standard deviation for random image generation
            mask_mean: Mean for random mask generation (regression only)
            mask_std: Standard deviation for random mask generation (regression only)
            segmentation: If True, generates integer masks for segmentation
            num_classes: Number of classes for segmentation masks
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.channels = channels
        self.height = height
        self.width = width
        self.image_mean = image_mean
        self.image_std = image_std
        self.mask_mean = mask_mean
        self.mask_std = mask_std
        self.segmentation = segmentation
        self.num_classes = num_classes
    
    def setup(self, stage=None):
        """Setup datasets for each stage."""
        dataset_kwargs = {
            "channels": self.channels,
            "height": self.height,
            "width": self.width,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "mask_mean": self.mask_mean,
            "mask_std": self.mask_std,
            "segmentation": self.segmentation,
            "num_classes": self.num_classes,
        }
        
        if stage == "fit" or stage is None:
            self.train_dataset = DummyDataset(
                num_samples=self.train_samples,
                **dataset_kwargs
            )
            self.val_dataset = DummyDataset(
                num_samples=self.val_samples,
                **dataset_kwargs
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = DummyDataset(
                num_samples=self.test_samples,
                **dataset_kwargs
            )
    
    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
