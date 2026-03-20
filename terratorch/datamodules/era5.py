import glob
import os

import lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def get_era5_uvtp122(ds: xr.Dataset, index: int = 0) -> dict[str, torch.Tensor]:
    """Retrieve climate data variables at 122 pressure levels.

    Args:
        ds: xarray Dataset containing the ERA5 data.
        index: Time index to select data from. Defaults to 0.

    Returns:
        A dictionary containing:
        - x: Input feature vector (u, v, temperature, pressure) as a tensor.
        - y: Reordered input tensor.
        - target: Output feature vector (theta, u'omega, v'omega) as a tensor.
        - lead_time: A tensor with lead time information.
    """
    # Select the dataset for the specific time index
    ds_t0: xr.Dataset = ds.isel(time=index)
    # u: zonal wind, x-component
    u = ds_t0["features"].isel(idim=slice(3, 125))
    # v: meridional wind, y-component
    v = ds_t0["features"].isel(idim=slice(125, 247))
    # theta: potential temperature
    theta = ds_t0["features"].isel(idim=slice(247, 369))
    # pressure: pressure variable
    pressure = ds_t0["features"].isel(idim=slice(369, 491))

    # Reorder --> theta, pressure, u, v
    tensor_x = torch.tensor(data=xr.concat(objs=[theta, pressure, u, v], dim="idim").data.compute())
    assert tensor_x.shape == torch.Size([488, 64, 128])

    # Load output labels from the dataset and convert to a tensor
    tensor_y = torch.tensor(data=ds_t0["output"].data.compute())
    assert tensor_y.shape == torch.Size([366, 64, 128])

    return {
        "x": tensor_x.unsqueeze(dim=0),
        "y": tensor_x,
        "target": tensor_y,
        "lead_time": torch.zeros(1),  # Placeholder for lead time
    }


class ERA5Dataset(Dataset):
    """ERA5 Dataset loaded into PyTorch tensors.

    This is a custom Dataset class for loading ERA5 climate data into tensors,
    used for the Gravity Wave Flux downstream application.

    Attributes:
        data_path: Path to the directory containing the NetCDF files.
        file_glob_pattern: Pattern to match the NetCDF files.
        ds: The xarray Dataset containing concatenated NetCDF data.
        sur_static: Tensor representing static surface variables like sine and cosine of latitudes and longitudes.
    """

    def __init__(
        self,
        data_path: str = "data/uvtp122",
        file_glob_pattern: str = "inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_*.nc",  # or "wxc_input_u_v_t_p_*.nc", or "era5_uvtp_uw_vw_uv_*.nc"
    ):
        """Initializes the ERA5Dataset class by loading NetCDF files.

        Args:
            data_path: The directory containing the NetCDF files.
            file_glob_pattern: The file pattern to match NetCDF files.
        Raises:
            ValueError: If no NetCDF files matching the pattern are found.
        """

        nc_files: list[str] = glob.glob(pathname=os.path.join(data_path, file_glob_pattern))

        if len(nc_files) == 0:
            raise ValueError(f"No finetuning NetCDF files found at {data_path}")

        self.ds: xr.Dataset = xr.open_mfdataset(paths=nc_files, chunks={"time": 1}, combine="nested", concat_dim="time")

        # Calculate static surface variables (latitude and longitude in radians)
        latitudes = self.ds.lat.data / 360 * 2.0 * torch.pi
        longitudes = self.ds.lon.data / 360 * 2.0 * torch.pi

        # Create a meshgrid of latitudes and longitudes
        latitudes, longitudes = torch.meshgrid(torch.as_tensor(latitudes), torch.as_tensor(longitudes), indexing="ij")
        # Stack sine and cosine of latitude and longitude to create static surface tensor
        self.sur_static = torch.stack([torch.sin(latitudes), torch.cos(longitudes), torch.sin(longitudes)], axis=0)

    def __len__(self) -> int:
        """Returns the total number of timesteps in the dataset.

        Returns:
            int: The number of timesteps (length of the time dimension).
        """
        return len(self.ds.time)

    def __getitem__(self, index: int = 0) -> dict[str, torch.Tensor]:
        """Get a tensor of shape (Time, Channels, Height, Width).

        Depending on the number of levels in the dataset (defined by `idim`),
        it calls the appropriate function to load the ERA5 data for a given index.

        Args:
            index: Index to select the timestep. Defaults to 0.

        Returns:
            dict[str, torch.Tensor]: A dictionary with the following keys:
                - "x": Input feature tensor.
                - "y": Reordered input tensor.
                - "target": Output feature tensor.
                - "lead_time": Tensor containing lead time information.
                - "static": Static surface tensor.
        """

        if len(self.ds.idim) == 491:  # 122 levels, wxc_input_*.nc
            batch = get_era5_uvtp122(ds=self.ds, index=index)

        batch["static"] = self.sur_static

        return batch


class ERA5DataModule(pl.LightningDataModule):
    """
    This module handles data loading, batching, and train/validation splits.

    Attributes:
        train_data_path: Path to training data.
        valid_data_path: Path to validation data.
        file_glob_pattern: Pattern to match NetCDF files.
        batch_size: Size of each mini-batch.
        num_workers: Number of subprocesses for data loading.
    """

    def __init__(
        self,
        train_data_path: str = "data/uvtp122",
        valid_data_path: str = "data/uvtp122",
        file_glob_pattern: str = "wxc_input_u_v_t_p_output_theta_uw_vw_*.nc",
        batch_size: int = 16,
        num_data_workers: int = 8,
    ):
        """Initializes the ERA5DataModule with the specified settings.

        Args:
            train_data_path: Directory containing training data.
            valid_data_path: Directory containing validation data.
            file_glob_pattern: Glob pattern to match NetCDF files.
            batch_size: Size of mini-batches. Defaults to 16.
            num_data_workers: Number of workers for data loading.
        """
        super().__init__()
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.file_glob_pattern = file_glob_pattern

        self.batch_size: int = batch_size
        self.num_workers: int = num_data_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None) -> tuple[Dataset, Dataset]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if stage == "fit":
            self.dataset_train = ERA5Dataset(data_path=self.train_data_path, file_glob_pattern=self.file_glob_pattern)
            # self.dataset_train = self.dataset_train.to(device)
            self.dataset_val = ERA5Dataset(data_path=self.valid_data_path, file_glob_pattern=self.file_glob_pattern)
            # self.dataset_val = self.dataset_val.to(device)
        elif stage == "predict":
            self.dataset_predict = ERA5Dataset(data_path=self.valid_data_path, file_glob_pattern=self.file_glob_pattern)
            # self.dataset_predict = self.dataset_predict.to(device)

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the training data."""
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            sampler=DistributedSampler(dataset=self.dataset_train, shuffle=True),
        )

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the validation data."""

        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            sampler=DistributedSampler(dataset=self.dataset_val, shuffle=False),
        )

    def predict_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the prediction data."""
        return DataLoader(
            dataset=self.dataset_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
