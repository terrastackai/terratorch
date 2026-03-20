import lightning as pl
from granitewxc.datasets.eccc import EcccHrdpsGdpsDataset
from granitewxc.utils.config import ExperimentConfig
from torch.utils.data import DataLoader


class ECCCDataModule(pl.LightningDataModule):
    """Data module for the ECCC dataset"""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        self.dl_kwargs = dict(
            batch_size=config.batch_size,
            num_workers=config.dl_num_workers,
            prefetch_factor=config.dl_prefetch_size,
            pin_memory=True,
            drop_last=True,
            persistent_workers=config.dl_num_workers > 0,
        )

        self.ds_kwargs = dict(
            json_static_var_path=config.data.static_data_index,
            surface_vars=config.data.input_surface_vars,
            vertical_pres_vars=config.data.vertical_pres_vars,
            vertical_level1_vars=config.data.vertical_level1_vars,
            vertical_level2_vars=config.data.vertical_level2_vars,
            other_vars=config.data.other,
            static_vars=config.data.input_static_surface_vars,
            output_vars=config.data.output_vars,
            downsample_factor=config.data.downsample_factor,
            n_random_windows=config.data.n_random_windows,
            crop_factor=config.data.crop_factor,
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """Set up the data loaders for training, validation, and testing."""
        if stage == "fit":
            self.train_dataset = EcccHrdpsGdpsDataset(
                json_file_path=self.config.data.data_training_index, **self.ds_kwargs
            )
        if stage == "test":
            self.test_dataset = EcccHrdpsGdpsDataset(
                json_file_path=self.config.data.data_test_index, **self.ds_kwargs, test=True
            )
        if stage == "val":
            self.valid_dataset = EcccHrdpsGdpsDataset(json_file_path=self.config.data.data_val_index, **self.ds_kwargs)
        if stage == "predict":
            self.predict_dataset = EcccHrdpsGdpsDataset(
                json_file_path=self.config.data.data_test_index, **self.ds_kwargs, test=True
            )

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the training data."""
        return DataLoader(dataset=self.train_dataset, **self.dl_kwargs)

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the validation data."""
        return DataLoader(dataset=self.valid_dataset, **self.dl_kwargs)

    def test_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the test data."""
        return DataLoader(dataset=self.test_dataset, **self.dl_kwargs)

    def predict_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the prediction data."""
        return DataLoader(dataset=self.predict_dataset, **self.dl_kwargs)
