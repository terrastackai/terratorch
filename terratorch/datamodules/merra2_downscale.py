from collections.abc import Callable
from typing import Any, Optional

from granitewxc.datasets.merra2 import Merra2DownscaleDataset
from granitewxc.utils.config import ExperimentConfig
from torch._tensor import Tensor
from torch.utils.data.dataloader import DataLoader
from torchgeo.datamodules import NonGeoDataModule


class Merra2DownscaleNonGeoDataModule(NonGeoDataModule):
    def __init__(
        self,
        data_path_surface: str,
        data_path_vertical: str,
        output_vars: list[str],
        input_surface_vars: list[str] | None = None,
        input_static_surface_vars: list[str] | None = None,
        input_vertical_vars: list[str] | None = None,
        input_levels: list[float] | None = None,
        time_range: slice = None,
        climatology_path_surface: str | None = None,
        climatology_path_vertical: str | None = None,
        transforms: list[Callable] = [],
        n_input_timestamps=1,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            Merra2DownscaleDataset,
            time_range=time_range,
            data_path_surface=data_path_surface,
            data_path_vertical=data_path_vertical,
            climatology_path_surface=climatology_path_surface,
            climatology_path_vertical=climatology_path_vertical,
            input_surface_vars=input_surface_vars,
            input_static_surface_vars=input_static_surface_vars,
            input_vertical_vars=input_vertical_vars,
            input_levels=input_levels,
            n_input_timestamps=n_input_timestamps,
            output_vars=output_vars,
            transforms=transforms,
            **kwargs,
        )

        self.aug = lambda x: x

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        return super()._dataloader_factory(split)

    def setup(self, stage: str) -> None:

        if stage == "train":
            self.train_dataset = self.dataset_class(  # type: ignore[call-arg]
                **self.kwargs
            )
        if stage == "val":
            self.val_dataset = self.dataset_class(  # type: ignore[call-arg]
                **self.kwargs
            )
        if stage == "test":
            self.test_dataset = self.dataset_class(  # type: ignore[call-arg]
                **self.kwargs
            )
        if stage == "predict":
            self.predict_dataset = self.dataset_class(  # type: ignore[call-arg]
                **self.kwargs
            )

        return super().setup(stage)
