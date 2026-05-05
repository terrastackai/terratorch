## Multitemporal Data

This folder contains examples for working with multitemporal data in TerraTorch.

Standard spatial data is processed as 4D tensors `(B, C, H, W)`. Multitemporal data adds a time dimension and is represented as 5D tensors `(B, C, T, H, W)` in TerraTorch.

To work with temporal data, you need a compatible datamodule. For example, the generic TerraTorch datamodules support an `expand_temporal_dimension` argument, which reshapes inputs from `(T·C, H, W)` to `(C, T, H, W)`.  
We also provide a `MultiTemporalCropClassificationDataModule`, adapted for the HLS multitemporal crop classification dataset and used in the examples below.

From there, you have two main modeling options:

1. **Temporal backbone**  
   Use a backbone with native temporal modeling, such as Prithvi.

2. **Non-temporal backbone + TemporalWrapper**  
   Wrap any TerraTorch backbone with the `TemporalWrapper` and configure temporal aggregation in latent space.

### Examples
This folder includes a demo notebook and sample YAML for both approaches. Both examples use the same input data and task, and focus on differences in model initialization (`model_args`).

- **Prithvi Multitemporal**: [`multitemporal_model_segementation_crop.ipynb`](multitemporal_model_segementation_crop.ipynb) 
  ([Open in Colab](https://colab.research.google.com/github/torchgeo/terratorch/blob/main/examples/multitemporal_data/multitemporal_model_segementation_crop.ipynb))

- **TemporalWrapper**: [`temporalwrapper_segementation_crop.ipynb`](temporalwrapper_segementation_crop.ipynb)  
  ([Open in Colab](https://colab.research.google.com/github/torchgeo/terratorch/blob/main/examples/multitemporal_data/temporalwrapper_segementation_crop.ipynb))