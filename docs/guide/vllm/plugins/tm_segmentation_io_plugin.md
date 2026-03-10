# Terramind Segmentation IOProcessor Plugin

This plugin targets segmentation tasks for Terramind models, and assumes
multimodal input data to be provided via URLs or file paths, organized in
separate directories by modality. The plugin performs a tiled inference
according to the parameters given in the model configuration. During
initialization, the plugin accesses the model's data module configuration from
the vLLM configuration and instantiates a DataModule object dynamically.

Currently, the plugin is targeting TerraMind models finetuned on the
[ImpactMesh](https://github.com/IBM/ImpactMesh) dataset and expects exactly
three modalities to be given in input: DEM, S1RTC, S2L2A.

This plugin is installed as `terratorch_tm_segmentation`.

## Plugin specification

### Model requirements

This plugin expects the model to take three parameters for inference, one per
input modality.

Below an example input model specification accepted by this plugin. The user can
change the shapes of the tensors according to their model requirements but the
number and names of the fields must be kept unchanged to work with ImpactMesh
data modules.

```json title="Model input specification accepted by the Terramind Segmentation IOProcessor plugin"
"input": {
  "data": {
    "S2L2A": {
      "type": "torch.Tensor",
      "shape": [
        12,
        4,
        256,
        256
      ]
    },
    "S1RTC": {
      "type": "torch.Tensor",
      "shape": [
        2,
        4,
        256,
        256
      ]
    },
    "DEM": {
      "type": "torch.Tensor",
      "shape": [
        1,
        4,
        256,
        256
      ]
    }
  }
}
```

Full details on TerraTorch models input model specification for vLLM are
available [here](../prepare_your_model.md#model-input-specification).

### Plugin configuration

This plugin allows for additional configuration data to be passed via the
`TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG` environment variable. If set, the
variable should contain the plugin configuration in json string format.

The plugin configuration format is defined in the `PluginConfig` class.

:::terratorch.vllm.plugins.segmentation.types.PluginConfig

### Request Data Format

The input format for the plugin is defined in the `RequestData` class.

:::terratorch.vllm.plugins.segmentation.types.RequestData

The `indices` field is ignored by this plugin.

The optional `out_path` field allows you to specify a custom output directory
for the generated GeoTiff file on a per requests basis, when `out_data_format`
is set to `"path"`. If `out_path` is not provided, the plugin will use the
default output path from the plugin configuration (set via the
`TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG` environment variable).

**Example request payload with URL input and base64 output:**

```json
{
  "data_format": "url",
  "out_data_format": "b64_json",
  "data": {
    "DEM": "https://example.com/path/to/dem_file",
    "S1RTC": "https://example.com/path/to/S1RTC_file",
    "S2L2A": "https://example.com/path/to/S2L2A_file"
  }
}
```

**Example request payload with path input and path output:**

```json
{
  "data_format": "path",
  "out_data_format": "path",
  "data": "/path/to/input/directory"
}
```

**Example request payload with URL input and custom path output:**

```json
{
  "data_format": "url",
  "out_data_format": "path",
  "out_path": "/custom/output/directory",
  "data": {
    "DEM": "https://example.com/path/to/dem_file",
    "S1RTC": "https://example.com/path/to/S1RTC_file",
    "S2L2A": "https://example.com/path/to/S2L2A_file"
  }
}
```

#### Multimodal Data Organization

The structure of the `data` field in the RequestData structure depends on the
`data_format` field. When using URL-based input (`data_format: "url"`), the
plugin expects one URL for each modality file.

For example, your request includes:

```json
{
  "data_format": "url",
  "data": {
    "DEM": "https://example.com/path/to/dem_file",
    "S1RTC": "https://example.com/path/to/S1RTC_file",
    "S2L2A": "https://example.com/path/to/S2L2A_file"
  }
}
```

When using path-based input (`data_format: "path"`), provide the root directory
path that already contains the modality subdirectories organized in the same
structure.

```json
{
  "data_format": "path",
  "data": "/path/to/input/directory/"
}
```

Your directory structure should look like this:

```
/path/to/input/directory/
├── DEM/
│   └── FILE_NAME_DEM.tiff
├── S1RTC/
│   └── FILE_NAME_S1RTC.zarr.zip
└── S2L2A/
    └── FILE_NAME_S2L2A.zarr.zip
```

Each modality has its own subdirectory containing the respective data files.

<!-- prettier-ignore-start -->
!!! warning "One input bundle per request supported"
    The plugin currently supports only one input bundle per request
    (one file per modality). Do not place more than one file in each subfolder.
<!-- prettier-ignore-end -->

### Request Output Format

The output format for the plugin is defined in the `RequestOutput` class.

:::terratorch.vllm.plugins.segmentation.types.RequestOutput

### Plugin Defaults

#### Tiled Inference Parameters

By default the plugin uses the same horizontal and vertical crop value of 512
when computing image tiles. Users can use different crop values by specifying
them in their model `config.json` file. See the example below that overrides the
default values with vertical and horizontal crop values of 256.

```json title="Custom tiled inference parameters in model configuration"
{
  "pretrained_cfg": {
    "model": {
      "init_args": {
        "tiled_inference_parameters": {
          "h_crop": 256,
          "w_crop": 256,
          "delta": 8
        }
      }
    }
  }
}
```

Please note, the `tiled_inference_parameters` field is not mandatory in the
model configuration. Full details on the model configuration file can be found
[here](../prepare_your_model.md#vllm-compatible-model-configuration).

Full details on the available tiled inference parameters are available in the
`TiledInferenceParameters` class.
:::terratorch.vllm.plugins.segmentation.types.TiledInferenceParameters

#### Default Output Directory

If no `out_path` is specified in the request payload and no output folder is
configured in the plugin configuration (via the
`TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG` environment variable), the plugin
will default to writing output files to the user's home directory. This default
only impacts requests that set `out_data_format: "path"`.

#### Data Module Configuration

This plugin dynamically instantiates a data module based on the configuration in
the model's `config.json` file. The data module is then used for loading the
input data. By default, the plugin configures the data module in `predict` mode
and sets the `predict_data_root` of the DataModule to the input data folder.
More info on the data module configuration can be found
[here](../prepare_your_model.md#data-module-configuration).

<!-- prettier-ignore-start -->
!!! info "Using a different data module"
    This plugin currently supports
    [ImpactMesh](https://github.com/IBM/ImpactMesh), imposing a certain structure
    for the input data. e.g., a `DEM` input file or subfolder is always expected to
    be present, and is used for retrieving the input file metadata. Users interested
    in using a different data module might do so but they will have to guarantee the
    same behavior as the ImpactMesh ones.
<!-- prettier-ignore-end -->
