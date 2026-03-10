# Segmentation IOProcessor Plugin

This plugin targets segmentation tasks and allows for the input image to be
split in tiles of a size that depends on the model and on an arbitrary number of
bands.

During initialization, the plugin accesses the model's data module configuration
from the vLLM configuration and instantiates a DataModule object dynamically.

This plugin is installed as `terratorch_segmentation`.

## Plugin specification

### Model requirements

This plugin expects the model to take two parameters for inference. The first,
named `pixel_values`, points to a tensor containing the raw image data extracted
from the input tiff. The second parameter, named `location_coords`, is optional
and points to a tensor containing geospatial coordinates for the image.

Below an example input model specification accepted by this plugin. The user can
change the shapes of the tensors according to their model requirements but the
number and names of the fields must be kept unchanged.

```json title="Model input specification accepted by the Segmentation IOProcessor plugin"
"input":{
    "target": "pixel_values",
    "data":{
        "pixel_values":{
            "type": "torch.Tensor",
            "shape": [6, 512, 512]
        },
        "location_coords":{
            "type":"torch.Tensor",
            "shape": [1, 2]
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

Depending on the values set in `data_format`, the plugin expects `data` to
contain a string that complies to the format. Similarly, `out_data_format`
controls the data format returned to the user. The field `indices` can be
customised by the user and it is expected to be a list of integers.

The optional `out_path` field allows you to specify a custom output directory
for storing output files when `out_data_format` is set to `'path'`. If provided,
this path overrides the plugin's default `output_path` configuration. The
specified path must exist and be writable, otherwise a `ValueError` will be
raised during request processing.

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
          "w_crop": 256
        }
      }
    }
  }
}
```

Please note, the `tiled_inference_parameters` field is not mandatory in the
model configuration. Full details on the model configuration file can be found
[here](../prepare_your_model.md#vllm-compatible-model-configuration).

#### Default Output Directory

If no `out_path` is specified in the request payload and no output folder is
configured in the plugin configuration (via the
`TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG` environment variable), the plugin
will default to writing output files to the user's home directory.

#### Image Input Indices

By default the plugin extracts bands at indices `[0, 1, 2, 3, 4, 5]` from the
input image. The user can customise this for each image, by setting the
`indices` field accordingly in the inference request payload.
