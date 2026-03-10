# vLLM IOProcessor Plugins

vLLM's IOProcessor plugins are a mechanism that enables processing of
input/output inference data from/to any modality. So, as an example, these
plugins allow for the output of a model to be transformed into an image.

TerraTorch provides plugins for the handling of input/output GeoTiff images when
serving models via vLLM.

More information can be found in the
[vLLM official documentation](https://docs.vllm.ai/en/latest/design/io_processor_plugins.html).

## Using IOProcessor Plugins

IOProcessor plugins are instantiated at vLLM startup time via a dedicated flag
`--io_processor_plugin`. The snippet below shows an example of a vLLM server
started for serving a TerraTorch model using the
`terratorch_segmentation_plugin`.

```bash
vllm serve \
    --model=ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
    --model-impl terratorch \
    --task embed --trust-remote-code \
    --skip-tokenizer-init --enforce-eager \
    --io-processor-plugin terratorch_segmentation
```

Inference requests are then sent to the vLLM server URL under the `/pooling`
endpoint.

The format of the inference payload is plugin dependent. Check the list of
[available plugins](#available-terratorch-ioprocessor-plugins) to see the
plugin-specific payload format.

## Available TerraTorch IOProcessor plugins

| Plugin name                                                          | Tasks Supported       | Description                                                                                                                                                                                                                      |
| -------------------------------------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [terratorch_segmentation](./plugins/segmentation_io_plugin.md)       | Semantic Segmentation | Plugin operations: Splits the image in tiles, performs inference on all the tiles and creates a GeoTiff out of all the inference outputs.<br>input format: GeoTiff<br> output format: GeoTiff                                    |
| [terratorch_tm_segmentation](./plugins/tm_segmentation_io_plugin.md) | Semantic Segmentation | Plugin operations: Handles multimodal inputs (e.g., DEM, S1RTC, S2L2A) organized by modality, performs tiled inference and creates a GeoTiff output.<br>input format: multimodal (GeoTiff, zarr.zip) <br> output format: GeoTiff |
