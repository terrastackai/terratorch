# Initializing and serving a model with vLLM in image-to-image mode

This section shows an example of how to bootstrap a TerraTorch model on vLLM and
perform an image-to-image inference, from a GeoTiff input to a GeoTiff output,
using an IOProcessor plugin. This section assumes that you have
[prepared your model for serving with vLLM](./prepare_your_model.md) and you
have identified the
[IOProcessor to be used](./vllm_io_plugins.md#available-terratorch-ioprocessor-plugins).

The example in the rest of this document uses the
[Prithvi-EO-2.0-300M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11)
model finetuned to segment the extent of floods on Sentinel-2 images from the
Sen1Floods11 dataset and the
[`terratorch_segmentation`](./plugins/segmentation_io_plugin.md) IOProcessor
plugin. However, the commands can be adapted to work with any other supported
models and plugins.

## Starting the vLLM serving instance

The information required to start the serving instance is the model identifier
on HuggingFace and the name of the IOProcessor plugin. In this example:

- Model identifier: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`
- IO Processor plugin name: `terratorch_segmentation`

To start the serving instance, run the below command:

```bash title="Starting a vLLM serving instance"
vllm serve ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
--io-processor-plugin terratorch_segmentation \
--model-impl terratorch \
--skip-tokenizer-init \
--enforce-eager \
--enable-mm-embeds
```

The snippet below shows the logs of a successfully initialized vLLM serving
instance

```title="vLLM instance ready to serve requests"
(APIServer pid=532339) INFO 10-01 09:01:06 [launcher.py:44] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=532339) INFO 10-01 09:01:06 [launcher.py:44] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=532339) INFO 10-01 09:01:06 [launcher.py:44] Route: /invocations, Methods: POST
(APIServer pid=532339) INFO 10-01 09:01:06 [launcher.py:44] Route: /metrics, Methods: GET
(APIServer pid=532339) INFO:     Started server process [532339]
(APIServer pid=532339) INFO:     Waiting for application startup.
(APIServer pid=532339) INFO:     Application startup complete.
```

## Send An Inference Request To The Model

TerraTorch models can be served in vLLM via the `/pooling` endpoint. The snippet
below shows an example payload that can be used to send an inference request to
the model when using the `terratorch_segmentation` IOProcessor plugin. Refer to
the documentation of the
[available IOProcessors](./vllm_io_plugins.md#available-terratorch-ioprocessor-plugins)
for more information on the expected payload format.

```python title="Request payload for the terratorch_segmentation plugin"
request_payload = {
    "data": {
        "data": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff",
        "data_format": "url",
        "out_data_format": "path",
        "image_format": "geoTiff"
    },
    "model": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
    "softmax": False
}
```

The user can save this payload in a file named `payload.json`.

<!-- prettier-ignore-start -->
!!! info
    In this example the input image is in the form of a URL. The structure of
    the input payload depends on the IO Processor plugin used with the model.
    See the list of [available plugins](./vllm_io_plugins.md#available-terratorch-ioprocessor-plugins)
    for more details.
<!-- prettier-ignore-end -->

With this payload the IOProcessor plugin will download the input geoTiff from a
URL and return the path on local filesystem of the output geoTiff.

Assuming the vLLM server is listening on `localhost:8000` the snippet below
shows how to send the inference request and retrieve the output file path.

```bash title="Request inference to the vLLM serving instance"
curl -s -H "Content-Type: application/json" \
--data @payload.json \
http://localhost:8000/pooling \
| jq -r '.data.data'

```
