# Initializing and serving a model with vLLM in tensor-to-tensor mode

This section shows an example of how to bootstrap a TerraTorch model on vLLM and
perform a tensor-to-tensor inference. This section assumes that you have
[prepared your model for serving with vLLM](./prepare_your_model.md).

The examples in the rest of this document will use the
[Prithvi-EO-2.0-300M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11)
model finetuned to segment the extent of floods on Sentinel-2 images from the
Sen1Floods11 dataset, the commands can be adapted to work with any other
supported models.

## Starting the vLLM serving instance

The information required to start the serving instance is the model identifier
on HuggingFace. In this example:

- Model identifier: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`

To start the serving instance, run the below command:

```bash title="Starting a vLLM serving instance"
vllm serve \
--model='ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11' \
--trust-remote-code \
--dtype float16 \
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

TerraTorch models can be served in vLLM via the `/pooling` endpoint with the
below payload

```python title="vLLM perform a tensor-to-tensor inference"

import base64
import requests
import torch
import io
import numpy as np

torch.set_default_dtype(torch.float16)


def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client", "Content-Type": "application/json"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response


def decompress(output):
    np_result = np.frombuffer(
        base64.b64decode(output), dtype=np.float32)
    return np_result.reshape(1, 2, 512, 512)


def main():
    api_url = f"http://localhost:8000/pooling"
    model_name = 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11'

    pixel_values = torch.full((6, 512, 512), 1.0, dtype=torch.float16)
    location_coords = torch.full((1, 2), 1.0, dtype=torch.float16)

    buffer_tiff = io.BytesIO()
    torch.save(pixel_values, buffer_tiff)
    buffer_tiff.seek(0)
    binary_data = buffer_tiff.read()
    base64_tensor_embedding = base64.b64encode(binary_data).decode('utf-8')

    buffer_coord = io.BytesIO()
    torch.save(location_coords, buffer_coord)
    buffer_coord.seek(0)
    binary_data = buffer_coord.read()
    base64_coord_embedding = base64.b64encode(binary_data).decode('utf-8')

    prompt = {
        "model": model_name,
        "additional_data": {
            "prompt_token_ids": [1]
        },
        "encoding_format": "base64",
        "messages": [
            {
                "role": "user",
                "content": [
                        {"type": "image_embeds",
                         "image_embeds": {
                            "pixel_values": base64_tensor_embedding,
                            "location_coords": base64_coord_embedding,
                            },
                        }
                        ],
            }]
    }

    pooling_response = post_http_request(prompt=prompt, api_url=api_url)
    numpy_data = decompress(pooling_response.json()["data"][0]["data"])
    print(f"Returned result: {numpy_data}")


if __name__ == "__main__":
    main()
```

In this example, the user sends a request payload that is already composed of
pytorch tensors that in this case contain dummy data. In a real scenario, the
user would have to pre-process the input images into tensors before submitting
them to the model for inference. In tensor-to-tensor mode, the model processes
the request and returns a response payload in the format of a numpy array.

<!-- prettier-ignore-start -->
!!! info 
    In this example the model expects two tensors in input: `pixel_values` with
    shape `(6, 512, 512)` and `location_coords` with shape `(1, 2)`. The structure of
    the input payload depends on the model itself and can be found in the `input`
    section of the model
    [config.json](./prepare_your_model.md#model-input-specification).
<!-- prettier-ignore-end -->
