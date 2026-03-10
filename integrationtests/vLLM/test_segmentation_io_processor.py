import os
import tempfile
from pathlib import Path

import imagehash
import pytest
import requests
from PIL import Image

from .config import models, models_output, input_images
from .utils import download_files_to_local, make_request_and_get_hash, get_server, server


# Group tests by model to avoid restarting vLLM instance
# Sort by model_name first, then by other parameters
tests_per_model = sorted(
    [
        (model, image, data_format, out_data_format)
        for model in models_output.keys()
        for image in models_output[model].keys()
        for data_format in ["url", "path"]
        for out_data_format in ["b64_json", "path"]
    ],
    key=lambda x: (x[0], x[1], x[2], x[3]),
)


@pytest.mark.parametrize("model_name, image_name, data_format, out_data_format", tests_per_model)
def test_serving_segmentation_plugin(get_server, model_name, image_name, data_format, out_data_format):
    model = models[model_name]["location"]
    io_processor_plugin = models[model_name]["io_processor_plugin"]
    input_config = input_images[image_name]
    image_url = input_config["image_url"]
    expected_hash = models_output[model_name][image_name]

    server_args = [
        "--skip-tokenizer-init",
        "--enforce-eager",
        "--max-num-seqs",
        "32",
        "--io-processor-plugin",
        io_processor_plugin,
        "--model-impl",
        "terratorch",
        "--enable-mm-embeds",
    ]

    server = get_server(model, server_args=server_args)

    # Prepare input data based on data_format
    if data_format == "url":
        image_data = image_url
    else:  # path
        download_dir = Path(server.tmpdir.name) / f"downloaded_inputs_{model_name}_{image_name}"
        download_dir.mkdir(parents=True, exist_ok=True)
        image_data = download_files_to_local(image_url, download_dir)

    # Create input config with the specified formats
    test_input = {
        **input_config,
        "data_format": data_format,
        "out_data_format": out_data_format,
    }

    # Make request and verify hash
    image_hash = make_request_and_get_hash(server, model, test_input, image_data, data_format)
    assert image_hash == expected_hash


@pytest.mark.parametrize(
    "model_name,image_name",
    [
        ("terramind_base_flood", "flood"),
        ("prithvi_300m_sen1floods11", "valencia"),
    ],
)
def test_custom_out_path_override(get_server, model_name, image_name):
    """Test that the out_path field in the request overrides the plugin configuration."""
    model = models[model_name]["location"]
    io_processor_plugin = models[model_name]["io_processor_plugin"]
    input_config = input_images[image_name]

    image_url = input_config["image_url"]

    server_args = [
        "--skip-tokenizer-init",
        "--enforce-eager",
        "--max-num-seqs",
        "32",
        "--io-processor-plugin",
        io_processor_plugin,
        "--model-impl",
        "terratorch",
        "--enable-mm-embeds",
    ]

    server = get_server(model, server_args=server_args)

    # Create a custom output directory with automatic cleanup
    # In some systems /tmp might not we writable but we assume the user
    # can always write in the current directory.
    curr_dir = Path.cwd()
    with tempfile.TemporaryDirectory(dir=curr_dir) as custom_output_dir:
        request_payload = {
            "data": {
                "data": image_url,
                "data_format": "url",
                "out_data_format": "path",
                "out_path": custom_output_dir,  # Custom output path
                "image_format": "",
            },
            "model": model,
        }

        url = f"{server.instance.base_url}/pooling"
        ret = requests.post(url, json=request_payload)
        assert ret.status_code == 200

        response = ret.json()
        file_name = response["data"]["data"]

        # Verify the file was created in the custom output directory
        assert file_name.startswith(custom_output_dir), (
            f"Expected file to be in {custom_output_dir}, but got {file_name}"
        )

        # Verify the file exists
        assert os.path.exists(file_name), f"Output file {file_name} does not exist"

        # Verify the image hash matches expected output
        image_hash = str(imagehash.phash(Image.open(file_name)))
        assert image_hash == models_output[model_name][image_name]


@pytest.mark.parametrize(
    "model_name,image_name",
    [
        ("terramind_base_flood", "flood"),
        ("prithvi_300m_sen1floods11", "valencia"),
    ],
)
def test_custom_out_path_validation(get_server, model_name, image_name):
    """Test that invalid out_path raises appropriate errors."""
    from pathlib import Path

    model = models[model_name]["location"]
    io_processor_plugin = models[model_name]["io_processor_plugin"]
    input_config = input_images[image_name]

    image_url = input_config["image_url"]

    server_args = [
        "--skip-tokenizer-init",
        "--enforce-eager",
        "--max-num-seqs",
        "32",
        "--io-processor-plugin",
        io_processor_plugin,
        "--model-impl",
        "terratorch",
        "--enable-mm-embeds",
    ]

    server = get_server(model, server_args=server_args)

    # Test 1: Non-existent path should raise an error
    request_payload = {
        "data": {
            "data": image_url,
            "data_format": "url",
            "out_data_format": "path",
            "out_path": "/nonexistent/path/that/does/not/exist",
            "image_format": "",
        },
        "model": model,
    }

    url = f"{server.instance.base_url}/pooling"
    ret = requests.post(url, json=request_payload)
    assert ret.status_code != 200, "Expected error for non-existent path"

    # Test 2: Non-writable path should raise an error
    # In some systems /tmp might not we writable but we assume the user
    # can always write in the current directory.
    curr_dir = Path.cwd()
    with tempfile.TemporaryDirectory(dir=curr_dir) as tmpdir:
        readonly_dir = Path(tmpdir) / "readonly"
        readonly_dir.mkdir()

        # Set read-only permissions: 0o444 = r--r--r-- (no write permissions for anyone)
        readonly_dir.chmod(0o444)

        request_payload["data"]["out_path"] = str(readonly_dir)

        ret = requests.post(url, json=request_payload)
        assert ret.status_code != 200, "Expected error for non-writable path"
