import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import imagehash
import pytest
import requests
from PIL import Image


class VLLMServer:
    def __init__(
        self,
        model_name: str,
        server_args: Optional[list[str]],
        server_envs: Optional[dict[str, str]] = None,
        port: int = 8000,
        timeout: int = 240,
    ):
        self.port = port
        self.base_url = f"http://localhost:{port}"
        cmd = ["vllm", "serve", model_name, "--port", str(port)] + server_args

        env = os.environ.copy()
        if server_envs:
            env.update(server_envs)
        self.proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)

        # Wait for server to be ready
        url = f"{self.base_url}/health"
        start_time = time.time()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    break
            except Exception:
                if time.time() - start_time > timeout:
                    # If still running let's kill the process
                    self.kill_proc()
                    raise TimeoutError("vLLM server did not start within timeout.")
            time.sleep(1)

    def kill_proc(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
                print("vLLM server terminated.")
            except subprocess.TimeoutExpired:
                self.proc.kill()
                print("vLLM server forcefully killed.")

    def __del__(self):
        self.kill_proc()


@pytest.fixture(scope="session")
def server():
    class Holder:
        instance = None
        tmpdir = None
        model_name = None

        def _delete_server(self):
            if self.instance:
                self.instance.kill_proc()
                self.tmpdir.cleanup()

        def init_server(self, model_name, **kwargs):
            self._delete_server()
            curr_dir = Path.cwd()
            self.tmpdir = tempfile.TemporaryDirectory(dir=curr_dir)
            plugin_config = {"output_path": self.tmpdir.name}
            server_envs = {
                "TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG": json.dumps(plugin_config),
                "VLLM_LOGGING_LEVEL": "DEBUG",
            }
            # 10 minutes timeout for vLLM to start
            self.instance = VLLMServer(model_name, server_envs=server_envs, timeout=600, **kwargs)
            self.model_name = model_name
            return self

    return Holder()


@pytest.fixture
def get_server(server):
    def _get(model_name, **kwargs):
        if server.instance is None or server.model_name != model_name:
            return server.init_server(model_name=model_name, **kwargs)
        return server

    return _get


def download_files_to_local(image_url, download_dir):
    """Download files from URLs to local paths.

    For multi-modal inputs (dict), creates subdirectories for each modality
    and returns the root directory path. For single files, returns the file path.
    """
    if isinstance(image_url, dict):
        # Multi-modal input (e.g., TerraMind models)
        # Create subdirectories for each modality
        for modality, url in image_url.items():
            modality_dir = download_dir / modality
            modality_dir.mkdir(exist_ok=True)
            local_file = modality_dir / Path(url).name
            response = requests.get(url)
            local_file.write_bytes(response.content)
        # Return the root directory path for multi-modal inputs
        return str(download_dir)
    else:
        # Single file input
        local_file = download_dir / Path(image_url).name
        response = requests.get(image_url)
        local_file.write_bytes(response.content)
        return str(local_file)


def make_request_and_get_hash(server, model, input_config, image_data, data_format):
    """Make a request and return the image hash."""
    request_payload = {
        "data": {
            "data": image_data,
            "data_format": data_format,
            "out_data_format": input_config["out_data_format"],
            "image_format": "",
        },
        "model": model,
        "softmax": False,
    }

    if "indices" in input_config:
        request_payload["data"]["indices"] = input_config["indices"]

    if "out_path" in input_config:
        request_payload["data"]["out_path"] = input_config["out_path"]

    url = f"{server.instance.base_url}/pooling"
    ret = requests.post(url, json=request_payload)
    assert ret.status_code == 200

    response = ret.json()

    if request_payload["data"]["out_data_format"] == "b64_json":
        decoded_image = base64.b64decode(response["data"]["data"])
        file_name = Path(server.tmpdir.name) / f"{uuid.uuid4()}.tiff"
        file_name.write_bytes(decoded_image)
    else:
        file_name = Path(response["data"]["data"])

    return str(imagehash.phash(Image.open(file_name)))
