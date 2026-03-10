# Custom Modules Example

This example shows how to register and use your own backbone module with TerraTorch.

## Folder Layout

Place files like this:

```text
examples/custom_modules/
├── config.yaml
└── custom_modules/
    ├── __init__.py
    └── hello_geo_module.py
```

## Why `__init__.py` matters

TerraTorch imports the package from custom_modules_path. By default, this is set to $PWD/custom_modules (your current working directory).

Since registration decorators (like @register_model) only execute when the corresponding module is actually imported, the \__init__.py serves as the entry point to ensure your custom components are "seen" and registered by TerraTorch during the loading process.

Therefore, `custom_modules/__init__.py` must import your module class so it is registered:

```python
from custom_modules.hello_geo_module import HelloGeoModule

__all__ = ["HelloGeoModule"]
```

If this import is missing, TerraTorch cannot instantiate `backbone: HelloGeoModule`.

## Environment Setup

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If your environment already exists, just activate it:

```bash
source .venv/bin/activate
```

## Run the Example

From this directory (`examples/custom_modules`):

```bash
terratorch fit -c ./config.yaml
```

## Key Config Values

`config.yaml` uses:

- `custom_modules_path: ./custom_modules`
- `model.init_args.model_factory: EncoderDecoderFactory`
- `model.init_args.model_args.backbone: HelloGeoModule`

This tells TerraTorch to import your package and resolve `HelloGeoModule` from the backbone registry.

## Use TERRATORCH_CUSTOM_MODULE_PATH environment variable to specify an alternative path to your custom modules.

To set a different path for your custom modules, you can use the `TERRATORCH_CUSTOM_MODULE_PATH` environment variable. The `custom_modules_path` field in your config.yaml will take precedence over this environment variable, so ensure it is present or set to None if you wish to specify an alternative path this way. For example:

```bash
 os.environ["TERRATORCH_CUSTOM_MODULE_PATH"] = "examples/custom_modules_alternative/custom_modules
 terratorch fit -c "./config.yaml" # config.yaml should not set `custom_modules_path`.
 ```
