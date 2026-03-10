# How to Add Custom Modules to TerraTorch

TerraTorch is designed to be extensible, allowing you to integrate your own custom components, such as models (backbones, decoders), tasks, datamodules, callbacks, or augmentation transforms, into the fine-tuning pipeline. This is primarily achieved by making your custom Python code discoverable by TerraTorch and then referencing your custom classes within the YAML configuration file.

This tutorial outlines the steps required to add and use a custom module. We'll use a simple custom model component as an example.

## Prerequisites

*   A working installation of TerraTorch.
*   Your custom Python code (e.g., a new model architecture implemented as a `torch.nn.Module`).

## Step 1: Create Your Custom Module File(s)

First, organize your custom code into Python files (`.py`). It's recommended to place these files within a dedicated directory. For example, let's create a directory named `custom_modules` in your project's working directory and place our custom model definition inside a file named `my_custom_model.py`:

```text
my_project_root/
├── custom_modules/
│   ├── __init__.py       <-- **Required** to make the directory a Python package
│   └── my_custom_model.py
├── my_config.yaml
└── ... (other project files)
```

This is important!

```python
#custom_modules/__init__.py
from custom_modules.my_custom_model import MySimpleCNN

__all__ = ["MySimpleCNN"]

```

Inside `custom_modules/my_custom_model.py`, define your custom class or function. If you intend for TerraTorch's factories to discover this module (e.g., to use it as a backbone or decoder selected by name), you **must** register it using the appropriate registry decorator.

For instance, to register a simple custom CNN as a backbone:

```python
# custom_modules/my_custom_model.py
import torch
import torch.nn as nn
# Import the relevant registry
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

# Register the class with the backbone registry
@TERRATORCH_BACKBONE_REGISTRY.register
class MySimpleCNN(nn.Module):
    # Note: Backbones typically don't take num_classes directly in __init__
    # They output features which are then processed by a decoder/head.
    # This example is simplified for demonstration.
    def __init__(self, in_channels: int, out_features: int = 512): # Example: output feature size
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Example pooling
        self.fc = nn.Linear(32, out_features) # Example final layer
        print(f"Initialized MySimpleCNN backbone with in_channels={in_channels}, out_features={out_features}")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]: # Backbones often return a list of features
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # Return as a list, mimicking multi-stage feature outputs
        return [x]

# You could also define custom tasks, datamodules, decoders (registering with TERRATORCH_DECODER_REGISTRY), etc. here
# from terratorch.tasks import SemanticSegmentationTask
# class MyCustomTask(SemanticSegmentationTask):
#     # ... override methods ...
#     pass
```

The `@TERRATORCH_BACKBONE_REGISTRY.register` line makes `MySimpleCNN` available to be selected by name (i.e., `"MySimpleCNN"`) in the `backbone` field of your model configuration. Similar registries exist for decoders (`TERRATORCH_DECODER_REGISTRY`), necks, and full models.

Make sure your custom classes inherit from appropriate base classes if needed (e.g., `torch.nn.Module` for models, `lightning.pytorch.LightningModule` or `terratorch.tasks.BaseTask` for tasks, `lightning.pytorch.LightningDataModule` for datamodules).

## Step 2: Inform TerraTorch About Your Custom Module Directory

TerraTorch needs to know where to find your custom code. You can specify the path to your custom modules directory using the `custom_modules_path` argument either in your YAML configuration file or directly via the command line.

**Option A: In YAML Configuration (`my_config.yaml`)**

Add the `custom_modules_path` key at the top level of your configuration:

```yaml
# my_config.yaml
custom_modules_path: ./custom_modules  # Path relative to where you run terratorch

# ... rest of your configuration ...

model:
  # ... other model config ...
data:
  # ... data config ...
trainer:
  # ... trainer config ...

```

**Option B: Via Command Line**

```bash
terratorch fit --config my_config.yaml --custom_modules_path ./custom_modules
```

When provided, TerraTorch will add the specified directory (`./custom_modules` in this case) to Python's `sys.path`, making the modules within it importable.

### How `backbone: HelloGeoModule` is resolved at runtime

Given a config like:

```yaml
init_args:
  model_factory: EncoderDecoderFactory
  model_args:
    backbone: HelloGeoModule
```

the runtime flow is:

1. TerraTorch reads `custom_modules_path` and imports that package early in CLI startup.
2. Importing your package executes decorators such as `@TERRATORCH_BACKBONE_REGISTRY.register`.
3. `EncoderDecoderFactory` calls `_get_backbone(...)`, which calls `BACKBONE_REGISTRY.build("HelloGeoModule", ...)`.
4. The registry instantiates the class registered under the name `HelloGeoModule`.

So the class name is not resolved by Python reflection from YAML directly; it is resolved by TerraTorch's registry.

Important implication: your package import must execute the file that contains the decorator.
In practice, ensure your `custom_modules/__init__.py` imports the module that defines `HelloGeoModule`.

```python
# custom_modules/__init__.py
from custom_modules.hello_geo_module import HelloGeoModule
```

If that import is missing, registration never runs and you will see an error like:
`The model HelloGeoModule could not be instantiated from any source.`

## Step 3: Use Your Custom Component in the Configuration

Now that TerraTorch can find and import your custom code, you can reference your custom classes in the YAML configuration.

*   **For registered components (Recommended for backbones, decoders, etc.):** If you registered your class (like `MySimpleCNN` above), you can simply use its class name as the identifier:

    ```yaml
    # my_config.yaml
    custom_modules_path: ./custom_modules

    model:
      class_path: terratorch.models.EncoderDecoder # Example using a standard factory
      init_args:
        backbone: "MySimpleCNN" # Reference the registered custom backbone by name
        decoder: "UNetDecoder" # Example using a standard decoder
        model_args:
          backbone_in_channels: 3 # Passed to MySimpleCNN.__init__
          # backbone_out_features: 512 # Default or specify if needed
          decoder_num_classes: 5 # Example number of output classes
          # ... other args for backbone/decoder if needed
    # ... rest of config
    ```

*   **For components not using the registry or for full class path reference:** You can always reference a class using its full Python path: `<directory_name>.<filename>.<ClassName>`. This is useful for custom tasks, callbacks, or if you choose not to register a model component and instead specify its full path.

    ```yaml
    # my_config.yaml
    custom_modules_path: ./custom_modules

    # Example: Using the custom model directly via class_path (less common for backbones/decoders)
    # model:
    #   class_path: custom_modules.my_custom_model.MySimpleCNN # Using full path
    #   init_args:
    #     in_channels: 3
    #     out_features: 512

    # Example: Using a custom task
    task:
      class_path: custom_modules.my_custom_task_module.MyCustomTask # Assuming it's defined elsewhere
      init_args:
        # ... task args ...

    # Example: Using a custom callback
    trainer:
      callbacks:
        - class_path: custom_modules.my_custom_callbacks.MyCallback
          init_args:
            # ... callback args ...
    # ... rest of config
    ```

## Summary

By following these steps:
1.  Creating your custom Python code in a dedicated directory.
2.  Specifying the path to this directory using `custom_modules_path`.
3.  Referencing your custom classes via their full Python path in the YAML configuration.

You can seamlessly integrate your own modules into the TerraTorch framework. 
