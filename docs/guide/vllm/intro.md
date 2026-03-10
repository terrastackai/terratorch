# Serving TerraTorch models with vLLM

TerraTorch models can be served using the
[vLLM](https://github.com/vllm-project/vllm) serving engine. Currently, only
models using the `SemanticSegmentationTask` or `PixelwiseRegressionTask` tasks
can be served with vLLM.

TerraTorch models can be served with vLLM in _tensor-to-tensor_ or
_image-to-image_ mode. The tensor-to-tensor mode is the default mode and is
natively enabled by vLLM. For the image-to-image mode, TerraTorch uses a feature
in vLLM called
[IOProcessor plugins](https://docs.vllm.ai/en/v0.13.0/design/io_processor_plugins/#writing-an-io-processor-plugin),
enabling processing and generation of data in any modality (e.g., geoTiff). In
TerraTorch, we provide pre-defined IOProcessor plugins. Check the list of
[available plugins](./vllm_io_plugins.md#available-terratorch-ioprocessor-plugins).

To enable your model to be served via vLLM, follow the steps below:

1. **Ensure TerraTorch Integration**: Verify the model you want to serve is
   either already a core model, or learn how to
   [add your model to TerraTorch](../models.md#adding-a-new-model).
2. **Create a Model _config.json_**: Create a
   [vLLM compatible _config.json_](./prepare_your_model.md).
3. **Determine IOProcessor Plugin Needs**: If serving in image-to-image mode,
   identify an [IOProcessor plugin](./vllm_io_plugins.md) that suits your model
   or
   [build one yourself](https://docs.vllm.ai/en/latest/design/io_processor_plugins/).
4. **Make your Model Accessible to vLLM**: Host your model weights and
   config.json on Hugging Face, or store them in a local directory accessible by
   the vLLM instance.

To validate the steps above, start a vLLM serving instance that loads your model
and perform an inference in [tensor-to-tensor mode](./serving_a_model_tensor.md)
or in [image-to-image mode](./serving_a_model_image.md).
