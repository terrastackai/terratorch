<!---
<img src="https://github.com/user-attachments/assets/f7c9586f-6220-4a53-9669-2aee3300b492#light-only" alt="TerraTorch"  width="400"/>
<img src="assets/logo_white.png#dark-only" alt="TerraTorch"  width="400"/>
-->
<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/f8c9586f-6220-4a53-9669-2aee3300b492">
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo_white.png">
  <center><img style="display: block; margin-left: auto; margin-right: auto"; src="https://github.com/user-attachments/assets/f7c9586f-6220-4a53-9669-2aee3300b492" alt="TerraTorch"  width="400"/></center>
</picture>

<!--
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figs/logo_inv.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/figs/logo.png">
</picture>
-->

[![huggingface](https://img.shields.io/badge/Hugging_Face-join-FFD21E?logo=huggingface)](https://huggingface.co/ibm-nasa-geospatial)
[![pypi](https://badge.fury.io/py/terratorch.svg)](https://pypi.org/project/terratorch)
[![tests](https://github.com/terrastackai/terratorch/actions/workflows/test.yaml/badge.svg)](https://github.com/terrastackai/terratorch/actions/workflows/test.yaml)
[![MkDocs](https://img.shields.io/badge/MkDocs-526CFE?logo=materialformkdocs&logoColor=fff)](https://terrastackai.github.io/terratorch/)
<!--
![cov](https://github.com/terrastackai/terratorch/raw/main/assets/coverage-badge.svg)
-->
[![PyPI Downloads](https://img.shields.io/pypi/dm/terratorch.svg?label=PyPI%20downloads)](https://pypi.org/project/terratorch/)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/terratorch.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/terratorch)

Please, read the contribution guidelines (see `Contribution` below) if you want to contribute to
TerraTorch.

## Overview
TerraTorch is a PyTorch domain library based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and the [TorchGeo](https://github.com/microsoft/torchgeo) domain library
for geospatial data. 

Please also try our HPO/NAS tool: [Iterate](https://github.com/terrastackai/iterate)

## Disclaimer
TerraTorch provides tools for fine-tuning and using pretrained models.
No models are hosted by TerraTorch. TerraTorch only provides the training and inference framework.  

User responsibility: It is the sole responsibility of the user to verify that the license of any model they download, fine-tune, or deploy allows their intended use.
The TerraTorch maintainers do not provide legal advice and are not liable for any misuse of third-party models.

<hr>
<a href="https://youtu.be/LNKovSef5lU">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png" alt="YouTube" width="20">
  Video: Introduction to TerraTorch
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png" alt="YouTube" width="20">
</a>
<hr>


TerraTorch’s main purpose is to provide a flexible fine-tuning framework for Geospatial Foundation Models, which can be interacted with at different abstraction levels. The library provides:

- Convenient modelling tools:
    - Flexible trainers for Image Segmentation, Classification and Pixel Wise Regression fine-tuning tasks
    - Model factories that allow to easily combine backbones and decoders for different tasks
    - Ready-to-go datasets and datamodules that require only to point to your data with no need of creating new custom classes
    - Launching of fine-tuning tasks through CLI and flexible configuration files, or via jupyter notebooks
- Easy access to:
    - Open source pre-trained Geospatial Foundation Model backbones:
      * [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)
      * [TerraMind](https://research.ibm.com/blog/terramind-esa-earth-observation-model)
      * [SatMAE](https://sustainlab-group.github.io/SatMAE/)
      * [ScaleMAE](https://github.com/bair-climate-initiative/scale-mae)
      * Satlas (as implemented in [TorchGeo](https://github.com/microsoft/torchgeo))
      * DOFA (as implemented in [TorchGeo](https://github.com/microsoft/torchgeo))
      * SSL4EO-L and SSL4EO-S12 models (as implemented in [TorchGeo](https://github.com/microsoft/torchgeo))
      * [Clay](https://github.com/Clay-foundation/model)
    - Backbones available in the [timm](https://github.com/huggingface/pytorch-image-models) (Pytorch image models)
    - Decoders available in [SMP](https://github.com/qubvel/segmentation_models.pytorch) (Pytorch Segmentation models with pre-training backbones) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) packages
    - Fine-tuned models such as [granite-geospatial-biomass](https://huggingface.co/ibm-granite/granite-geospatial-biomass)
    - All GEO-Bench datasets and datamodules
    - All [TorchGeo](https://github.com/microsoft/torchgeo) datasets and datamodules 

## Installation

### Pip
In order to use the file `pyproject.toml` it is necessary to guarantee `pip>=21.8`. If necessary upgrade `pip` using `python -m pip install --upgrade pip`. 

For a stable point-release, use `pip install terratorch==<version>`.

[comment]: <If you prefer to get the most recent version of the main branch, install the library with `pip install git+https://github.com/terrastackai/terratorch.git`.>
To get the most recent version of the branch `main`, install the library with `pip install git+https://github.com/terrastackai/terratorch.git`.

### Conda
TerraTorch is also available on `conda-forge`, to install from there do `conda install -c conda-forge terratorch`. 

### Pipx
Alternatively, it is possible to install using [pipx](https://github.com/pypa/pipx) via `pipx install terratorch`, which creates an isolated environment and allows the user to run the application as a common CLI tool, with no need of installing dependencies or activating environments.

### Gdal
TerraTorch requires gdal to be installed, which can be quite a complex process.
If you don't have GDAL set up on your system, we recommend using a conda
environment and installing it with `conda install -c conda-forge gdal`. If you
are installing from `conda-forge` it probably won't be a problem. 

### Install as a developer
To install as a developer (e.g. to extend the library):
```
git clone https://github.com/terrastackai/terratorch.git
cd terratorch
pip install -r requirements_test.txt
conda install -c conda-forge gdal
pip install -e .
```
To install terratorch with partial (work in development) support for Weather Foundation Models, `pip install -e .[wxc]`, which currently works just for `Python >= 3.11`. 

## Documentation

To get started, check out the [quick start guide](https://terrastackai.github.io/terratorch/quick_start).

Developers, check out the [architecture overview](https://terrastackai.github.io/terratorch/architecture).

[TerraTorch: The Geospatial Foundation Models Toolkit on arXiv](https://arxiv.org/abs/2503.20563)
## Contributing

This project welcomes contributions and suggestions. Ways to contribute or get involved:

- Join our [Discord](https://discord.gg/x8AWY8QQ)
- Create an [Issue](https://github.com/terrastackai/terratorch/issues) (for bugs or feature requests)
- Contribute via [PR](https://github.com/terrastackai/terratorch/pulls)
- Join our [duoweekly](https://romeokienzler.medium.com/the-duoweekly-manifesto-eaa6c1f542c8) community calls taking place [Tuesdays 4:30 PM - 5 PM CEST](https://teams.microsoft.com/l/meetup-join/19%3ameeting_MWJhMThhMTMtMjc3MS00YjAyLWI3NTMtYTI0NDQ3NWY3ZGU2%40thread.v2/0?context=%7b%22Tid%22%3a%22fcf67057-50c9-4ad4-98f3-ffca64add9e9%22%2c%22Oid%22%3a%227f7ab87a-680c-4c93-acc5-fbd7ec80823a%22%7d) and [Thursdays 2:30 PM - 3 PM CEST](https://teams.microsoft.com/l/meetup-join/19%3ameeting_MWJhMThhMTMtMjc3MS00YjAyLWI3NTMtYTI0NDQ3NWY3ZGU2%40thread.v2/0?context=%7b%22Tid%22%3a%22fcf67057-50c9-4ad4-98f3-ffca64add9e9%22%2c%22Oid%22%3a%227f7ab87a-680c-4c93-acc5-fbd7ec80823a%22%7d).

You can find more detailed contribution guidelines [here](https://terrastackai.github.io/terratorch/stable/contributing/). 

If you want to meet the GitHub DCO checks, you **need** to do your commits as below:
```
git commit -s -m <message>
```
It will sign the commit with your ID and the check will be met. 

## Credits

TerraTorch is supported by the EU’s Horizon Europe program under Grant Agreement number 101131841 and also received funding from the Swiss State Secretariat for Education, Research and Innovation (SERI) and the UK Research and Innovation (UKRI).


## License

This project is primarily licensed under the **Apache License 2.0**. 

However, some files contain code licensed under the **MIT License**. These files are explicitly listed in [`MIT_FILES.txt`](./MIT_FILES.txt).

By contributing to this repository, you agree that your contributions will be licensed under the Apache 2.0 License unless otherwise stated.

For more details, see the [LICENSE](./LICENSE) file.
