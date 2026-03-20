# Copyright contributors to the Terratorch project

import pdb
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torchvision.models.detection
from torch import nn
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from terratorch.models.model import (
    Model,
    ModelFactory,
    ModelOutput,
)
from terratorch.models.necks import NeckSequential, build_neck_list
from terratorch.models.utils import extract_prefix_keys
from terratorch.registry import MODEL_FACTORY_REGISTRY

from .utils import TerratorchGeneralizedRCNNTransform, _get_backbone

SUPPORTED_TASKS = ["object_detection"]


def _check_all_args_used(kwargs):
    """
    Check if all arguments are used.

    Args:
        kwargs: dict: Dictionary of arguments.

    """
    if kwargs:
        msg = f"arguments {kwargs} were passed but not used."
        raise ValueError(msg)


@MODEL_FACTORY_REGISTRY.register
class ObjectDetectionModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        framework: str,
        num_classes: int | None = None,
        necks: list[dict] | None = None,
        **kwargs,
    ) -> Model:
        """
        Generic model factory that combines an encoder and necks with the detection models, called framework, in torchvision.detection.

        Further arguments to be passed to the backbone_ and framework_.

        Args:
            task (str): Task to be performed. Currently supports "object_detection".
            backbone (str, nn.Module): Backbone to be used. If a string, will look for such models in the different
                registries supported (internal terratorch registry, timm, ...). If a torch nn.Module, will use it
                directly. The backbone should have and `out_channels` attribute and its `forward` should return a list[Tensor].
            framework (str): object detection framework to be used between "faster-rcnn", "fcos", "retinanet" for object detection and "mask-rcnn" for instance segmentation.
            num_classes (int, optional): Number of classes. None for regression tasks.
            necks (list[dict]): nn.Modules to be called in succession on encoder features
                before passing them to the decoder. Should be registered in the NECKS_REGISTRY registry.
                Expects each one to have a key "name" and subsequent keys for arguments, if any.
                Defaults to None, which applies the identity function.

        Returns:
            nn.Module: Full torchvision detection model.
        """
        task = task.lower()
        if task not in SUPPORTED_TASKS:
            msg = f"Task {task} not supported. Please choose one of {SUPPORTED_TASKS}"
            raise NotImplementedError(msg)
        backbone_kwargs, kwargs = extract_prefix_keys(kwargs, "backbone_")
        framework_kwargs, kwargs = extract_prefix_keys(kwargs, "framework_")

        backbone = _get_backbone(backbone, **backbone_kwargs)
        if "in_channels" in kwargs.keys():
            in_channels = kwargs["in_channels"]
        else:
            in_channels = (
                len(backbone_kwargs["model_bands"])
                if "model_bands" in backbone_kwargs.keys()
                else len(backbone_kwargs["bands"])
            )

        try:
            out_channels = backbone.out_channels
        except AttributeError as e:
            msg = "backbone must have out_channels attribute"
            raise AttributeError(msg) from e
        # pdb.set_trace()
        if necks is None:
            necks = []
        neck_list, channel_list = build_neck_list(necks, out_channels)

        neck_module = NeckSequential(*neck_list)

        combined_backbone = BackboneWrapper(backbone, neck_module, channel_list)
        # pdb.set_trace()

        if framework == "faster-rcnn":
            sizes = ((32), (64), (128), (256), (512))
            sizes = sizes[: len(combined_backbone.channel_list)]
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(sizes)
            anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

            roi_pooler = MultiScaleRoIAlign(
                featmap_names=["feat0", "feat1", "feat2", "feat3"], output_size=7, sampling_ratio=2
            )

            model = torchvision.models.detection.FasterRCNN(
                combined_backbone,
                num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                _skip_resize=True,
                image_mean=np.repeat(0, in_channels),
                image_std=np.repeat(1, in_channels),
                **framework_kwargs,
            )

        elif framework == "fcos":
            sizes = ((8,), (16,), (32,), (64,), (128,), (256,))
            sizes = sizes[: len(combined_backbone.channel_list)]
            aspect_ratios = ((1.0,), (1.0,), (1.0,), (1.0,), (1.0,), (1.0,)) * len(sizes)
            anchor_generator = AnchorGenerator(
                sizes=sizes,
                aspect_ratios=aspect_ratios,
            )

            model = torchvision.models.detection.FCOS(
                combined_backbone,
                num_classes,
                anchor_generator=anchor_generator,
                _skip_resize=True,
                image_mean=np.repeat(0, in_channels),
                image_std=np.repeat(1, in_channels),
                **framework_kwargs,
            )
        elif framework == "retinanet":
            sizes = (
                (16, 20, 25),
                (32, 40, 50),
                (64, 80, 101),
                (128, 161, 203),
                (256, 322, 406),
                (512, 645, 812),
            )
            sizes = sizes[: len(combined_backbone.channel_list)]
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(sizes)
            anchor_generator = AnchorGenerator(sizes, aspect_ratios)
            head = RetinaNetHead(
                combined_backbone.out_channels,
                anchor_generator.num_anchors_per_location()[0],
                num_classes,
                norm_layer=partial(torch.nn.GroupNorm, 32),
            )

            model = torchvision.models.detection.RetinaNet(
                combined_backbone,
                num_classes,
                anchor_generator=anchor_generator,
                head=head,
                _skip_resize=True,
                image_mean=np.repeat(0, in_channels),
                image_std=np.repeat(1, in_channels),
                **framework_kwargs,
            )

        elif framework == "mask-rcnn":
            sizes = ((32), (64), (128), (256), (512))
            sizes = sizes[: len(combined_backbone.channel_list)]
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(sizes)
            anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

            rpn_head = torchvision.models.detection.faster_rcnn.RPNHead(
                combined_backbone.out_channels, anchor_generator.num_anchors_per_location()[0], conv_depth=2
            )
            box_head = torchvision.models.detection.faster_rcnn.FastRCNNConvFCHead(
                (combined_backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
            )
            mask_head = torchvision.models.detection.mask_rcnn.MaskRCNNHeads(
                combined_backbone.out_channels, [256, 256, 256, 256], 1, norm_layer=nn.BatchNorm2d
            )
            roi_pooler = MultiScaleRoIAlign(
                featmap_names=["feat0", "feat1", "feat2", "feat3"], output_size=7, sampling_ratio=2
            )

            model = torchvision.models.detection.MaskRCNN(
                combined_backbone,
                num_classes=num_classes,
                rpn_anchor_generator=anchor_generator,
                rpn_head=rpn_head,
                box_head=box_head,
                box_roi_pool=roi_pooler,
                mask_roi_pool=roi_pooler,
                mask_head=mask_head,
                _skip_resize=True,
                image_mean=np.repeat(0, in_channels),
                image_std=np.repeat(1, in_channels),
                **framework_kwargs,
            )

        else:
            raise ValueError(f"Framework type '{framework}' is not valid.")
        # some decoders already include a head
        # for these, we pass the num_classes to them
        # others dont include a head
        # for those, we dont pass num_classes
        # model.transform = IdentityTransform()
        model.transform = TerratorchGeneralizedRCNNTransform(
            model.transform.min_size,
            model.transform.max_size,
            model.transform.image_mean,
            model.transform.image_std,
            model.transform.size_divisible,
            model.transform.fixed_size,
            _skip_resize=model.transform._skip_resize,
        )

        return ObjectDetectionModel(model, framework)


class BackboneWrapper(nn.Module):
    def __init__(self, backbone, necks, channel_list):
        """
        BackboneWrapper class that wraps a backbone and necks.

        Args:
            backbone (nn.Module): Backbone module.
            necks (nn.Module): Necks module.
            channel_list (list): List of output channels for each neck.

        Returns:
            dict: Dictionary of output features from necks.
        """
        super().__init__()
        self.backbone = backbone
        self.necks = necks
        self.out_channels = channel_list[-1]
        self.channel_list = channel_list

    def forward(self, x, **kwargs):
        """
        Forward pass of the model.

        Parameters:
        x (torch.Tensor): Input tensor.
        **kwargs (dict): Additional keyword arguments.

        Returns:
        torch.Tensor: Output tensor.
        """
        x = self.backbone(x, **kwargs)
        x = self.necks(x)

        return x


class ObjectDetectionModel(Model):
    def __init__(self, torchvision_model, model_name) -> None:
        """
        Wrapper for torchvision models.

        Args:
            torchvision_model (torchvision.models): A torchvision model.
            model_name (str): The name of the model.

        Returns:
            None
        """

        super().__init__()
        self.torchvision_model = torchvision_model
        self.model_name = model_name

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model.

        Parameters:
        x (torch.Tensor): Input tensor.
        **kwargs (dict): Additional keyword arguments.

        Returns:
        torch.Tensor: Output tensor.
        """
        return ModelOutputObjectDetection(self.torchvision_model(x, *args, **kwargs))

    def freeze_encoder(self):
        """
        Freeze the encoder of the model.
        """
        for param in self.torchvision_model.backbone.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        """
        Freeze the decoder of the model.
        """
        if self.model_name == "faster-rcnn":
            for param in self.torchvision_model.rpn.parameters():
                param.requires_grad = False
                for param in self.torchvision_model.roi_heads.parameters():
                    param.requires_grad = False
        elif self.model_name == "fcos":
            for param in self.torchvision_model.head.parameters():
                param.requires_grad = False
        elif self.model_name == "retinanet":
            for param in self.torchvision_model.head.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Model type '{self.model_name}' is not valid.")


@dataclass
class ModelOutputObjectDetection(ModelOutput):
    output: dict


class ImageContainer:
    def __init__(self, tensor):
        self.tensors = tensor
