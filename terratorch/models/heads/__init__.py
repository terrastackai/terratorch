# Copyright contributors to the Terratorch project

import warnings

from terratorch.models.heads.regression_head import RegressionHead
from terratorch.models.heads.scalar_head import ScalarHead
from terratorch.models.heads.segmentation_head import SegmentationHead


# TODO: Remove in a version v1.3 or later
class ClassificationHead(ScalarHead):
    def __init__(self, *args, **kwargs):
        warnings.warn("ClassificationHead is deprecated. Use ScalarHead instead", DeprecationWarning)
        super().__init__(*args, **kwargs)


__all__ = ["RegressionHead", "ScalarHead", "SegmentationHead"]
