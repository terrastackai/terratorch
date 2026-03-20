# Copyright contributors to the Terratorch project

import warnings

from torch import Tensor, nn


class ScalarHead(nn.Module):
    """Classification and Scalar Regression head"""

    # how to allow cls token?
    def __init__(
        self,
        in_dim: int,
        num_outputs: int | None = None,
        num_classes: int | None = None,
        dim_list: list[int] | None = None,
        dropout: float = 0,
        linear_after_pool: bool = False,
    ) -> None:
        """Constructor

        Args:
            in_dim (int): Input dimensionality
            num_outputs (int, optional): Number of predicted variables for regression.
            num_classes (int, optional): Number of output classes for classification
            dim_list (list[int] | None, optional):  List with number of dimensions for each Linear
                layer to be created. Defaults to None.
            dropout (float, optional): Dropout value to apply. Defaults to 0.
            linear_after_pool (bool, optional): Apply pooling first, then apply the linear layer. Defaults to False
        """
        super().__init__()
        if num_outputs is None and num_classes is None:
            raise ValueError("`num_outputs` or `num_classes` should be provided.")
        if num_outputs is not None and num_classes is not None:
            msg = "Both `num_outputs` and `num_classes` were provided, using `num_outputs`."
            warnings.warn(msg)

        self.num_classes = num_classes
        self.num_outputs = num_outputs or num_classes
        self.linear_after_pool = linear_after_pool
        if dim_list is None:
            pre_head = nn.Identity()
        else:

            def block(in_dim, out_dim):
                return nn.Sequential(nn.Linear(in_features=in_dim, out_features=out_dim), nn.ReLU())

            dim_list = [in_dim, *dim_list]
            pre_head = nn.Sequential(*[block(dim_list[i], dim_list[i + 1]) for i in range(len(dim_list) - 1)])
            in_dim = dim_list[-1]
        dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.head = nn.Sequential(
            pre_head,
            dropout,
            nn.Linear(in_features=in_dim, out_features=self.num_outputs),
        )

    def forward(self, x: Tensor):
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        if self.linear_after_pool:
            x = x.mean(axis=1)
            out = self.head(x)
        else:
            x = self.head(x)
            out = x.mean(axis=1)
        return out
