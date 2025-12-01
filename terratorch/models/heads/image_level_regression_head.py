from torch import Tensor, nn


class ImageLevelRegressionHead(nn.Module):
    """Image-level regression head.

    It is analogous to ClassificationHead, but instead of predicting the number of classes,
    it returns continuous values of dimension ``out_dim``.

    Args:
        in_dim (int): Number of feature channels in the input (C).
        out_dim (int): Size of the output regression vector. Default is 1.
        dim_list (list[int] | None, optional): List of sizes for linear layers
            to be added before the output layer. If None, no intermediate layers
            are added. Default is None.
        dropout (float, optional): Dropout rate. Default is 0.
        linear_after_pool (bool, optional): Whether to first perform pooling (averaging)
            over the H*W dimension, and then pass through the linear layers.
            If False, we first pass through the layers, and only at the end
            average the results. Default is False.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        dim_list: list[int] | None = None,
        dropout: float = 0.0,
        linear_after_pool: bool = False,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.linear_after_pool = linear_after_pool


        if dim_list is None:
            pre_head = nn.Identity()
        else:
            def block(in_features: int, out_features: int) -> nn.Module:
                return nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=out_features),
                    nn.ReLU()
                )

            dim_list = [in_dim] + dim_list
            layers = []
            for i in range(len(dim_list) - 1):
                layers.append(block(dim_list[i], dim_list[i + 1]))
            pre_head = nn.Sequential(*layers)
            in_dim = dim_list[-1]  # update in_dim to the size of the last layer

        dropout_layer = nn.Identity() if dropout == 0 else nn.Dropout(dropout)

        self.head = nn.Sequential(
            pre_head,
            dropout_layer,
            nn.Linear(in_features=in_dim, out_features=out_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Tensor of shape [B, out_dim] – image-level regression vector(s).
        """

        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        if self.linear_after_pool:

            x = x.mean(dim=1)
            out = self.head(x)
        else:
            x = self.head(x)
            out = x.mean(dim=1)

        return out
