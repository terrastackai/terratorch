from torch import nn


class MRL(nn.Module):
    """
    Matryoshka Representation Learning from the paper: https://arxiv.org/abs/2205.13147
    """

    def __init__(self, features, dolls: list = [16, 32, 64, 128, 256, 768]) -> None:
        super().__init__()
        self.dolls = dolls
        self.layers = nn.ModuleDict()
        for doll in dolls:
            self.layers[f"mrl_{doll}"] = nn.Linear(doll, features)

    def forward(self, x):
        "x: (batch, features)"
        logits = [self.layers[f"mrl_{doll}"](x[:, :doll]) for doll in self.dolls]
        return logits


class MRLLoss(nn.Module):
    def __init__(self, weights) -> None:
        super().__init__()
        self.weights = weights
        self.criterion = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, representations, targets):
        """
        representations: [(batch, features), ...]
        targets: (batch, features)
        """
        losses = [self.weights[i] * (1 - self.criterion(rep, targets)).mean() for i, rep in enumerate(representations)]
        return sum(losses) / len(losses)
