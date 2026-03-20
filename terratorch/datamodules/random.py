import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, task: str, num_samples: int, input_shape, output_shape=None, num_classes: int = 10):
        self.task = task
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.input_shape)

        if self.task == "regression":
            y_shape = self.output_shape or (1,)
            y = torch.randn(y_shape)

        elif self.task == "classification":
            y = torch.randint(0, self.num_classes, (1,)).item()

        elif self.task == "segmentation":
            if self.output_shape is None:
                raise ValueError("Segmentation task requires an `output_shape` (e.g. (H, W))")
            y = torch.randint(0, self.num_classes, self.output_shape)

        elif self.task == "detection":
            # Simulate COCO-style object detection: [x1, y1, x2, y2, class_id]
            num_boxes = torch.randint(1, 5, (1,)).item()
            boxes = torch.rand((num_boxes, 4))
            boxes[:, 2:] += boxes[:, :2]  # ensure x2 > x1 and y2 > y1
            boxes = torch.clamp(boxes, 0, 1)
            labels = torch.randint(0, self.num_classes, (num_boxes,))
            y = {"boxes": boxes, "labels": labels}

        else:
            raise ValueError(f"Unsupported task type: {self.task}")

        output = {"image": x, "label": y}
        return output


class RandomDataModule(LightningDataModule):
    def __init__(
        self,
        task: str = "classification",
        input_shape=(3, 64, 64),
        output_shape=None,
        batch_size=32,
        num_classes=10,
        train_samples=1000,
        val_samples=200,
    ):
        super().__init__()
        self.task = task
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train_samples = train_samples
        self.val_samples = val_samples

    def setup(self, stage=None):
        self.train_dataset = RandomDataset(
            task=self.task,
            num_samples=self.train_samples,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            num_classes=self.num_classes,
        )
        self.val_dataset = RandomDataset(
            task=self.task,
            num_samples=self.val_samples,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            num_classes=self.num_classes,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
