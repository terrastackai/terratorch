import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class GenericObjectDetectionDataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_dir=None,
        transforms=None,
    ):
        """
        image_dir: path to images
        label_dir: path to YOLO-format labels
        transforms: callable(image, target) -> image, target
        return_masks: if True, returns bbox masks [H,W]
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms

        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".tif"))])

        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.images)

    def _load_labels(self, label_path, W, H):
        boxes = []
        labels = []

        if not os.path.exists(label_path):
            return torch.empty((0, 4)), torch.empty((0,), dtype=torch.long)

        with open(label_path) as f:
            for line in f:
                cls, xc, yc, w, h = map(float, line.split())

                x1 = (xc - w / 2) * W
                y1 = (yc - h / 2) * H
                x2 = (xc + w / 2) * W
                y2 = (yc + h / 2) * H

                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))

        return (torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        if self.label_dir is not None:
            label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")
            boxes, labels = self._load_labels(label_path, W, H)
        else:
            boxes = torch.empty((0, 4))
            labels = torch.empty((0,), dtype=torch.long)

        image = self.to_tensor(image)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return {"image": image, **target}
