import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import terratorch.visualization as ttv


class CopyPasteObjectDetectionDataset(Dataset):
    def __init__(
        self,
        base_dataset,  # yields (image, boxes)
        object_folder,  # folder with PNGs (RGBA)
        paste_prob=0.7,
        scale_range=(0.5, 1.5),
        max_objects=3,
        image_size=None,  # (H, W) or None
    ):
        self.base_dataset = base_dataset
        self.object_paths = [
            os.path.join(object_folder, f) for f in os.listdir(object_folder) if f.lower().endswith(".png")
        ]
        assert len(self.object_paths) > 0

        self.paste_prob = paste_prob
        self.scale_range = scale_range
        self.max_objects = max_objects
        self.image_size = image_size

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def _load_object(self):
        obj = Image.open(random.choice(self.object_paths)).convert("RGBA")
        return obj

    def _paste_object(self, image, mask, obj):
        H, W = image.shape[-2:]

        scale = random.uniform(*self.scale_range)
        ow, oh = obj.size
        ow, oh = int(ow * scale), int(oh * scale)

        if ow >= W or oh >= H:
            return image, mask

        obj = obj.resize((ow, oh), Image.BILINEAR)
        ox = random.randint(0, W - ow)
        oy = random.randint(0, H - oh)

        obj_np = np.array(obj)
        rgb = torch.from_numpy(obj_np[..., :3]).permute(2, 0, 1) / 255.0
        alpha = torch.from_numpy(obj_np[..., 3] / 255.0)

        image[:, oy : oy + oh, ox : ox + ow] = image[:, oy : oy + oh, ox : ox + ow] * (1 - alpha) + rgb * alpha

        mask[oy : oy + oh, ox : ox + ow] |= alpha > 0.5

        bbox = torch.tensor([ox, oy, ox + ow, oy + oh], dtype=torch.float32)
        return image, mask, bbox

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        # Handle both dict and tuple returns
        if isinstance(item, dict):
            image = item["image"]
            boxes = item["boxes"]
            labels = item.get("labels", None)
        elif len(item) == 2:
            image, boxes = item
            labels = None
        else:
            image, boxes, labels = item[:3]

        if isinstance(image, Image.Image):
            image = self.to_tensor(image)

        if self.image_size is not None:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False
            ).squeeze(0)

        H, W = image.shape[-2:]
        mask = torch.zeros((H, W), dtype=torch.uint8)

        # mark original boxes
        for box in boxes:
            x1, y1, x2, y2 = box.int()
            mask[y1:y2, x1:x2] = 1

        if random.random() < self.paste_prob:
            n = random.randint(1, self.max_objects)
            for _ in range(n):
                obj = self._load_object()
                image, mask, bbox = self._paste_object(image, mask, obj)

                boxes = torch.cat([boxes, bbox.unsqueeze(0)], dim=0)

                # assuming class 1 for pasted objects
                if labels is not None:
                    labels = torch.cat([labels, torch.tensor([1], dtype=torch.long)], dim=0)

        return {"image": image.float(), "boxes": boxes, "mask": mask, "labels": labels}

    def plot(
        self,
        sample: dict[str, torch.Tensor],
        suptitle: str | None = None,
    ) -> plt.Figure:
        """Plot a detection sample using terratorch helpers (TensorBoard-safe)."""

        fig = ttv.plot_boxes_labels(
            image=sample["image"],
            boxes=sample["boxes"],
            show=False,  # REQUIRED for TensorBoard
        )

        if suptitle:
            fig.suptitle(suptitle)

        return fig
