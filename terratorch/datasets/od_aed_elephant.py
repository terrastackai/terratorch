import os
from typing import Optional

import lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection


class ElephantCocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transform=None):
        super().__init__(img_folder, ann_file)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # convert COCO target to detection-style dict
        boxes = []
        labels = []

        """ print('_+_+_+_+_+')
        print('getimtem') """

        for obj in target:
            x, y, w, h = obj["bbox"]

            # strict filter: positive size only
            if w > 0 and h > 0:
                x2, y2 = x + w, y + h
                if x2 > x and y2 > y:  # extra guard
                    """ print('_+_+_+_+_+')
                    print("appending box")
                    print([x, y, x2, y2]) """
                    boxes.append([x, y, x2, y2])
                    labels.append(obj["category_id"])

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        """ if len(boxes) == 0:
            # IMPORTANT: models like FasterRCNN break if no boxes are returned
            # safest is to skip this sample entirely
            return self.__getitem__((idx + 1) % len(self)) """

        if self.transform:
            img = self.transform(img)

        to_tensor = T.ToTensor()

        sample = {
            "boxes": boxes,
            "labels": labels,
            "image": to_tensor(img),
            "image_id": torch.tensor([self.ids[idx]]),
        }

        return sample
