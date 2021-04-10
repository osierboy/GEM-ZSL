import torch
import torch.utils.data as data

import numpy as np
from PIL import Image

class TestDataset(data.Dataset):

    def __init__(self, img_path, labels, transforms=None):
        self.img_path = img_path
        self.labels = torch.tensor(labels).long()
        self.classes = np.unique(labels)

        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.labels[index]

        return img, label

    def __len__(self):
        return self.labels.size(0)