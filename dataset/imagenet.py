import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable

class ImageNet(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(ImageNet, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []

        with open(os.path.join('./utils/map_clsloc.txt'), 'r') as file:
            lines = file.readlines()

        # Create the dictionary
        self.data_dict = {}

        # Iterate over each line
        for line in lines:
            # Strip any surrounding whitespace and split by comma
            parts = line.strip().split(',')
            # Extract the key and value
            key = parts[0]
            value = [int(parts[1]), parts[2]]
            # Add to the dictionary
            self.data_dict[key] = value

        # Assuming the structure is root/split/images, e.g., root/train/images
        split_dir = os.path.join(self.root, 'imagenet', self.split)
        
        # Load images and labels
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            class_idx = self.data_dict[class_name][0]
            for image_name in os.listdir(class_dir):
                with open(os.path.join(class_dir, image_name), 'rb') as f:
                    img = Image.open(f).convert('RGB')
                self.data.append(img)
                self.targets.append(class_idx)

        if download:
            # Code to download the dataset (if required)
            pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        # Apply transformations
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
