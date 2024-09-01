import os
import torch
from torchvision.datasets import Caltech101
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from typing import Optional, Callable, Tuple

class CustomCaltech101(Dataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False,
                 test_size: float = 0.2, random_state: int = 42):
        
        # Preload images and labels
        self.dataset = Caltech101(root, download=download)
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform
        
        for idx, (image, target) in enumerate(self.dataset):
            self.data.append(image.convert('RGB'))
            self.targets.append(target)
        
        # Split the data into train and test sets
        self.train = train
        train_data, test_data, train_targets, test_targets = train_test_split(
            self.data, self.targets, test_size=test_size, stratify=self.targets, random_state=random_state
        )
        
        if self.train:
            self.data, self.targets = train_data, train_targets
        else:
            self.data, self.targets = test_data, test_targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.data[index]
        target = self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

# Example usage
if __name__ == "__main__":
    # Define the root directory where the dataset will be stored
    root_dir = "/data/datasets"

    # Define the transform to apply to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create an instance of the custom dataset for training
    train_dataset = CustomCaltech101(root=root_dir, train=True, transform=transform, download=True)

    # Create an instance of the custom dataset for testing
    test_dataset = CustomCaltech101(root=root_dir, train=False, transform=transform, download=True)

    # Print some information
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")