import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.utils import download_and_extract_archive
import shutil

def normalize_tin_val_folder_structure(path,
                                       images_folder='images',
                                       annotations_file='val_annotations.txt'):
    # Check if files/annotations are still there to see
    # if we already run reorganize the folder structure.
    images_folder = os.path.join(path, images_folder)
    annotations_file = os.path.join(path, annotations_file)

    # Exists
    if not os.path.exists(images_folder) \
       and not os.path.exists(annotations_file):
        if not os.listdir(path):
            raise RuntimeError('Validation folder is empty.')
        return

    # Parse the annotations
    with open(annotations_file) as f:
        for line in f:
            values = line.split()
            img = values[0]
            label = values[1]
            img_file = os.path.join(images_folder, values[0])
            label_folder = os.path.join(path, label)
            os.makedirs(label_folder, exist_ok=True)
            try:
                shutil.move(img_file, os.path.join(label_folder, img))
            except FileNotFoundError:
                continue

    os.sync()
    assert not os.listdir(images_folder)
    shutil.rmtree(images_folder)
    os.remove(annotations_file)
    os.sync()

class TinyImageNet(Dataset):
    """Dataset for TinyImageNet-200"""
    base_folder = 'tiny-imagenet-200'
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    splits = ('train', 'val')
    filename = 'tiny-imagenet-200.zip'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(TinyImageNet, self).__init__()
        self.data_root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        
        if download:
            self.download()

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
        split_dir = os.path.join(self.data_root, 'tiny-imagenet-200', self.split)
        
        # Load images and labels
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name, 'images')
            class_idx = self.data_dict[class_name][0]
            for image_name in os.listdir(class_dir):
                with open(os.path.join(class_dir, image_name), 'rb') as f:
                    img = Image.open(f).convert('RGB')
                self.data.append(img)
                self.targets.append(class_idx)

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
    
    @property
    def dataset_folder(self):
        return os.path.join(self.data_root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.split)
    
    def _check_exists(self):
        return os.path.exists(self.split_folder)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

    def download(self):
        if self._check_exists():
            return
        download_and_extract_archive(
            self.url, self.data_root, filename=self.filename,
            remove_finished=True, md5=self.zip_md5)
        assert 'val' in self.splits
        normalize_tin_val_folder_structure(
            os.path.join(self.dataset_folder, 'val'))