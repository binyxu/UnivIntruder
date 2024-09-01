import os
from torchvision.datasets import CIFAR10, CIFAR100
import torch
import torchvision
from torchvision import transforms
from dataset import TinyImageNet, ImageNet, Caltech101
from utils.text_templates import imagenet_templates

def reverse_dict(label2class):
    # Use dictionary comprehension to reverse the key-value pairs
    class2label = {value: key for key, value in label2class.items()}
    return class2label

class GetDatasetMeta():
    def __init__(self, root, dataset_name) -> None:
        self.dataset_name = dataset_name
        self.root = root

    def get_dataset_label_names(self):
        if self.dataset_name == "CIFAR10":
            tmp = CIFAR10(root=self.root)
            label_dict = reverse_dict(tmp.class_to_idx)

        elif self.dataset_name == "CIFAR100":
            tmp = CIFAR100(root=self.root)
            label_dict = reverse_dict(tmp.class_to_idx)

        elif self.dataset_name == "TinyImageNet" or self.dataset_name == "ImageNet":
            with open(os.path.join('./utils/map_clsloc.txt'), 'r') as file:
                lines = file.readlines()

            label_dict = {}

            for line in lines:
                parts = line.strip().split(',')
                label_dict[int(parts[1])] = parts[2]

        elif self.dataset_name == "Caltech101":
            label_list = self.get_dataset().dataset.categories
            label_dict = {i: label_list[i] for i in range(101)}
            
        else:
            label_dict = None

        return label_dict
    
    def get_transformation(self):
        if self.dataset_name == "CIFAR10":
            size = 32
            normalize = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
        elif self.dataset_name == "CIFAR100":
            size = 32
            normalize = [[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]]
        elif self.dataset_name == "TinyImageNet" or self.dataset_name == "ImageNet":
            size = 224
            # normalize = [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
            normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        elif self.dataset_name == "Caltech101":
            size = 224
            normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        else:
            return None
        preprocess = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.Normalize(mean=normalize[0], std=normalize[1]),
        ])
        return preprocess

    def get_template(self):
        return imagenet_templates

    def get_dataset(self, train=False, download=False, transform=None, **kwargs):
        if self.dataset_name == "CIFAR10":
            target_dataset = CIFAR10(root=self.root, train=train, download=download, transform=transform)
        elif self.dataset_name == "CIFAR100":
            target_dataset = CIFAR100(root=self.root, train=train, download=download, transform=transform)
        elif self.dataset_name == "TinyImageNet":
            target_dataset = TinyImageNet(root=self.root, split='train' if train else 'test', download=download, transform=transform)
        elif self.dataset_name == "ImageNet":
            target_dataset = ImageNet(root=self.root, split='train' if train else 'val', download=download, transform=transform)
        elif self.dataset_name == "Caltech101":
            target_dataset = Caltech101(root=self.root, transform=transform, train=train)
        return target_dataset
    
    def get_clean_model(self):
        if self.dataset_name == "CIFAR10":
            model_visual = torch.hub.load("chenyaofo/pytorch-cifar-models", 'cifar10_resnet44', pretrained=True)
        elif self.dataset_name == "CIFAR100":
            model_visual = torch.hub.load("chenyaofo/pytorch-cifar-models", 'cifar100_resnet44', pretrained=True)
        elif self.dataset_name == "TinyImageNet" or self.dataset_name == "ImageNet":
            model_visual = torchvision.models.resnet50(pretrained=True)
        return model_visual
    
    def n_classes(self):
        if self.dataset_name == "CIFAR10":
            n = 10
        elif self.dataset_name == "CIFAR100":
            n = 100
        elif self.dataset_name == "TinyImageNet":
            n = 500
        elif self.dataset_name == "ImageNet":
            n = 1000
        elif self.dataset_name == "Caltech101":
            n = 101
        return n
    

class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, transform=None):
        self.original_dataset = original_dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.original_dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.original_dataset)