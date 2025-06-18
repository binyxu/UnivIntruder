import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import tqdm
import os
import sys
import numpy as np
import random
import torch.nn.functional as F
import pprint
from data_util import GetDatasetMeta, InMemoryDataset
from model_util import UniversalPerturbation, BackdoorEval, NoTargetDataset
from utils.eval_path import imagenet_models, cifar10_models, cifar100_models
import argparse


def calculate_norm(dataset, trigger, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    norms = []
    for idx, (x, y) in enumerate(dataset):
        x_1 = transform(x)
        x_1_hat = trigger(x_1)
        x_0 = x_1 * 0.5 + 0.5
        x_0_hat = x_1_hat * 0.5 + 0.5
        x_0_hat = x_0_hat.to('cpu')
        norms.append(torch.norm(x_0_hat - x_0, p=float('inf')).item())
        
    mean = np.average(norms)
    var = np.var(norms)
    print(f"Mean l-inf norm: {mean:.4f}, Variance l-inf norm: {var:.4f}")


class ModelLoader:
    def __init__(self, dataset='ImageNet'):
        self.dataset = dataset

    def __len__(self):
        if self.dataset == 'ImageNet':
            return len(imagenet_models)
        elif self.dataset == 'CIFAR10':
            return len(cifar10_models)
        elif self.dataset == 'CIFAR100':
            return len(cifar100_models)
        else:
            raise ValueError("Invalid dataset. Please choose from 'ImageNet', 'CIFAR10', or 'CIFAR100'.")
    
    def get_model_by_index(self, index):
        model = None
        
        if self.dataset == 'ImageNet':
            name = str(imagenet_models[index]).split(' ')[1]
            model = imagenet_models[index](pretrained=True)
        elif self.dataset == 'CIFAR10':
            name = cifar10_models[index]
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=True)
        elif self.dataset == 'CIFAR100':
            name = cifar100_models[index]
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=True)
        else:
            raise ValueError("Invalid dataset. Please choose from 'ImageNet', 'CIFAR10', or 'CIFAR100'.")
        
        if model:
            model.eval()
        
        return model, name



parser = argparse.ArgumentParser(description="UnivIntruder evaluation script with configurable hyperparameters")
parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--target', type=int, default=8, help='Target class index')
parser.add_argument('--eps', type=int, default=32, help='Perturbation budget in 1/255, e.g., 32 for 32/255')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training/evaluation')
parser.add_argument('--image_size', type=int, default=32, help='Input image resolution')
parser.add_argument('--data_path', default='/data/datasets', help='Root directory for datasets')
parser.add_argument('--pood', default='TinyImageNet', help='Public out-of-distribution dataset name')
parser.add_argument('--tgt_dataset', default='CIFAR10', help='Target dataset name')
parser.add_argument('--max_step', type=int, default=3000, help='Maximum training steps')
parser.add_argument('--ckpt', default='', help='Path to trigger checkpoint for evaluation')
parser.add_argument('--split', type=int, default=1, help='Fraction to subsample evaluation set')
args = parser.parse_args()

def main():
    # Parameters
    device = args.device
    target_class = args.target
    epsilon = args.eps/255
    image_size = args.image_size
    ckpt = args.ckpt
    data_path = args.data_path
    tgt_dataset = args.tgt_dataset
    split = args.split

    tgt_data_meta = GetDatasetMeta(data_path, tgt_dataset)
    tgt_transform = tgt_data_meta.get_transformation()
    num_classes = len(tgt_data_meta.get_dataset_label_names())

    a = torch.load(ckpt)
    trigger_model = UniversalPerturbation((3, image_size, image_size), epsilon, initialization=a, device=device)
    trigger_model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        trigger_model, 
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        *tgt_transform.transforms,
    ])

    test_set = tgt_data_meta.get_dataset(transform=transform)
    pure_set = tgt_data_meta.get_dataset(transform=None) 
    calculate_norm(pure_set, trigger_model, image_size)

    test_set = NoTargetDataset(test_set, target_class)
    test_set, _ = torch.utils.data.random_split(test_set, [len(test_set)//split, len(test_set) - len(test_set)//split])
    test_set = InMemoryDataset([(X.detach().to('cpu'), y) for (X, y) in test_set])

    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    models = ModelLoader(tgt_dataset)
    acc_lst = []
    for i in range(len(models)):
        model, name = models.get_model_by_index(i)
        evaluator = BackdoorEval(predictor=model, device=device, target_class=target_class, num_classes=num_classes, target_only=True, top5=True)
        acc = evaluator(test_loader)
        acc['model_name'] = name
        acc_lst.append(acc)
        print(acc)
    return acc_lst

if __name__ == '__main__':
    acc = main()
    pprint.pprint(acc)