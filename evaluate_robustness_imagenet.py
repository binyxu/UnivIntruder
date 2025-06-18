import torch
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import pprint
import sys
sys.path.insert(1, '..')
from model_util import UniversalPerturbation, NoTargetDataset
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset, Subset
from dataset import ImageNet
from robustbench.utils import load_model
import argparse
import os


class BackdoorEval():
    def __init__(self, predictor, num_classes, device, target_class, target_only=True, top5=False):
        self.device = device
        self.num_classes = num_classes
        self.target_class = target_class
        self.target_only = target_only
        self.predict = predictor.to(self.device)
        self.predict.eval()
        self.top5 = top5

    def __call__(self, data_loader):
        total_predictions = torch.zeros(self.num_classes)
        top1_correct = 0
        top5_attack_success = 0
        top1_attack_success = 0
        total_samples = 0

        for i, (inputs, labels, *other_info) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.predict(inputs)[:, :self.num_classes]
            preds = torch.argmax(outputs, dim=1).detach().cpu()

            total_predictions += torch.bincount(preds, minlength=self.num_classes)
            total_samples += inputs.size(0)

            # Calculate robust accuracy (accuracy on original labels)
            top1_correct += (preds == labels.detach().cpu()).sum().item()

            top5_preds = torch.topk(outputs, 5, dim=1).indices
            top5_attack_success += sum([self.target_class in top5_preds[i].detach().cpu() for i in range(inputs.size(0))])

            # Calculate attack success rate (predicted as target class when it shouldn't be)
            top1_attack_success += (preds == self.target_class).sum().item()

        if self.target_only:
            # Calculate percentages
            top1_acc = (top1_correct / total_samples) * 100
            top1_attack_success_rate = (top1_attack_success / total_samples) * 100
            top5_attack_success_rate = (top5_attack_success / total_samples) * 100

            return {
                'Top-1 Attack Success Rate': top1_attack_success_rate,
                'Top-5 Attack Success Rate': top5_attack_success_rate,
                'Robust Accuracy': top1_acc,
            }
        else:
            # Normalize counts to percentages for all classes
            total_predictions = (total_predictions / total_predictions.sum()) * 100
            return total_predictions


parser = argparse.ArgumentParser(description="UnivIntruder robustness_eval script with configurable hyperparameters")
parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--target', type=int, default=8, help='Target class index')
parser.add_argument('--eps', type=int, default=32, help='Perturbation budget in 1/255, e.g., 32 for 32/255')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training/evaluation')
parser.add_argument('--image_size', type=int, default=224, help='Input image resolution')
parser.add_argument('--data_path', default='/data/datasets', help='Root directory for datasets')
parser.add_argument('--tgt_dataset', default='ImageNet', help='Target dataset name')
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

    # tgt_transform = [[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]]
    num_classes = 1000

    a = torch.load(ckpt)
    trigger_model = UniversalPerturbation((3, image_size, image_size), epsilon, initialization=a, device=device)
    trigger_model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        trigger_model, 
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        # transforms.Normalize(tgt_transform[0], tgt_transform[1]),
    ])

    test_set = ImageNet(root=args.data_path, split='val', download=False, transform=transform)
    indices = random.sample(range(len(test_set)), 1000)
    test_set = Subset(test_set, indices)


    test_set = NoTargetDataset(test_set, target_class)

    test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

    names = [
        'Debenedetti2022Light_XCiT-S12',
        'Engstrom2019Robustness',
        'Wong2020Fast',
        'Singh2023Revisiting_ConvNeXt-L-ConvStem',
        'Chen2024Data_WRN_50_2',
        'Standard_R50',
        'Liu2023Comprehensive_ConvNeXt-L',
        'Liu2023Comprehensive_ConvNeXt-B',
        'Singh2023Revisiting_ConvNeXt-S-ConvStem',
        'Salman2020Do_R50',
        'Liu2023Comprehensive_Swin-L',
        'Liu2023Comprehensive_Swin-B',
        'Salman2020Do_50_2',
        'Peng2023Robust',
        'Singh2023Revisiting_ViT-B-ConvStem',
        'Singh2023Revisiting_ViT-S-ConvStem',
        'Mo2022When_ViT-B',
        'Bai2024MixedNUTS',
        'Debenedetti2022Light_XCiT-L12',
        'Salman2020Do_R18',
        'Singh2023Revisiting_ConvNeXt-T-ConvStem',
        'Debenedetti2022Light_XCiT-M12',
        'Singh2023Revisiting_ConvNeXt-B-ConvStem',
        'Amini2024MeanSparse',
        'Mo2022When_Swin-B'
    ]
    names.sort()

    accs = []
    for name in names:
        model = load_model(model_name=name,
                    dataset='imagenet',
                    threat_model='Linf')
        evaluator = BackdoorEval(predictor=model, device=device, target_class=target_class, num_classes=num_classes, target_only=True, top5=True)
        acc = evaluator(test_loader)
        print(name, acc['Robust Accuracy'], acc['Top-1 Attack Success Rate'], acc['Top-5 Attack Success Rate'])
        accs.append([name, acc['Robust Accuracy'], acc['Top-1 Attack Success Rate'], acc['Top-5 Attack Success Rate']])
    return accs

if __name__ == '__main__':
    accs = main()
    pprint.pprint(accs)