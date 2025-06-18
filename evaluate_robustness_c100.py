import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pprint
from model_util import UniversalPerturbation, NoTargetDataset
from torchvision.datasets import CIFAR10, CIFAR100
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

            # Calculate attack success rate (predicted as target class when it shouldn't be)
            top1_attack_success += (preds == self.target_class).sum().item()

        if self.target_only:
            # Calculate percentages
            top1_acc = (top1_correct / total_samples) * 100
            top1_attack_success_rate = (top1_attack_success / total_samples) * 100

            return {
                'Top-1 Attack Success Rate': top1_attack_success_rate,
                'Robust Accuracy': top1_acc
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
parser.add_argument('--image_size', type=int, default=32, help='Input image resolution')
parser.add_argument('--data_path', default='/data/datasets', help='Root directory for datasets')
parser.add_argument('--tgt_dataset', default='CIFAR100', help='Target dataset name')
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
    num_classes = 100

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

    test_set = CIFAR100(root=args.data_path, train=False, download=False, transform=transform)

    test_set = NoTargetDataset(test_set, target_class)

    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    names = [
        "Rade2021Helper_R18_ddpm",
        ]

    for name in names:
        model = load_model(model_name=name,
                    dataset='cifar100',
                    threat_model='Linf')
        # clean_acc, robust_acc = benchmark(model,
        #                                 dataset='cifar100',
        #                                 threat_model='Linf',
        #                                 eps=8/255)
        # model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
        evaluator = BackdoorEval(predictor=model, device=device, target_class=target_class, num_classes=num_classes, target_only=True, top5=True)
        acc = evaluator(test_loader)
        print(name, acc['Robust Accuracy'], acc['Top-1 Attack Success Rate'])
    # return acc

if __name__ == '__main__':
    acc = main()
    pprint.pprint(acc)