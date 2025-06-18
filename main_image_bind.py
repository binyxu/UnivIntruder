import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import random
from PIL import Image
import torch.nn.functional as F
from data_util import GetDatasetMeta, TransformedDataset, InMemoryDataset
from model_util import TrainableAffineTransform, UniversalPerturbation, BackdoorEval, NoTargetDataset
from loss_function.loss_image_bind import ImageBindLoss
import argparse


# Fix random seeds for reproducibility
def fix_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
def save_images(x, x_hat, render_num=64, output_dir='rendered_images', step=0):
    os.makedirs(output_dir, exist_ok=True)

    num_rows = int(render_num ** 0.5 / 2) * 2

    img_lst = []
    for i in range(int(render_num / 2)):
        img_lst.append(x[i])
        img_lst.append(x_hat[i])
    
    grid = torchvision.utils.make_grid(img_lst, nrow=num_rows, padding=2)
    
    torchvision.utils.save_image(grid * 0.5 + 0.5, os.path.join(output_dir, str(step) + '.png'), nrow=num_rows)



parser = argparse.ArgumentParser(description="UnivIntruder training script with configurable hyperparameters")
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
    fix_random_seeds(0)

    # Parameters
    device = args.device
    target_class = args.target
    eps = args.eps
    epsilon = args.eps/255
    batch_size = args.batch_size
    save_every_step = 200
    image_size = args.image_size
    data_path = args.data_path
    src_dataset = args.pood
    tgt_dataset = args.tgt_dataset
    max_step = args.max_step
    out_path = f'./experiments/{tgt_dataset}/epsilon_{eps}_target_{target_class}'
    pretrain = None
    flip = True
    download = True
    simple_out = False
    top5 = True
    split = args.split

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    src_data_meta = GetDatasetMeta(data_path, src_dataset)
    tgt_data_meta = GetDatasetMeta(data_path, tgt_dataset)

    train_set_un = src_data_meta.get_dataset(transform=transform, train=True if src_dataset != 'ImageNet' else False, download=download)
    test_set = tgt_data_meta.get_dataset(transform=transform, download=download)
    test_set = NoTargetDataset(test_set, target_class)
    test_set, _ = torch.utils.data.random_split(test_set, [len(test_set)//split, len(test_set) - len(test_set)//split])
    src_label_text_dict = src_data_meta.get_dataset_label_names()
    tgt_label_text_dict = tgt_data_meta.get_dataset_label_names()

    tgt_transform = tgt_data_meta.get_transformation()
    y_negative = set(tgt_label_text_dict.values())
    y_negative.discard(tgt_label_text_dict[target_class])

    train_set = NoTargetDataset(train_set_un, target = next((k for k, v in src_label_text_dict.items() if v == tgt_label_text_dict[target_class]), None))
    train_set = InMemoryDataset([i for i in train_set])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)

    if pretrain:
        input_tensor = torch.load(pretrain)
        output_tensor = F.interpolate(input_tensor.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0)
        trigger_model = UniversalPerturbation((3, image_size, image_size), epsilon, initialization=output_tensor, device=device)
    else:
        trigger_model = UniversalPerturbation((3, image_size, image_size), epsilon, initialization=None, device=device)

    visual_model = tgt_data_meta.get_clean_model()
    evaluator = BackdoorEval(visual_model, len(list(tgt_label_text_dict.keys())), device, target_class, simple_out, top5)

    # Optimizer
    optimizer = optim.Adam(trigger_model.parameters(), lr=0.01, weight_decay=1e-5)

    # Loss function
    clip_loss_func = ImageBindLoss(
        device,
        lambda_direction=1-naive,
        lambda_naive=naive,
    )
    
    clip_loss_func.precompute_text_features(list(src_label_text_dict.values()), templates=src_data_meta.get_template())
    clip_loss_func.precompute_text_features(list(tgt_label_text_dict.values()), templates=tgt_data_meta.get_template())

    # Training loop
    global_step = 0
    while global_step <= max_step:
        average_epoch_loss = []
        for x, y in train_loader:
            if global_step > max_step:
                    break
            bs = x.size(0)
            x = x.to(device).to(torch.float32)
            x_hat = trigger_model(x)  # Apply the perturbation

            grad_transform = TrainableAffineTransform(bs, 0.25, flip=flip)

            x_hat_trans = grad_transform(x_hat)
            
            if src_label_text_dict:
                y_source = [src_label_text_dict[int(y_i)] for y_i in y]
            else:
                y_source = None
            y_target = [tgt_label_text_dict[int(target_class)] for _ in y]

            # Compute the loss
            loss = clip_loss_func(x, y_source, x_hat_trans, y_target, y_negative)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                trigger_model.trigger.clamp_(-1, 1)

            average_epoch_loss.append(loss.item())
            global_step += 1
            
            if global_step % save_every_step == 0:
                save_images(x, x_hat_trans, render_num=batch_size, output_dir=os.path.join(out_path, 'log_images'), step=global_step)  # Save first 5 images from the last batch

                trigger_model.eval()
                trigger_model = trigger_model.to('cpu')
                transform = transforms.Compose([
                    trigger_model, 
                    transforms.Normalize((-1, -1, -1), (2, 2, 2)),
                    *tgt_transform.transforms,
                ])
                current_test_set = TransformedDataset(test_set, transform)
                test_loader = DataLoader(current_test_set, batch_size=32, shuffle=True)
                acc = evaluator(test_loader)
                
                if simple_out:
                    out_str = f'Step {global_step}, Loss: {sum(average_epoch_loss)/len(average_epoch_loss):.4f}, acc: {acc:.4f}'
                else:
                    out_str = ''
                    if top5:
                        out_str = f"Top-1 Accuracy: {acc['Top-1 Accuracy']:.4f}, Top-5 Accuracy: {acc['Top-5 Accuracy']:.4f} "
                        acc = acc['Class Percentages']
                    top_three_classes = sorted(acc.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_classes_str = ", ".join([f'{cls}: {pct:.2f}%' for cls, pct in top_three_classes])
                    out_str += f'Step {global_step}, Loss: {sum(average_epoch_loss)/len(average_epoch_loss):.4f}, Top 3 Accuracies: {top_classes_str}'
                with open(os.path.join(out_path, 'log.txt'), 'a') as file:
                    file.write(out_str + '\n')
                print(out_str)

                a = trigger_model.trigger.detach().cpu()
                os.makedirs(os.path.join(out_path, 'ckpts'), exist_ok=True)
                torch.save(a, os.path.join(out_path, f'ckpts/clip_trigger_{global_step}.pth'))

                trigger_model = trigger_model.to(device)
                trigger_model.train()


if __name__ == '__main__':
    main()
