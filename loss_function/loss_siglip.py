import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from torchvision import transforms
import numpy as np
from typing import Union, List
import os
import random

class DirectionLoss(torch.nn.Module):
    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()
        self.loss_type = loss_type
        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        return self.loss_func(x, y)

class SigLIPLoss(torch.nn.Module):
    def __init__(self, device, lambda_direction=1., lambda_naive=0., direction_loss_type='cosine', model_name='google/siglip-base-patch16-224', class_names=None):
        super(SigLIPLoss, self).__init__()
        self.device = device
        
        # 加载 SigLIP 模型和处理器
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()  # 确保模型在评估模式
        
        # 手动图像预处理（保持梯度可计算）
        self.image_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
        ])

        # 初始化变量
        self.target_direction = None
        self.patch_text_directions = None
        self.direction_loss = DirectionLoss(direction_loss_type)
        self.patch_direction_loss = torch.nn.CosineSimilarity(dim=2)
        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)
        self.lambda_naive = lambda_naive
        self.lambda_direction = lambda_direction
        self.src_text_features = None
        self.target_text_features = None
        self.angle_loss = torch.nn.L1Loss()
        self.text_class_features = dict()
        self.image_class_features = dict()
        self.predicted_classes = class_names
        self.mse = torch.nn.MSELoss()

    def tokenize(self, strings: list):
        """将文本字符串转换为 token"""
        return self.processor(text=strings, padding="max_length", return_tensors="pt").to(self.device)

    def encode_text(self, tokens) -> torch.Tensor:
        """编码文本 token 为特征"""
        with torch.no_grad():  # 文本特征通常不需要梯度
            return self.model.get_text_features(**tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像为特征"""
        images = images.to(self.device)
        return self.model.get_image_features(images)  # 注意：这里不使用 no_grad，以便保留梯度

    def get_text_features(self, class_str: Union[str, List[str]]) -> torch.Tensor:
        """获取文本特征，如果未预计算则计算"""
        if isinstance(class_str, str):
            class_str = [class_str]
        text_features = []
        for cls in class_str:
            if cls not in self.text_class_features:
                self.precompute_text_features([cls])
            text_features.append(self.text_class_features[cls])
        return torch.stack(text_features, dim=0)

    def precompute_text_features(self, class_str: List[str], templates=None, norm: bool = True):
        """预计算文本特征"""
        for classes in class_str:
            if classes not in self.text_class_features:
                if templates:
                    template_text = self.compose_text_with_templates(classes, templates)
                else:
                    template_text = [classes]
                tokens = self.tokenize(template_text)
                text_features = self.encode_text(tokens)
                text_features = text_features.mean(dim=0)
                if norm:
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                self.text_class_features[classes] = text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        """获取图像特征"""
        img = img.to(self.device)
        img = self.image_transform(img)  # 手动预处理
        image_features = self.encode_images(img)
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features

    def compute_text_direction(self, source_class: Union[str, List[str]], target_class: Union[str, List[str]], broadcast=False) -> torch.Tensor:
        """计算文本方向"""
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)
        if broadcast:
            text_direction = (target_features.T.unsqueeze(0) - source_features.unsqueeze(-1))
        else:
            text_direction = (target_features - source_features)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        return text_direction

    def compose_text_with_templates(self, text: str, templates=None) -> list:
        """使用模板生成文本"""
        return [template.format(text) for template in templates]

    def clip_directional_loss(self, src_img: torch.Tensor, source_class: Union[str, List[str]], target_img: torch.Tensor, target_class: Union[str, List[str]], negative_class: Union[str, List[str]]) -> torch.Tensor:
        """计算方向损失"""
        self.target_direction = self.compute_text_direction(source_class, target_class)
        self.negative_direction = self.compute_text_direction(source_class, negative_class, broadcast=True)

        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = (target_encoding - src_encoding)
        edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7)

        logit_target = self.cosine_sim(self.target_direction, edit_direction)
        logit_negative = self.cosine_sim(self.negative_direction, edit_direction.unsqueeze(-1))
        pp = torch.exp(logit_target)
        pn = torch.sum(torch.exp(logit_negative), dim=-1)
        p = pp / (pp + pn)
        return -torch.log(p).mean()

    def clip_class_loss(self, target_img: torch.Tensor, target_class: Union[str, List[str]], negative_class: Union[str, List[str]]) -> torch.Tensor:
        """计算分类损失"""
        text_features = self.get_text_features(target_class)
        negative_text_features = self.get_text_features(negative_class)
        image_features = self.get_image_features(target_img)

        logit_target = self.cosine_sim(text_features, image_features)
        logit_negative = self.cosine_sim(negative_text_features.unsqueeze(0).expand(len(target_img), -1, -1).permute(0, 2, 1), image_features.unsqueeze(-1))
        pp = torch.exp(logit_target)
        pn = torch.sum(torch.exp(logit_negative), dim=-1)
        p = pp / (pp + pn)
        return -torch.log(p).mean()

    def forward(self, src_img: torch.Tensor, source_class: Union[str, List[str]], target_img: torch.Tensor, target_class: Union[str, List[str]], negative_class: Union[str, List[str]], texture_image: torch.Tensor = None):
        """前向传播"""
        clip_loss = 0.0

        if self.lambda_naive:
            naive_loss = self.clip_class_loss(target_img, target_class, negative_class)
            clip_loss += self.lambda_naive * naive_loss

        if self.lambda_direction:
            direction_loss = self.clip_directional_loss(src_img, source_class, target_img, target_class, negative_class)
            clip_loss += self.lambda_direction * direction_loss

        return clip_loss

    def predict(self, img: torch.Tensor):
        """预测相似度"""
        image_features = self.get_image_features(img)
        text_features = self.get_text_features(self.predicted_classes)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity

if __name__ == '__main__':
    device = 'cuda'
    siglip_loss = SigLIPLoss(device, class_names=['dog', 'cat', 'boy'])
    src_img = torch.zeros(2, 3, 32, 32, device=device, requires_grad=True)
    tgt_img = torch.zeros(2, 3, 32, 32, device=device, requires_grad=True)
    loss = siglip_loss(src_img, ['dog', 'cat'], tgt_img, ['boy', 'cat'], ['negative'])
    print(f"Loss: {loss.item()}")
    
    # 检查梯度
    loss.backward()
    print(f"Gradient of src_img: {src_img.grad is not None}")
    print(f"Gradient of tgt_img: {tgt_img.grad is not None}")

    src_img = torch.zeros(64, 3, 32, 32, device=device, requires_grad=True)
    tgt_img = torch.zeros(64, 3, 32, 32, device=device, requires_grad=True)
    loss = siglip_loss(src_img, 'dog', tgt_img, 'cat', 'negative')
    print(f"Loss: {loss.item()}")