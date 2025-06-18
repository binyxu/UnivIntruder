import torch
import torchvision.transforms as transforms
import numpy as np
from typing import Union, List
import os

import open_clip
import random
from PIL import Image


class CLIPLoss(torch.nn.Module):
    def __init__(self, device, lambda_direction=1., direction_loss_type='cosine', clip_model='ViT-B/32', pretrained=None, class_names=None):
        super(CLIPLoss, self).__init__()

        self.device = device
        self.model, _, clip_preprocess = open_clip.create_model_and_transforms(
            clip_model, 
            pretrained=pretrained,
            jit=True,
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model)

        self.clip_preprocess = clip_preprocess

        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

        self.target_direction      = None
        self.patch_text_directions = None

        self.patch_direction_loss = torch.nn.CosineSimilarity(dim=2)
        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)

        self.lambda_direction  = lambda_direction

        self.src_text_features = None
        self.target_text_features = None
        self.angle_loss = torch.nn.L1Loss()

        self.text_class_features = dict()
        self.image_class_features = dict()
        self.predicted_classes = class_names
        self.mse = torch.nn.MSELoss()

    def tokenize(self, strings: list):
        return self.tokenizer(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)
    
    def get_text_features(self, class_str: str) -> torch.Tensor:
        text_features = [self.text_class_features[i] for i in class_str]
        return torch.stack(text_features, dim=0)
    
    def precompute_text_features(self, class_str: str, templates=None, norm: bool = True) -> torch.Tensor:
        for classes in class_str:
            if classes not in self.text_class_features:
                template_text = self.compose_text_with_templates(classes, templates)
                tokens = self.tokenizer(template_text).to(self.device)
                text_features = self.encode_text(tokens).detach()
                text_features = text_features.mean(dim=0)
                if norm:
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                self.text_class_features[classes] = text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def compute_text_direction(self, source_class: Union[str, List[str]], target_class: Union[str, List[str]], broadcast=False) -> torch.Tensor:
        source_features = self.get_text_features(source_class) 
        target_features = self.get_text_features(target_class) 

        if broadcast:
            text_direction = (target_features.T.unsqueeze(0) - source_features.unsqueeze(-1))
        else:
            text_direction = (target_features - source_features)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction

    def compose_text_with_templates(self, text: str, templates=None) -> list:
        return [template.format(text) for template in templates]
            
    def clip_directional_loss(self, src_img: torch.Tensor, source_class: Union[str, List[str]], target_img: torch.Tensor, target_class: Union[str, List[str]], negative_class: Union[str, List[str]]) -> torch.Tensor:
        self.target_direction = self.compute_text_direction(source_class, target_class)
        self.negative_direction = self.compute_text_direction(source_class, negative_class, broadcast=True)

        src_encoding    = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = (target_encoding - src_encoding)
        edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7)

        logit_target = self.cosine_sim(self.target_direction, edit_direction)
        logit_negative = self.cosine_sim(self.negative_direction, edit_direction.unsqueeze(-1))
        pp = torch.exp(logit_target)
        pn = torch.sum(torch.exp(logit_negative), dim=-1)
        p = pp/(pp+pn)
        return -torch.log(p).mean()

    
    def forward(self, src_img: torch.Tensor, source_class: Union[str, List[str]], target_img: torch.Tensor, target_class: Union[str, List[str]], negative_class: Union[str, List[str]], texture_image: torch.Tensor = None):
        clip_loss = 0.0

        if self.lambda_direction:
            direction_loss = self.clip_directional_loss(src_img, source_class, target_img, target_class, negative_class)
            clip_loss += self.lambda_direction * direction_loss

        return clip_loss
    

if __name__ == '__main__':
    device = 'cuda'
    clip_loss = CLIPLoss(device)
    src_img = torch.zeros(2, 3, 32, 32, device=device)
    tgt_img = torch.zeros(2, 3, 32, 32, device=device)
    a = clip_loss(src_img, ['dog', 'cat'], tgt_img, ['boy', 'cat'])

    src_img = torch.zeros(64, 3, 32, 32, device=device)
    tgt_img = torch.zeros(64, 3, 32, 32, device=device)
    a = clip_loss(src_img, 'dog', tgt_img, 'cat')