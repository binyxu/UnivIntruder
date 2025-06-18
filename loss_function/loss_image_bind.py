import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from typing import Union, List
import os
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


class DirectionLoss(torch.nn.Module):
    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()
        self.loss_type = loss_type
        self.loss_func = {
            'mse': torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae': torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        return self.loss_func(x, y)


class ImageBindLoss(torch.nn.Module):
    def __init__(self, device, lambda_direction=1., lambda_naive=0., direction_loss_type='cosine', pretrained=True, class_names=None):
        super(ImageBindLoss, self).__init__()
        self.device = device

        # Load ImageBind model
        self.model = imagebind_model.imagebind_huge(pretrained=pretrained)
        self.model.eval()  # Eval mode, but gradients will still flow for adversarial attacks
        self.model.to(device)

        # Loss components (unchanged from CLIPLoss)
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
        """Tokenize text strings using ImageBind's data.load_and_transform_text."""
        return data.load_and_transform_text(strings, self.device)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode tokenized text using ImageBind model, without gradients."""
        inputs = {ModalityType.TEXT: tokens}
        with torch.no_grad():  # Text features are typically fixed, no gradients needed
            embeddings = self.model(inputs)
        return embeddings[ModalityType.TEXT]

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess tensor images from [-1, 1] to ImageBind's expected format."""
        x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        x = TF.resize(x, 224, interpolation=TF.InterpolationMode.BICUBIC)
        x = TF.center_crop(x, 224)
        x = TF.normalize(
            x,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        return x

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using ImageBind model, preserving gradients."""
        images = self.preprocess(images).to(self.device)
        inputs = {ModalityType.VISION: images}
        embeddings = self.model(inputs)  # No torch.no_grad() to allow gradients
        return embeddings[ModalityType.VISION]

    def get_text_features(self, class_str: Union[str, List[str]]) -> torch.Tensor:
        """Retrieve precomputed text features for given classes."""
        if isinstance(class_str, str):
            class_str = [class_str]
        text_features = [self.text_class_features[i] for i in class_str]
        return torch.stack(text_features, dim=0)

    def precompute_text_features(self, class_str: Union[str, List[str]], templates=None, norm: bool = True) -> None:
        """Precompute and store text features for given classes."""
        if isinstance(class_str, str):
            class_str = [class_str]
        for classes in class_str:
            if classes not in self.text_class_features:
                template_text = self.compose_text_with_templates(classes, templates)
                tokens = self.tokenize(template_text)
                text_features = self.encode_text(tokens)
                text_features = text_features.mean(dim=0)
                if norm:  # ImageBind embeddings are already normalized, but kept for consistency
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                self.text_class_features[classes] = text_features

    def precompute_image_features(self, images: dict, norm: bool = True) -> None:
        """Precompute and store image features for given classes."""
        for classes in images:
            if classes not in self.image_class_features:
                class_images = images[classes]
                all_images = class_images
                batch_size = 16
                num_batches = (all_images.size(0) + batch_size - 1) // batch_size

                batched_image_features = []
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, all_images.size(0))
                    batch = all_images[start_idx:end_idx]
                    batch_features = self.encode_images(batch)
                    batched_image_features.append(batch_features.detach())  # Detach to save memory

                image_features = torch.cat(batched_image_features, dim=0)
                image_features = image_features.mean(dim=0)
                if norm:
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                self.image_class_features[classes] = image_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        """Get image features with optional normalization."""
        image_features = self.encode_images(img)
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features

    def compute_text_direction(self, source_class: Union[str, List[str]], target_class: Union[str, List[str]], broadcast=False) -> torch.Tensor:
        """Compute the direction between source and target text features."""
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)
        if broadcast:
            text_direction = (target_features.T.unsqueeze(0) - source_features.unsqueeze(-1))
        else:
            text_direction = (target_features - source_features)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        return text_direction

    def compose_text_with_templates(self, text: str, templates=None) -> list:
        """Compose text with templates (assumes templates are provided)."""
        if templates is None:
            templates = ["{}"]  # Default template if none provided
        return [template.format(text) for template in templates]

    def clip_directional_loss(self, src_img: torch.Tensor, source_class: Union[str, List[str]], target_img: torch.Tensor, target_class: Union[str, List[str]], negative_class: Union[str, List[str]]) -> torch.Tensor:
        """Compute directional loss using ImageBind embeddings."""
        self.target_direction = self.compute_text_direction(source_class, target_class).detach()
        self.negative_direction = self.compute_text_direction(source_class, negative_class, broadcast=True).detach()

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
        """Compute class loss using ImageBind embeddings."""
        text_features = self.get_text_features(target_class)
        negative_text_features = self.get_text_features(negative_class)
        image_features = self.get_image_features(target_img)

        logit_target = self.cosine_sim(text_features, image_features)
        logit_negative = self.cosine_sim(
            negative_text_features.unsqueeze(0).expand(len(target_img), -1, -1).permute(0, 2, 1),
            image_features.unsqueeze(-1)
        )
        pp = torch.exp(logit_target)
        pn = torch.sum(torch.exp(logit_negative), dim=-1)
        p = pp / (pp + pn)
        return -torch.log(p).mean()

    def forward(self, src_img: torch.Tensor, source_class: Union[str, List[str]], target_img: torch.Tensor, target_class: Union[str, List[str]], negative_class: Union[str, List[str]], texture_image: torch.Tensor = None) -> torch.Tensor:
        """Forward pass combining naive and directional losses."""
        clip_loss = 0.0
        # if self.lambda_naive:
        #     naive_loss = self.clip_class_loss(target_img, target_class, negative_class)
        #     clip_loss += self.lambda_naive * naive_loss
        if self.lambda_direction:
            direction_loss = self.clip_directional_loss(src_img, source_class, target_img, target_class, negative_class)
            clip_loss += self.lambda_direction * direction_loss
        return clip_loss

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities using ImageBind embeddings."""
        image_features = self.get_image_features(img)
        text_features = self.get_text_features(self.predicted_classes)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imagebind_loss = ImageBindLoss(device)

    # Test with small batch
    src_img = torch.zeros(2, 3, 32, 32, device=device)
    tgt_img = torch.zeros(2, 3, 32, 32, device=device)
    loss = imagebind_loss(src_img, ['dog', 'cat'], tgt_img, ['boy', 'cat'], ['bird', 'fish'])
    print(f"Loss with batch size 2: {loss.item()}")

    # Test with larger batch
    src_img = torch.zeros(64, 3, 32, 32, device=device)
    tgt_img = torch.zeros(64, 3, 32, 32, device=device)
    loss = imagebind_loss(src_img, 'dog', tgt_img, 'cat', 'bird')
    print(f"Loss with batch size 64: {loss.item()}")