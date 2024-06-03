import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .data import load_and_transform_text
from .models import imagebind_model
from .models.imagebind_model import ModalityType


class ImageBindEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    def __init__(self, pretrained="models/imagebind_huge.pth"):
        super().__init__()
        self.imagebind_model = imagebind_model.imagebind_huge(
            pretrained=pretrained)
        self.imagebind_model.eval()

        for p in self.imagebind_model.parameters():
            p.requires_grad_(False)

    def encode_image(self, x):
        """
        Args:
            x: lies in [0, 1] space.
        """
        def transform_data(image):
            image = F.interpolate(image, 224, mode="bicubic")
            mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).type_as(image)
            std = torch.tensor((0.26862954, 0.26130258, 0.27577711)).type_as(image)
            mean = rearrange(mean, "f -> 1 f 1 1")
            std = rearrange(std, "f -> 1 f 1 1")
            image = (image - mean) / std
            return image.to(next(self.parameters()).device)
        inputs = {ModalityType.VISION: transform_data(x)}
        with torch.no_grad():
            embeddings = self.imagebind_model(inputs)
        return embeddings[ModalityType.VISION]
    
    def encode_text(self, x):
        inputs = {ModalityType.TEXT: load_and_transform_text(x, next(self.parameters()).device)}
        with torch.no_grad():
            embeddings = self.imagebind_model(inputs)
        return embeddings[ModalityType.TEXT]
    
    def encode(self, image, text):
        """
        Args:
            image: lies in [0, 1] space.
            x: a list of strings
        Returns:
            batch size, 2, 1024
        """
        return torch.stack([self.encode_image(image), self.encode_text(text)], dim=1)
    
if __name__ == "__main__":
    image_bind_embedder = ImageBindEmbedder().cuda()
    image = torch.randn(4, 3, 256, 256)
    text = ["aaa", "bbb"]
    image_emb = image_bind_embedder.encode_image(image)
    text_emb = image_bind_embedder.encode_text(text)
    
    print(image_emb.shape)
    print(text_emb.shape)