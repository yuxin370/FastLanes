"""Ordinary three-channel classifiers for compact-mosaic training throughput."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from torchvision.models import mobilenet_v2, resnet50
from torchvision.models.vision_transformer import VisionTransformer

MODEL_NAMES = ("resnet50", "mobilenet_v2", "vitti")
MODEL_LABELS = {
    "resnet50": "torchvision.resnet50(weights=None)",
    "mobilenet_v2": "torchvision.mobilenet_v2(weights=None)",
    "vitti": "torchvision.VisionTransformer(ViT-Ti/16, weights=None)",
}


class CompactViTTi(VisionTransformer):
    """Patch16 ViT-Ti with one parameter shape for both 84 and 224 inputs.

    Zero-pad only the right/bottom edge to a complete patch. Interpolate the
    learned 14x14 positional grid for smaller inputs; never resize the image.
    Both operations execute in forward() and belong to measured training time.
    """
    def __init__(self):
        super().__init__(image_size=224, patch_size=16, num_layers=12,
                         num_heads=3, hidden_dim=192, mlp_dim=768,
                         dropout=0., attention_dropout=0., num_classes=1000)

    def forward(self, images):
        height, width = images.shape[-2:]
        pad_h, pad_w = (-height) % self.patch_size, (-width) % self.patch_size
        if pad_h or pad_w:
            images = F.pad(images, (0, pad_w, 0, pad_h))
        patches = self.conv_proj(images)
        grid_h, grid_w = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = torch.cat((self.class_token.expand(len(images), -1, -1), tokens), dim=1)
        position = self.encoder.pos_embedding
        if (grid_h, grid_w) != (14, 14):
            grid = position[:, 1:].reshape(1, 14, 14, self.hidden_dim).permute(0, 3, 1, 2)
            grid = F.interpolate(grid, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
            position = torch.cat((position[:, :1], grid.flatten(2).transpose(1, 2)), dim=1)
        tokens = self.encoder.dropout(tokens + position)
        tokens = self.encoder.layers(tokens)
        tokens = self.encoder.ln(tokens)
        return self.heads(tokens[:, 0])


def build_model(name):
    if name == "resnet50":
        return resnet50(weights=None, num_classes=1000)
    if name == "mobilenet_v2":
        return mobilenet_v2(weights=None, num_classes=1000)
    if name == "vitti":
        return CompactViTTi()
    raise ValueError(f"unknown compact model: {name}")


def input_policy(name, shape):
    channels, height, width = shape
    result = dict(source_shape=list(shape), model_spatial_shape=[height, width], image_resize="none")
    if name == "vitti":
        pad_h, pad_w = (-height) % 16, (-width) % 16
        result.update(model_spatial_shape=[height+pad_h, width+pad_w], patch_size=16,
                      patch_grid=[(height+pad_h)//16, (width+pad_w)//16],
                      sequence_tokens=1+((height+pad_h)//16)*((width+pad_w)//16),
                      padding=dict(right=pad_w, bottom=pad_h, value=0., location="model forward"),
                      position_encoding=dict(learned_base_grid=[14, 14], interpolation="bicubic",
                                             align_corners=False, class_token="unchanged"),
                      padding_and_position_interpolation_in_timed_forward=True)
    return result
