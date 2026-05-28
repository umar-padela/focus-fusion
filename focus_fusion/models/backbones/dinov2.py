import pathlib
import sys

import torch
import torch.nn as nn
from torch import Tensor

# Inject submodule path so `import dinov2` resolves to third_party/dinov2
_THIRD_PARTY = pathlib.Path(__file__).parents[4] / "third_party"
_DINOV2_REPO = _THIRD_PARTY / "dinov2"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2Backbone(nn.Module):
    """Frozen DINOv2 ViT backbone that returns patch embeddings for multi-camera input.

    Default: dinov2_vits14 (ViT-S/14, D_v=384, 22M params).
    Accepts (B, 6, 3, H, W) and returns (B, 6, P, D_v), no CLS token.
    Applies ImageNet normalization internally (normalize_input=True by default).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        img_size: int = 448,
        freeze: bool = True,
        normalize_input: bool = True,
    ) -> None:
        """
        Args:
            model_name: torch.hub model name (dinov2_vits14 / dinov2_vitb14 / etc.)
            img_size: input resolution — must be divisible by 14; default 448 (P=1024)
            freeze: freeze all backbone parameters (should always be True)
            normalize_input: apply ImageNet mean/std normalisation inside forward.
                True by default — assumes raw [0,1] float images from the dataloader.
        """
        super().__init__()
        assert img_size % 14 == 0, f"img_size={img_size} must be divisible by 14 (DINOv2 patch stride)"
        self.img_size = img_size
        self.num_patches = (img_size // 14) ** 2
        self.normalize_input = normalize_input

        if _DINOV2_REPO.exists():
            model = torch.hub.load(
                str(_DINOV2_REPO),
                model_name,
                source="local",
                pretrained=True,
            )
        else:
            # Fallback: download from hub (requires internet; avoid in training runs)
            model = torch.hub.load("facebookresearch/dinov2", model_name)

        if freeze:
            model.requires_grad_(False)

        self.model = model

        if normalize_input:
            mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: (B, 6, 3, H, W)  float32, range [0,1] (ImageNet-normalised unless
                    normalize_input=True was set in constructor)
        Returns:
            patch_embs: (B, 6, P, D_v)  — D_v=384 for ViT-S, 768 for ViT-B; P = (img_size/14)², no CLS token
        """
        B, C, _, H, W = images.shape  # C = num_cameras = 6

        flat = images.view(B * C, 3, H, W)  # (B*6, 3, H, W)

        if self.normalize_input:
            flat = (flat - self.mean) / self.std

        with torch.autocast(device_type=flat.device.type, dtype=torch.float16):
            # get_intermediate_layers returns list of (B*6, P, D_v) tensors
            patch_embs = self.model.get_intermediate_layers(
                flat, n=1, return_class_token=False
            )[0]  # (B*6, P, D_v)

        patch_embs = patch_embs.float()
        return patch_embs.view(B, C, self.num_patches, patch_embs.shape[-1])
