# filters/aesthetic_score.py

import os
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import Optional

class AestheticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 1024), nn.GELU(),
            nn.Linear(1024, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 16), nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AestheticScorer:
    """
    CLIP-based aesthetic scorer with regression head, remapping checkpoint keys.
    """

    def __init__(self, weights_path: Optional[str] = None):
        # 1) Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2) CLIP backbone
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)  # type: ignore
        # 3) CLIP processor
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")  # type: ignore

        # 4) Determine weight file path (always ends up a str)
        if weights_path is None:
            wp = os.path.join(os.path.dirname(__file__), "ava_logos_linearMSE.pth")
        else:
            wp = weights_path

        if not os.path.exists(wp):
            raise FileNotFoundError(f"Aesthetic weights not found: {wp}")

        # 5) Load regression head and remap keys
        self.head = AestheticHead().to(self.device)
        raw_state = torch.load(wp, map_location=self.device)
        mapped = {k.replace("layers", "net"): v for k, v in raw_state.items()}
        self.head.load_state_dict(mapped, strict=True)
        self.head.eval()

    def score(self, image) -> float:
        """
        Returns a normalized aesthetic score in [0,1]. Accepts a PIL.Image or a file path.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.proc(images=image, return_tensors="pt")  # type: ignore
        # Move tensors to device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        with torch.no_grad():
            feats = self.clip.get_image_features(**inputs)  # type: ignore
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            out = self.head(feats).squeeze().cpu().item()

        return float(max(0.0, min(1.0, out)))

    def _extract_clip_features_batch(self, pil_images):
        """
        Batch CLIP feature extraction for a list of PIL.Images.
        """
        inputs = self.proc(images=pil_images, return_tensors="pt")  # type: ignore
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        with torch.no_grad():
            feats = self.clip.get_image_features(**inputs)  # type: ignore
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.cpu().numpy()
