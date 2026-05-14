# filters/aesthetic_score.py
from __future__ import annotations

import os
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPModel
from PIL import Image
from typing import Optional

# --- load CLIP once on CPU outside joblib workers ---
CPU_CLIP = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    torch_dtype=torch.float32,
    device_map=None,
).to("cpu")  # type: ignore # force eager load on CPU
CPU_TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
CPU_IMAGE_PROC = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


class AestheticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 1024), nn.GELU(),
            nn.Linear(1024, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 16), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


class AestheticScorer:
    """CLIP-based aesthetic scorer (thread-safe)."""

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Each worker gets its own copy (cloned from CPU)
        self.clip = CPU_CLIP.to(self.device) # type: ignore
        self.tokenizer = CPU_TOKENIZER
        self.image_proc = CPU_IMAGE_PROC

        wp = weights_path or os.path.join(os.path.dirname(__file__), "ava_logos_linearMSE.pth")
        if not os.path.exists(wp):
            raise FileNotFoundError(wp)

        self.head = AestheticHead().to(self.device)
        state = torch.load(wp, map_location=self.device)
        self.head.load_state_dict({k.replace("layers", "net"): v for k, v in state.items()}, strict=True)
        self.head.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score(self, image) -> float:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        pixel = self.image_proc(image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            feat = self.clip.get_image_features(pixel)
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            score = self.head(feat).squeeze().cpu().item()
        return float(max(0.0, min(1.0, score)))

    def _extract_clip_features_batch(self, pil_images):
        pixel = self.image_proc(pil_images, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            feats = self.clip.get_image_features(pixel)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.cpu().numpy()