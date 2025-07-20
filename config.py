# config.py
# Cross-platform, env-overridable, zero-placeholder.

import os
import platform
import torch

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
INSIGHTFACE_MODELS_DIR = os.path.join(DATA_DIR, "insightface_models")
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", os.path.join(BASE_DIR, "output"))

# --- Image loader ---
IMAGE_MAX_SIZE = (
    int(os.environ.get("MAX_IMAGE_WIDTH", 1024)),
    int(os.environ.get("MAX_IMAGE_HEIGHT", 1024)),
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# --- Pipeline thresholds (tunable per env) ---
BLUR_THRESHOLD      = float(os.getenv("BLUR_THRESHOLD", 30.0))
BLUR_FATAL          = float(os.getenv("BLUR_FATAL", 5.0))
EAR_THRESHOLD       = float(os.getenv("EAR_THRESHOLD", 0.35))
AESTHETIC_THRESHOLD = float(os.getenv("AESTHETIC_THRESHOLD", 0.35))
VIP_COSINE_THRESH   = float(os.getenv("VIP_COSINE_THRESH", 0.75))
DUPLICATE_THRESH    = float(os.getenv("DUPLICATE_THRESH", 0.30))
EXP_PCTL_LOW        = float(os.getenv("EXP_PCTL_LOW", 5.0))
EXP_PCTL_HIGH       = float(os.getenv("EXP_PCTL_HIGH", 95.0))

# --- Batch & cache ---
CLIP_BATCH_SIZE = int(os.getenv("CLIP_BATCH_SIZE", 32))

# --- Model paths ---
YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    os.path.join(BASE_DIR, "yolov8n.pt")
)
AESTHETIC_WEIGHTS = os.environ.get(
    "AESTHETIC_WEIGHTS",
    os.path.join(BASE_DIR, "filters", "ava_logos_linearMSE.pth")
)

# --- System auto-detect ---
CPU_CORES = os.cpu_count() or 1
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM    = platform.system()