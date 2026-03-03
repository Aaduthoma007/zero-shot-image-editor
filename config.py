"""
Configuration for the Zero-Shot Image Editing Assistant.

Design Decision:
- GPU-first with float16 for speed + lower VRAM usage
- Attention slicing enabled to prevent GPU memory overflow / laptop overheating
- CPU fallback with float32 for systems without CUDA
"""

import os
import torch

# ─── Device Detection ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ─── Model Settings ──────────────────────────────────────────────────
MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models")

# LoRA adapter (transfer learning) – set to None to disable
# Example community LoRA: "stabilityai/stable-diffusion-xl-base-1.0"
LORA_WEIGHTS_ID = None  # e.g. "lora-library/anime-style" or a local path
LORA_SCALE = 0.7  # How strongly the LoRA adapter influences the output

# ─── Pipeline Defaults ───────────────────────────────────────────────
DEFAULT_STRENGTH = 0.55        # How much to transform (0 = no change, 1 = full)
DEFAULT_GUIDANCE_SCALE = 7.5   # How closely to follow the prompt
DEFAULT_NUM_STEPS = 30         # Inference steps (lower = faster, less quality)
DEFAULT_IMAGE_SIZE = 512       # Resize input to this (saves memory)

# ─── Memory Optimization Flags ───────────────────────────────────────
ENABLE_ATTENTION_SLICING = True   # Slices attention to reduce peak VRAM
ENABLE_VAE_SLICING = True         # Slices VAE decoding for less memory
ENABLE_CPU_OFFLOAD = False        # Offload model layers to CPU when not in use

# ─── Flask Settings ──────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "static", "outputs")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload

# ─── Logging ─────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
