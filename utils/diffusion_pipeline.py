"""
Diffusion Pipeline — Stable Diffusion img2img with LoRA (Transfer Learning)

Architecture:
  Base Model:  runwayml/stable-diffusion-v1-5 (pre-trained on LAION-5B)
  Transfer:    LoRA adapters loaded on top of frozen base model
  Pipeline:    StableDiffusionImg2ImgPipeline from HuggingFace Diffusers

Memory Safety:
  - Uses float16 on CUDA to halve VRAM usage (~4 GB instead of ~8 GB)
  - Attention slicing splits attention computation into chunks
  - VAE slicing decodes latents in slices to avoid OOM
  - These prevent laptop GPU from overheating
"""

import logging
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

logger = logging.getLogger(__name__)


class ImageEditor:
    """
    Wraps HuggingFace Diffusers' StableDiffusionImg2ImgPipeline.

    Transfer Learning approach:
        1. Load a pre-trained Stable Diffusion model (no training from scratch)
        2. Optionally load LoRA weights — small adapter matrices injected into
           the cross-attention layers of the UNet.  This is the transfer learning
           mechanism: the base model stays frozen, and the LoRA weights encode
           domain-specific style knowledge learned from a much smaller dataset.
        3. At inference time, the LoRA adapter modulates the base model's
           behaviour, enabling stylistic control without fine-tuning the
           full ~1 B parameter model.
    """

    def __init__(self):
        self.pipe = None
        self.device = None
        self.lora_loaded = False

    # ── Model Loading ─────────────────────────────────────────────────

    def load_model(self):
        """
        Load the Stable Diffusion img2img pipeline onto GPU (preferred) or CPU.

        This does NOT train anything — it loads pre-trained weights, which is
        the foundation of transfer learning: reusing knowledge learned on a
        massive dataset (LAION-5B, ~5 billion image-text pairs).
        """
        from config import (
            MODEL_ID, MODEL_CACHE_DIR, DEVICE, DTYPE,
            ENABLE_ATTENTION_SLICING, ENABLE_VAE_SLICING,
            ENABLE_CPU_OFFLOAD, LORA_WEIGHTS_ID, LORA_SCALE,
        )

        self.device = DEVICE
        logger.info(f"Loading model '{MODEL_ID}' on {self.device} with dtype={DTYPE}")

        # ── Load pre-trained base model ──────────────────────────────
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            cache_dir=MODEL_CACHE_DIR,
            safety_checker=None,           # Disable NSFW filter for speed
            requires_safety_checker=False,
        )

        # ── Move to device ───────────────────────────────────────────
        if ENABLE_CPU_OFFLOAD and self.device == "cuda":
            # Keeps model layers on CPU, moves them to GPU only when needed
            self.pipe.enable_model_cpu_offload()
            logger.info("CPU offload enabled — layers move to GPU on demand")
        else:
            self.pipe = self.pipe.to(self.device)
            logger.info(f"Pipeline moved to {self.device}")

        # ── Memory optimizations (prevent laptop overheating) ────────
        if ENABLE_ATTENTION_SLICING:
            self.pipe.enable_attention_slicing(slice_size="auto")
            logger.info("Attention slicing enabled — reduces peak VRAM")

        if ENABLE_VAE_SLICING:
            self.pipe.enable_vae_slicing()
            logger.info("VAE slicing enabled — decodes latents in chunks")

        # ── Transfer Learning: Load LoRA adapter ─────────────────────
        if LORA_WEIGHTS_ID:
            self._load_lora(LORA_WEIGHTS_ID, LORA_SCALE)

        logger.info("✅ Model loaded and ready for inference")

    def _load_lora(self, lora_id: str, scale: float):
        """
        Load LoRA (Low-Rank Adaptation) weights on top of the base model.

        How LoRA works (transfer learning):
            - The base model's attention weight matrices W are frozen.
            - Two small matrices A (down-projection) and B (up-projection)
              are added:  W' = W + scale * (B @ A)
            - Only A and B are stored (a few MB vs. GB for the full model).
            - This lets us adapt the model's style/domain without retraining.

        Args:
            lora_id:  HuggingFace Hub ID or local path to LoRA weights.
            scale:    Multiplier for how strongly the LoRA influences output.
        """
        try:
            logger.info(f"Loading LoRA adapter: {lora_id} (scale={scale})")
            self.pipe.load_lora_weights(lora_id)
            self.pipe.fuse_lora(lora_scale=scale)
            self.lora_loaded = True
            logger.info("✅ LoRA adapter loaded — transfer learning active")
        except Exception as e:
            logger.warning(f"⚠️ Could not load LoRA weights: {e}. Continuing without LoRA.")
            self.lora_loaded = False

    # ── Image Editing ─────────────────────────────────────────────────

    def edit_image(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.55,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
    ) -> Image.Image:
        """
        Perform img2img editing using the loaded pipeline.

        How img2img works:
            1. Encode the input image into the latent space using the VAE encoder.
            2. Add noise to the latent proportional to `strength`
               (strength=0.5 means 50% noise, so 50% of the original is preserved).
            3. Denoise the latent guided by the text prompt using the UNet.
            4. Decode the denoised latent back to pixel space via the VAE decoder.

        Args:
            image:              Input PIL image.
            prompt:             Text description of the desired edit.
            negative_prompt:    Things to avoid (e.g., "blurry, distorted").
            strength:           0.0 = no change, 1.0 = complete reimagination.
            guidance_scale:     Prompt adherence (higher = follows prompt more).
            num_inference_steps: Denoising iterations (more = better quality).

        Returns:
            Edited PIL image.
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from config import DEFAULT_IMAGE_SIZE

        # Resize to standard size for consistent VRAM usage
        image = self._prepare_image(image, DEFAULT_IMAGE_SIZE)

        logger.info(
            f"Generating edit  |  strength={strength}  |  guidance={guidance_scale}  "
            f"|  steps={num_inference_steps}  |  device={self.device}"
        )

        # ── Run the img2img pipeline ─────────────────────────────────
        with torch.no_grad():  # No gradient computation (inference only)
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "blurry, distorted, low quality, deformed",
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )

        output_image = result.images[0]
        logger.info("✅ Image generated successfully")
        return output_image

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _prepare_image(image: Image.Image, target_size: int) -> Image.Image:
        """Resize image to target_size while maintaining aspect ratio, then center-crop."""
        image = image.convert("RGB")
        w, h = image.size
        scale = target_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Center crop to exact target_size x target_size
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        image = image.crop((left, top, left + target_size, top + target_size))
        return image

    def get_model_info(self) -> dict:
        """Return info about current pipeline state."""
        return {
            "model_loaded": self.pipe is not None,
            "device": str(self.device),
            "dtype": str(self.pipe.dtype) if self.pipe else None,
            "lora_active": self.lora_loaded,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
