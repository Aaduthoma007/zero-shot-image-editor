"""
Prompt Parser — Intelligent prompt analysis for image editing.

Extracts structured information from free-form user prompts:
  - Style instructions  (e.g., "oil painting", "watercolor", "cyberpunk")
  - Object edits        (e.g., "add a hat", "remove the background")
  - Intensity modifiers (e.g., "slightly", "dramatically", "subtle")

This enhances the raw user prompt into a more effective Stable Diffusion prompt
by appending quality boosters and building a negative prompt.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Keyword Dictionaries ─────────────────────────────────────────────

STYLE_KEYWORDS = {
    "oil painting": "oil painting style, thick brushstrokes, textured canvas",
    "watercolor": "watercolor painting, soft washes, flowing colors, wet edges",
    "pencil sketch": "pencil sketch, graphite drawing, detailed linework",
    "anime": "anime style, cel shading, vibrant colors, manga aesthetic",
    "cyberpunk": "cyberpunk style, neon lighting, futuristic, dark atmosphere",
    "vintage": "vintage photograph, film grain, faded colors, retro mood",
    "pop art": "pop art style, bold colors, halftone dots, comic aesthetic",
    "impressionist": "impressionist painting, soft brushstrokes, light play, Monet-style",
    "photorealistic": "photorealistic, ultra detailed, 8k, sharp focus",
    "fantasy": "fantasy art, magical atmosphere, ethereal lighting, dreamlike",
    "noir": "film noir style, high contrast, dramatic shadows, black and white",
    "pixel art": "pixel art style, 8-bit, retro game aesthetic",
    "3d render": "3d render, octane render, volumetric lighting, smooth surfaces",
    "cartoon": "cartoon style, bold outlines, flat colors, playful",
    "gothic": "gothic style, dark atmosphere, ornate details, dramatic",
    "minimalist": "minimalist style, clean lines, simple composition, negative space",
    "surreal": "surrealist art, dreamlike, impossible geometry, Dali-inspired",
    "studio ghibli": "Studio Ghibli style, Miyazaki aesthetic, warm colors, whimsical",
}

INTENSITY_MODIFIERS = {
    # Low intensity → lower strength
    "slightly": 0.3,
    "subtle": 0.3,
    "a bit": 0.35,
    "a little": 0.35,
    "gently": 0.35,
    # Medium intensity → default strength
    "moderately": 0.5,
    "": 0.55,  # default
    # High intensity → higher strength
    "very": 0.7,
    "dramatically": 0.8,
    "extremely": 0.85,
    "completely": 0.9,
    "totally": 0.9,
}

OBJECT_EDIT_PATTERNS = [
    r"(?:add|place|put|insert)\s+(?:a\s+)?(.+?)(?:\s+(?:to|on|in|near)\s+|$)",
    r"(?:remove|delete|erase|take away)\s+(?:the\s+)?(.+?)(?:\s+from\s+|$)",
    r"(?:change|replace|swap|turn)\s+(?:the\s+)?(.+?)\s+(?:to|into|with)\s+(.+)",
    r"(?:make)\s+(?:it|the image|the photo)\s+(?:look\s+)?(?:like\s+)?(?:a\s+)?(.+)",
]

# Quality boosters appended to every prompt
QUALITY_SUFFIXES = [
    "high quality",
    "detailed",
    "professional",
]

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, distorted, low quality, deformed, ugly, bad anatomy, "
    "watermark, text, logo, out of frame, duplicate, morbid, mutilated, "
    "poorly drawn, extra limbs, missing limbs, floating limbs"
)


class PromptParser:
    """
    Parses free-form user prompts into structured data for the diffusion pipeline.

    Example:
        Input:   "make it look like a dramatic oil painting"
        Output:  {
            "enhanced_prompt": "oil painting style, thick brushstrokes, ... high quality, detailed",
            "negative_prompt": "blurry, distorted, ...",
            "detected_style": "oil painting",
            "suggested_strength": 0.8,  # "dramatic" modifier detected
            "object_edits": [],
            "original_prompt": "make it look like a dramatic oil painting"
        }
    """

    def parse(self, prompt: str) -> dict:
        """
        Parse a user prompt into structured editing instructions.

        Args:
            prompt: Raw natural language prompt from the user.

        Returns:
            Dict with enhanced_prompt, negative_prompt, detected_style,
            suggested_strength, object_edits, and original_prompt.
        """
        prompt_lower = prompt.lower().strip()
        logger.info(f"Parsing prompt: '{prompt}'")

        # 1. Detect style
        detected_style, style_enhancement = self._detect_style(prompt_lower)

        # 2. Detect intensity modifier → suggested strength
        suggested_strength = self._detect_intensity(prompt_lower)

        # 3. Detect object edits
        object_edits = self._detect_object_edits(prompt_lower)

        # 4. Build enhanced prompt
        enhanced_prompt = self._build_enhanced_prompt(
            prompt, style_enhancement, object_edits
        )

        # 5. Build negative prompt
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

        result = {
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "detected_style": detected_style,
            "suggested_strength": suggested_strength,
            "object_edits": object_edits,
            "original_prompt": prompt,
        }

        logger.info(f"Parsed result: style={detected_style}, strength={suggested_strength}")
        return result

    # ── Private Methods ──────────────────────────────────────────────

    def _detect_style(self, prompt: str) -> tuple:
        """Find the best matching style keyword in the prompt."""
        for keyword, enhancement in STYLE_KEYWORDS.items():
            if keyword in prompt:
                logger.info(f"  Detected style: {keyword}")
                return keyword, enhancement
        return None, None

    def _detect_intensity(self, prompt: str) -> float:
        """Find intensity modifiers and return a suggested strength value."""
        for modifier, strength in sorted(
            INTENSITY_MODIFIERS.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if modifier and modifier in prompt:
                logger.info(f"  Detected intensity modifier: '{modifier}' → strength={strength}")
                return strength
        return INTENSITY_MODIFIERS[""]  # default

    def _detect_object_edits(self, prompt: str) -> list:
        """Extract object-level edit instructions via regex patterns."""
        edits = []
        for pattern in OBJECT_EDIT_PATTERNS:
            matches = re.finditer(pattern, prompt, re.IGNORECASE)
            for match in matches:
                edit_text = match.group(0).strip()
                if edit_text and len(edit_text) > 3:
                    edits.append(edit_text)
        return edits

    def _build_enhanced_prompt(
        self, original: str, style_enhancement: str | None, object_edits: list
    ) -> str:
        """Assemble the final enhanced prompt."""
        parts = []

        if style_enhancement:
            parts.append(style_enhancement)

        # Always include the original prompt for context
        parts.append(original)

        # Add quality boosters
        parts.extend(QUALITY_SUFFIXES)

        enhanced = ", ".join(parts)
        logger.info(f"  Enhanced prompt: '{enhanced[:100]}...'")
        return enhanced
