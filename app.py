"""
Zero-Shot Image Editing Assistant — Flask Application

Routes:
    GET  /              → Serve the web UI
    POST /upload        → Upload an image, return filename
    POST /generate      → Run img2img pipeline, return output filename
    GET  /download/<fn> → Download a generated image
    GET  /model-info    → Return current pipeline state (for debug)

Design:
    - Single-file Flask server (MVP simplicity)
    - Lazy model loading (loads on first /generate request)
    - Thread-safe via Flask's built-in request handling
"""

import os
import uuid
import logging
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
from werkzeug.utils import secure_filename

from config import (
    UPLOAD_FOLDER, OUTPUT_FOLDER, ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH, LOG_LEVEL,
    DEFAULT_STRENGTH, DEFAULT_GUIDANCE_SCALE, DEFAULT_NUM_STEPS,
)
from utils.diffusion_pipeline import ImageEditor
from utils.prompt_parser import PromptParser

# ── Logging Setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Flask App ────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── Singletons ───────────────────────────────────────────────────────
editor = ImageEditor()
parser = PromptParser()
model_loaded = False


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_model():
    """Lazy-load the model on first generation request."""
    global model_loaded
    if not model_loaded:
        logger.info("First generation request — loading model...")
        editor.load_model()
        model_loaded = True


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    """
    Upload an image.

    Expects: multipart/form-data with field 'image'
    Returns: { "filename": "...", "url": "/static/uploads/..." }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Generate unique filename
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex[:12]}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)

    file.save(filepath)
    logger.info(f"Uploaded: {unique_name}")

    return jsonify({
        "filename": unique_name,
        "url": f"/static/uploads/{unique_name}",
    })


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate an edited image.

    Expects JSON:
        {
            "filename": "uploaded_image.jpg",
            "prompt": "make it look like an oil painting",
            "strength": 0.55,          (optional)
            "guidance_scale": 7.5,     (optional)
            "num_steps": 30            (optional)
        }

    Returns: { "output_url": "/static/outputs/...", "output_filename": "...", "prompt_analysis": {...} }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    filename = data.get("filename")
    prompt = data.get("prompt", "").strip()

    if not filename:
        return jsonify({"error": "No filename specified"}), 400
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Verify uploaded file exists
    input_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    if not os.path.exists(input_path):
        return jsonify({"error": "Uploaded image not found"}), 404

    # Parse user parameters
    strength = float(data.get("strength", DEFAULT_STRENGTH))
    guidance_scale = float(data.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    num_steps = int(data.get("num_steps", DEFAULT_NUM_STEPS))

    # Clamp values
    strength = max(0.1, min(1.0, strength))
    guidance_scale = max(1.0, min(20.0, guidance_scale))
    num_steps = max(10, min(100, num_steps))

    try:
        # 1. Parse the prompt
        parsed = parser.parse(prompt)

        # Use suggested strength if user hasn't explicitly set one
        if "strength" not in data:
            strength = parsed["suggested_strength"]

        # 2. Load model (lazy)
        ensure_model()

        # 3. Open the image
        input_image = Image.open(input_path)

        # 4. Run the diffusion pipeline
        output_image = editor.edit_image(
            image=input_image,
            prompt=parsed["enhanced_prompt"],
            negative_prompt=parsed["negative_prompt"],
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
        )

        # 5. Save output
        output_name = f"edited_{uuid.uuid4().hex[:12]}.png"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        output_image.save(output_path, "PNG")
        logger.info(f"Output saved: {output_name}")

        return jsonify({
            "output_url": f"/static/outputs/{output_name}",
            "output_filename": output_name,
            "prompt_analysis": {
                "original": parsed["original_prompt"],
                "enhanced": parsed["enhanced_prompt"],
                "detected_style": parsed["detected_style"],
                "suggested_strength": parsed["suggested_strength"],
                "object_edits": parsed["object_edits"],
            },
        })

    except Exception as e:
        logger.exception("Generation failed")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500


@app.route("/download/<filename>")
def download(filename):
    """Download a generated output image."""
    filepath = os.path.join(OUTPUT_FOLDER, secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath, as_attachment=True)


@app.route("/model-info")
def model_info():
    """Return current model/pipeline state (debug endpoint)."""
    return jsonify(editor.get_model_info())


# ── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("🚀 Starting Zero-Shot Image Editing Assistant")
    logger.info("   Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False)
