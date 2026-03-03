# 🎨 Zero-Shot Image Editing Assistant

> Upload an image → Describe your edit → Get AI-generated results  
> Powered by **Stable Diffusion img2img** + **LoRA Transfer Learning**

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [How It Works](#-how-it-works)
- [Architecture](#-architecture)
- [Transfer Learning Explained](#-transfer-learning-explained)
- [Setup Instructions](#-setup-instructions)
- [GPU Configuration](#-gpu-configuration)
- [Model Download](#-model-download)
- [Running the App](#-running-the-app)
- [Example Prompts](#-example-prompts)
- [Docker Deployment](#-docker-deployment)
- [HuggingFace Spaces Deployment](#-huggingface-spaces-deployment)
- [Troubleshooting](#-troubleshooting)
- [Performance Optimization](#-performance-optimization)
- [Folder Structure](#-folder-structure)

---

## 🧠 How It Works

1. **User uploads** a source image (PNG/JPG/WebP)
2. **User writes** a natural-language editing prompt
3. **Prompt Parser** extracts style keywords, intensity modifiers & object edits
4. **Stable Diffusion img2img** encodes the image into latent space, adds noise proportional to `strength`, and denoises guided by the text prompt
5. **LoRA adapter** (transfer learning) modulates the model's style behaviour
6. **Result** is displayed side-by-side with the original and can be downloaded

---

## 🏗 Architecture

```
Browser (HTML/CSS/JS)
    │
    ├── POST /upload     →  Save image to static/uploads/
    ├── POST /generate   →  Prompt Parser → Diffusion Pipeline → Output
    └── GET  /download   →  Serve output file
          │
   Flask Server (app.py)
          │
          ├── utils/prompt_parser.py      ← NLP-style prompt analysis
          └── utils/diffusion_pipeline.py ← SD img2img + LoRA
                    │
             HuggingFace Diffusers
                    │
         runwayml/stable-diffusion-v1-5 (pre-trained)
```

---

## 🔬 Transfer Learning Explained

This project uses **two levels of transfer learning** — neither trains a model from scratch:

### Level 1: Pre-Trained Stable Diffusion

The base model (`runwayml/stable-diffusion-v1-5`) was trained on **LAION-5B** (~5 billion image-text pairs). We load these pre-trained weights directly — this is the simplest form of transfer learning: **feature reuse**.

```python
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16  # Half-precision for GPU efficiency
)
pipe = pipe.to("cuda")  # GPU acceleration
```

### Level 2: LoRA (Low-Rank Adaptation)

LoRA is a **parameter-efficient fine-tuning** technique:

- The base model's attention weight matrices **W** are frozen (not changed)
- Two small matrices **A** (down-projection) and **B** (up-projection) are added: `W' = W + scale × (B @ A)`
- LoRA weights are only **a few MB** vs. the full model's **~4 GB**
- They encode domain-specific knowledge (e.g., anime style, oil painting style)
- We load pre-trained LoRA weights from HuggingFace Hub: `pipe.load_lora_weights("lora-library/some-style")`

**Why this is transfer learning:**
- The base model's knowledge is **transferred** to new tasks
- LoRA adapts the model to a new domain without retraining the full model
- Inference is fast because only the small adapter weights change the model's behaviour

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.10+**
- **8 GB+ RAM** (for CPU mode)
- **NVIDIA GPU with 4+ GB VRAM** (recommended, not required)
- **Git** installed

### Step 1: Clone / Navigate to the project

```bash
cd c:\Users\tomge\Desktop\ai_img
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install PyTorch with CUDA (if you have an NVIDIA GPU)

```bash
# Check your CUDA version first:
nvidia-smi

# Install matching PyTorch:
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (no GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 🖥 GPU Configuration

### Check GPU availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "")
```

### Memory optimization (already enabled in the code)

| Technique | What It Does | VRAM Savings |
|-----------|-------------|-------------|
| `torch.float16` | Half-precision computation | ~50% |
| `enable_attention_slicing()` | Splits attention into chunks | ~40% peak reduction |
| `enable_vae_slicing()` | Decodes latents in slices | ~30% peak reduction |
| `safety_checker=None` | Disables NSFW classifier | ~300 MB |

These are all enabled by default to **keep your laptop cool**.

---

## 📥 Model Download

The model downloads **automatically** on the first generation request. No manual download needed.

| What | Size | When |
|------|------|------|
| Stable Diffusion v1.5 | ~4 GB | First `/generate` call |
| LoRA weights (if configured) | ~10-50 MB | First `/generate` call |

Models are cached in the `models/` directory. Subsequent runs use the cache.

### Manual download (optional)

```python
from diffusers import StableDiffusionImg2ImgPipeline
StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir="./models")
```

---

## ▶️ Running the App

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## 💡 Example Prompts

| Prompt | Strength | Effect |
|--------|----------|--------|
| `make it look like an oil painting` | 0.55 | Adds brush-stroke texture |
| `transform into anime style, Studio Ghibli aesthetic` | 0.65 | Anime-style conversion |
| `add dramatic cyberpunk neon lighting` | 0.50 | Neon glow + dark atmosphere |
| `convert to pencil sketch with detailed shading` | 0.70 | Graphite sketch look |
| `make it look like a vintage photograph` | 0.45 | Film grain + faded colours |
| `slightly add sunset lighting` | 0.30 | Subtle warm tones |
| `dramatically change to watercolor painting` | 0.80 | Heavy watercolor effect |
| `transform into impressionist painting, Monet style` | 0.60 | Soft brushstrokes |

---

## 🐳 Docker Deployment

### Build and run

```bash
docker build -t zero-shot-editor .
docker run -p 5000:5000 zero-shot-editor
```

### With GPU (NVIDIA Container Toolkit required)

```bash
docker run --gpus all -p 5000:5000 zero-shot-editor
```

---

## 🤗 HuggingFace Spaces Deployment

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Gradio** or **Docker** SDK
3. Upload all project files
4. Set Space hardware to **T4 GPU** (free tier available)
5. The app will auto-build and deploy

For HuggingFace Spaces, you may need to change `app.py`'s host:

```python
app.run(host="0.0.0.0", port=7860)  # Spaces uses port 7860
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce image size in `config.py` (`DEFAULT_IMAGE_SIZE = 384`), or reduce `DEFAULT_NUM_STEPS` |
| `Model download stuck` | Check internet connection; try VPN if HuggingFace is blocked |
| `RuntimeError: Expected Float16` | You're on CPU — set `DTYPE = torch.float32` in `config.py` |
| `Very slow generation` | Normal for CPU (~2-5 min). Use GPU for ~20-30s generation |
| `Laptop gets hot` | Reduce `DEFAULT_NUM_STEPS` to 15-20, reduce `DEFAULT_IMAGE_SIZE` to 384 |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in your virtual environment |
| `Image looks distorted` | Lower the strength (0.3-0.4) to preserve more structure |
| `Output ignores prompt` | Increase `guidance_scale` (try 10-12) and `strength` (try 0.6-0.7) |

---

## ⚡ Performance Optimization

### Speed improvements

1. **Use GPU** — 10-20x faster than CPU
2. **Reduce steps** — 15-20 steps gives decent quality much faster
3. **Reduce image size** — 384px instead of 512px
4. **Use float16** — already enabled for GPU

### Memory savings

1. **Attention slicing** — already enabled
2. **VAE slicing** — already enabled
3. **CPU offload** — set `ENABLE_CPU_OFFLOAD = True` in `config.py` for very low VRAM GPUs
4. **Reduce batch size** — this app processes one image at a time (optimal)

### Quality tips

1. **Increase steps** to 40-50 for higher quality
2. **Guidance scale** of 7-9 is usually optimal
3. **Strength** of 0.4-0.6 preserves structure well
4. **Negative prompts** are auto-generated by the prompt parser

---

## 📁 Folder Structure

```
zero_shot_editor/
│
├── app.py                          # Flask server (routes, API)
├── config.py                       # All configuration in one place
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container deployment
├── .dockerignore                   # Docker build exclusions
├── README.md                       # This file
│
├── models/                         # Cached model weights (auto-created)
│
├── static/
│   ├── css/
│   │   └── style.css               # Dark glassmorphism UI
│   ├── js/
│   │   └── app.js                  # Frontend logic
│   ├── uploads/                    # User-uploaded images
│   └── outputs/                    # AI-generated outputs
│
├── templates/
│   └── index.html                  # Main web interface
│
└── utils/
    ├── __init__.py
    ├── diffusion_pipeline.py       # SD img2img + LoRA pipeline
    └── prompt_parser.py            # NLP-style prompt analysis
```

---

## 📄 License

MIT License — Use freely for academic and personal projects.
