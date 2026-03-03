/**
 * Zero-Shot Image Editor — Frontend Application Logic
 *
 * Handles:
 *  - Drag-and-drop + click file upload
 *  - Prompt input with character counter
 *  - Quick-prompt chip selection
 *  - Slider controls for strength / guidance / steps
 *  - API calls to /upload and /generate
 *  - Loading overlay with tips
 *  - Before/after comparison display
 *  - Download button
 */

// ── DOM Elements ──────────────────────────────────────────────────
const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const uploadPreview  = document.getElementById("upload-preview");
const previewImage   = document.getElementById("preview-image");
const btnRemove      = document.getElementById("btn-remove");
const promptInput    = document.getElementById("prompt-input");
const charCount      = document.getElementById("char-count");
const strengthSlider = document.getElementById("strength-slider");
const strengthValue  = document.getElementById("strength-value");
const guidanceSlider = document.getElementById("guidance-slider");
const guidanceValue  = document.getElementById("guidance-value");
const stepsSlider    = document.getElementById("steps-slider");
const stepsValue     = document.getElementById("steps-value");
const btnGenerate    = document.getElementById("btn-generate");
const resultSection  = document.getElementById("result-section");
const resultOriginal = document.getElementById("result-original");
const resultEdited   = document.getElementById("result-edited");
const analysisGrid   = document.getElementById("analysis-grid");
const btnDownload    = document.getElementById("btn-download");
const btnNewEdit     = document.getElementById("btn-new-edit");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingTip     = document.getElementById("loading-tip");

// ── State ─────────────────────────────────────────────────────────
let uploadedFilename = null;
let uploadedURL      = null;

// ── Loading Tips ──────────────────────────────────────────────────
const TIPS = [
    "First run downloads the model (~4 GB). This only happens once.",
    "Lower strength values preserve more of the original image.",
    "Higher guidance makes the edit follow your prompt more closely.",
    "The model uses LoRA transfer learning for efficient style adaptation.",
    "GPU mode (CUDA) is ~10x faster than CPU mode.",
    "Try combining style prompts: 'watercolor oil painting hybrid'.",
    "More quality steps = better results but slower generation.",
    "Attention slicing keeps your GPU cool and prevents overheating.",
];

// ── Drag & Drop ───────────────────────────────────────────────────

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
        handleFile(file);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) {
        handleFile(fileInput.files[0]);
    }
});

// ── File Handling ─────────────────────────────────────────────────

async function handleFile(file) {
    // Show local preview immediately
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        dropZone.style.display = "none";
        uploadPreview.style.display = "block";
    };
    reader.readAsDataURL(file);

    // Upload to server
    const formData = new FormData();
    formData.append("image", file);

    try {
        const res = await fetch("/upload", { method: "POST", body: formData });
        const data = await res.json();

        if (res.ok) {
            uploadedFilename = data.filename;
            uploadedURL = data.url;
            updateGenerateButton();
        } else {
            alert("Upload failed: " + (data.error || "Unknown error"));
            resetUpload();
        }
    } catch (err) {
        alert("Upload failed: " + err.message);
        resetUpload();
    }
}

function resetUpload() {
    uploadedFilename = null;
    uploadedURL = null;
    previewImage.src = "";
    uploadPreview.style.display = "none";
    dropZone.style.display = "block";
    fileInput.value = "";
    updateGenerateButton();
}

btnRemove.addEventListener("click", resetUpload);

// ── Prompt Input ──────────────────────────────────────────────────

promptInput.addEventListener("input", () => {
    charCount.textContent = `${promptInput.value.length} / 500`;
    updateGenerateButton();
});

// Quick prompt chips
document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
        promptInput.value = chip.dataset.prompt;
        charCount.textContent = `${promptInput.value.length} / 500`;
        promptInput.focus();
        updateGenerateButton();
    });
});

// ── Sliders ───────────────────────────────────────────────────────

strengthSlider.addEventListener("input", () => {
    strengthValue.textContent = (strengthSlider.value / 100).toFixed(2);
});

guidanceSlider.addEventListener("input", () => {
    guidanceValue.textContent = parseFloat(guidanceSlider.value).toFixed(1);
});

stepsSlider.addEventListener("input", () => {
    stepsValue.textContent = stepsSlider.value;
});

// ── Generate Button State ─────────────────────────────────────────

function updateGenerateButton() {
    const ready = uploadedFilename && promptInput.value.trim().length > 0;
    btnGenerate.disabled = !ready;
}

// ── Generate ──────────────────────────────────────────────────────

btnGenerate.addEventListener("click", async () => {
    if (!uploadedFilename || !promptInput.value.trim()) return;

    // Show loading
    showLoading();

    const payload = {
        filename: uploadedFilename,
        prompt: promptInput.value.trim(),
        strength: parseFloat(strengthSlider.value) / 100,
        guidance_scale: parseFloat(guidanceSlider.value),
        num_steps: parseInt(stepsSlider.value),
    };

    try {
        const res = await fetch("/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const data = await res.json();
        hideLoading();

        if (res.ok) {
            showResult(data);
        } else {
            alert("Generation failed: " + (data.error || "Unknown error"));
        }
    } catch (err) {
        hideLoading();
        alert("Generation failed: " + err.message);
    }
});

// ── Result Display ────────────────────────────────────────────────

function showResult(data) {
    // Set images
    resultOriginal.src = uploadedURL;
    resultEdited.src = data.output_url;

    // Set download link
    btnDownload.href = `/download/${data.output_filename}`;
    btnDownload.download = data.output_filename;

    // Build analysis panel
    const analysis = data.prompt_analysis;
    analysisGrid.innerHTML = "";

    addAnalysisRow("Original Prompt", analysis.original);
    addAnalysisRow("Enhanced Prompt", analysis.enhanced);

    if (analysis.detected_style) {
        addAnalysisRow("Detected Style", analysis.detected_style, true);
    }

    addAnalysisRow("Suggested Strength", analysis.suggested_strength.toFixed(2));

    if (analysis.object_edits && analysis.object_edits.length > 0) {
        addAnalysisRow("Object Edits", analysis.object_edits.join(", "));
    }

    // Show section
    resultSection.style.display = "block";
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function addAnalysisRow(label, value, isStyle = false) {
    const row = document.createElement("div");
    row.className = "analysis-row";
    row.innerHTML = `
        <span class="analysis-label">${label}</span>
        <span class="analysis-value${isStyle ? " style-tag" : ""}">${value}</span>
    `;
    analysisGrid.appendChild(row);
}

// ── New Edit ──────────────────────────────────────────────────────

btnNewEdit.addEventListener("click", () => {
    resultSection.style.display = "none";
    promptInput.value = "";
    charCount.textContent = "0 / 500";
    updateGenerateButton();
    document.getElementById("prompt-section").scrollIntoView({ behavior: "smooth" });
});

// ── Loading Overlay ───────────────────────────────────────────────

let tipInterval = null;

function showLoading() {
    loadingOverlay.style.display = "flex";
    btnGenerate.querySelector(".btn-text").style.display = "none";
    btnGenerate.querySelector(".btn-loading").style.display = "flex";
    btnGenerate.disabled = true;

    // Rotate tips
    let tipIdx = 0;
    loadingTip.textContent = TIPS[tipIdx];
    tipInterval = setInterval(() => {
        tipIdx = (tipIdx + 1) % TIPS.length;
        loadingTip.style.opacity = 0;
        setTimeout(() => {
            loadingTip.textContent = TIPS[tipIdx];
            loadingTip.style.opacity = 1;
        }, 300);
    }, 5000);
}

function hideLoading() {
    loadingOverlay.style.display = "none";
    btnGenerate.querySelector(".btn-text").style.display = "inline";
    btnGenerate.querySelector(".btn-loading").style.display = "none";
    updateGenerateButton();

    if (tipInterval) {
        clearInterval(tipInterval);
        tipInterval = null;
    }
}
