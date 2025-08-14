<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/f8affe3d-a8df-401d-ae3d-4586f0ae82a5" />

# Pattern Diffusion — Seamless Tile Generator (CLI + Gradio)

Small Windows-friendly wrapper around **Arrexel/pattern-diffusion** for generating seamless, tileable textures (e.g., wallpaper, fabrics, game materials).
Includes a command-line mode and a tiny Gradio web UI.

> **Credit:** The model and the seamless-tiling method are by **Arrexel**. Please see and support the original model here:
> **Hugging Face:** [https://huggingface.co/Arrexel/pattern-diffusion](https://huggingface.co/Arrexel/pattern-diffusion)

This wrapper follows the model card’s recommended inference recipe:

* Use **DDPM** scheduler
* **Roll the latents** during the first \~80% of steps
* Turn on **circular padding** for the UNet & VAE during the last \~20% of steps

---

## Features

* 🧩 Seamless, tileable SD2-based pattern generation
* 🖥️ Works great on NVIDIA GPUs (CUDA); CPU also supported
* 🧪 CLI for batchable runs, or `--gui` for a tiny local Gradio app
* ⚙️ Arguments for prompt/size/steps/guidance/seed/device
* 📦 Caches model locally (Hugging Face cache) so it can run offline after the first run

---

## Quick Start (Windows)

### 1) Clone or download this repo

```powershell
# PowerShell
cd "D:\AI"  #Use whatever folder you store your AI apps in
git clone <your-repo-url>.git "Pattern Diffusion"
cd "Pattern Diffusion"
```

> If you’re not using Git, just place `pattern_diffuse.py` in your target folder.

### 2) Create & activate a virtual environment

**PowerShell (preferred):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, either use Command Prompt (`.\.venv\Scripts\activate.bat`) or temporarily allow scripts for this session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3) Install PyTorch (CUDA build for NVIDIA)

For modern NVIDIA GPUs (e.g., RTX 4090), install a CUDA-enabled PyTorch wheel. Example for CUDA 12.8 wheels:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> If this fails on your system, visit pytorch.org “Get Started” and use their selector to grab the exact command for your setup.

### 4) Install the remaining dependencies

```powershell
pip install diffusers transformers accelerate pillow gradio
```

(If you don’t want the GUI, you can skip `gradio`.)

---

## Running

> The first run will download the model weights into your Hugging Face cache (e.g.,
> `C:\Users\<YOU>\AppData\Local\huggingface\hub\`).
> After that, you can run **fully offline** (see “Offline mode” below).

### A) Launch the Gradio UI

```powershell
# from inside your venv
python .\pattern_diffuse.py --gui
```

A local web UI will open in your browser. Adjust prompt/size/steps and click **Generate**.

### B) Command-line (CLI) example

```powershell
python .\pattern_diffuse.py `
  --prompt "vibrant watercolor floral pattern with pink, purple and blue flowers on white" `
  --width 1024 --height 1024 `
  --steps 50 `
  --guidance-scale 4.5 `
  --seed 123456 `
  --output floral.png `
  --device cuda
```

**Common flags:**

* `--prompt` *(string, required for CLI)*
* `--width` / `--height` *(ints; 1024×1024 is a good start)*
* `--steps` *(int; 50 default)*
* `--guidance-scale` *(float; default 4.5)*
* `--seed` *(int; -1/random if omitted)*
* `--device` *(cuda|cpu|mps; auto-picked if omitted)*
* `--gui` *(launch the Gradio app instead of CLI)*

Show help:

```powershell
python .\pattern_diffuse.py -h
```

---

## Offline mode (optional)

After the first successful run (which caches the model), you can force offline operation:

**PowerShell (session-only):**

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python .\pattern_diffuse.py --gui
```

Or hard-enforce in code by loading with `local_files_only=True` (will error if the cache isn’t present yet).

You can also move the cache to another drive by setting `HF_HOME` before the first download:

```powershell
$env:HF_HOME="D:\AI\hf"   # cache will be D:\AI\hf\hub
```

---

## What’s inside the script?

* Loads `Arrexel/pattern-diffusion` via Diffusers
* Swaps in **DDPM** for inference
* **Per-step callback** that:

  * rolls the **latents** (wrap-around) during the early steps
  * enables **circular padding** on UNet + VAE for the final steps
* CLI + optional Gradio UI

---

## Credits & License

* Model and method by **Arrexel**: [https://huggingface.co/Arrexel/pattern-diffusion](https://huggingface.co/Arrexel/pattern-diffusion)
* This repo only provides a thin, Windows-friendly wrapper (script + optional UI).
* **Licensing/usage:** See the model card on Hugging Face for the model’s license and usage terms.

---

### Troubleshooting

* **“Torch not compiled with CUDA enabled”** → Reinstall PyTorch with the CUDA wheel (see step 3).
* **Powershell activation blocked** → Use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or use Command Prompt’s `activate.bat`.
* **Out of VRAM** → Try 768×768 or 640×640, or reduce steps.

---

Happy pattern-making! 🧵
