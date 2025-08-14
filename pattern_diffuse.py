# pattern_diffuse.py
# Windows-friendly script for Arrexel/pattern-diffusion
# - CLI mode (default) OR --gui for a small local web UI
# - Implements DDPM + early latent rolling + late circular padding
# - Works on CUDA (NVIDIA), MPS (Apple), or CPU

import argparse
import os
import random
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from diffusers import StableDiffusionPipeline, DDPMScheduler


# -------------------------------
# Circular padding patch for Conv2d
# -------------------------------
def _conv2d_forward_circular(mod, x, w, b):
    # pad X then Y using wrap-around
    px = (mod._reversed_padding_repeated_twice[0],
          mod._reversed_padding_repeated_twice[1], 0, 0)
    py = (0, 0, mod._reversed_padding_repeated_twice[2],
          mod._reversed_padding_repeated_twice[3])
    x = F.pad(x, px, mode="circular")
    x = F.pad(x, py, mode="circular")
    return F.conv2d(x, w, b, mod.stride, _pair(0), mod.dilation, mod.groups)


def enable_circular(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # keep LoRA-compatible convs safe
            if hasattr(m, "lora_layer") and m.lora_layer is None:
                m.lora_layer = lambda *x: 0
            m._conv_forward = _conv2d_forward_circular.__get__(m, nn.Conv2d)


def disable_circular(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if hasattr(m, "lora_layer") and m.lora_layer is None:
                m.lora_layer = lambda *x: 0
            m._conv_forward = nn.Conv2d._conv_forward.__get__(m, nn.Conv2d)


# -------------------------------
# Per-step callback factory (needs total_steps)
# -------------------------------
def make_step_callback(total_steps: int, roll_pixels_latent: int = 64):
    """
    Returns a callback like: (pipe, step_idx, timestep, callback_kwargs) -> callback_kwargs
    - First 80% of steps: roll latents by (roll_pixels_latent, roll_pixels_latent)
    - At exactly 80% mark: turn on circular padding for UNet and VAE
    """
    turn_on_at = int(total_steps * 0.8)

    def step_callback(pipe, step_idx, _t, kwargs):
        # Late stage: enable circular padding just once
        if step_idx == turn_on_at:
            enable_circular(pipe.unet)
            enable_circular(pipe.vae)

        # Early stage: roll latents (wrap-around shift)
        if step_idx < turn_on_at:
            if "latents" in kwargs and isinstance(kwargs["latents"], torch.Tensor):
                kwargs["latents"] = torch.roll(
                    kwargs["latents"],
                    shifts=(roll_pixels_latent, roll_pixels_latent),
                    dims=(2, 3),
                )
        return kwargs

    return step_callback


# -------------------------------
# Build/load pipeline
# -------------------------------
def pick_device(user_choice: Optional[str] = None) -> str:
    if user_choice:
        return user_choice
    if torch.cuda.is_available():
        return "cuda"
    # mps is for Apple; harmless to skip on Windows
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_pipeline(device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        "Arrexel/pattern-diffusion",
        torch_dtype=dtype,
    ).to(device)

    # Use DDPM as recommended
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Start with circular padding OFF
    disable_circular(pipe.unet)
    disable_circular(pipe.vae)
    return pipe


# -------------------------------
# CLI generation
# -------------------------------
def generate_once(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 50,
    output: str = "out.png",
    device: Optional[str] = None,
    seed: Optional[int] = None,
    guidance_scale: float = 4.5,
):
    device = pick_device(device)
    pipe = build_pipeline(device)

    # Reproducibility
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)

    # Prepare per-step callback with knowledge of total steps
    step_cb = make_step_callback(steps, roll_pixels_latent=64)

    img = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        callback_on_step_end=step_cb,
    ).images[0]

    img.save(output)
    print(f"Saved: {os.path.abspath(output)}")
    print(f"Seed:  {seed}")
    return output, seed


# -------------------------------
# Gradio UI (optional)
# -------------------------------
def launch_gui():
    try:
        import gradio as gr
    except Exception as e:
        print("Gradio is not installed. Run: pip install gradio")
        raise

    # Reuse a single pipeline for responsiveness
    state = {"pipe": None, "device": pick_device(None)}

    def ensure_pipe():
        if state["pipe"] is None:
            state["pipe"] = build_pipeline(state["device"])
        return state["pipe"]

    def ui_generate(prompt, w, h, steps, guidance, roll_latent, seed):
        pipe = ensure_pipe()
        if seed is None or int(seed) < 0:
            seed_val = random.randint(0, 2**31 - 1)
        else:
            seed_val = int(seed)

        generator = torch.Generator(device=state["device"]).manual_seed(seed_val)

        # Rebuild callback with UI-selected roll amount
        step_cb = make_step_callback(int(steps), roll_pixels_latent=int(roll_latent))

        # Reset circular padding to OFF at start of each run
        disable_circular(pipe.unet)
        disable_circular(pipe.vae)

        image = pipe(
            prompt=prompt,
            width=int(w),
            height=int(h),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=generator,
            callback_on_step_end=step_cb,
        ).images[0]
        return image, f"Seed: {seed_val} | Device: {state['device']}"

    with gr.Blocks(title="Pattern Diffusion") as demo:
        gr.Markdown("## Pattern Diffusion — seamless tiling (DDPM + rolling + late circular padding)")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value="vibrant watercolor floral pattern with pink, purple and blue flowers on white")
        with gr.Row():
            w = gr.Slider(512, 1536, value=1024, step=64, label="Width")
            h = gr.Slider(512, 1536, value=1024, step=64, label="Height")
        with gr.Row():
            steps = gr.Slider(20, 100, value=50, step=1, label="Steps")
            guidance = gr.Slider(1.0, 9.0, value=4.5, step=0.1, label="Guidance Scale")
        with gr.Row():
            roll_latent = gr.Slider(0, 128, value=64, step=1, label="Latent roll (pixels)")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
        go = gr.Button("Generate")
        out_img = gr.Image(type="pil", label="Result")
        info = gr.Markdown()

        def _on_click(prompt, w, h, steps, guidance, roll_latent, seed):
            img, meta = ui_generate(prompt, w, h, steps, guidance, roll_latent, seed)
            return img, meta

        go.click(
            _on_click,
            inputs=[prompt, w, h, steps, guidance, roll_latent, seed],
            outputs=[out_img, info],
        )

    demo.launch()


# -------------------------------
# Entrypoint
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Arrexel/pattern-diffusion")
    p.add_argument("--prompt", help="Text prompt (CLI mode)")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=4.5)
    p.add_argument("--output", default="out.png")
    p.add_argument("--device", choices=["cuda", "cpu", "mps"], default=None)
    p.add_argument("--seed", type=int, default=-1, help="-1 for random")
    p.add_argument("--gui", action="store_true", help="Launch Gradio UI instead of CLI generation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.gui:
        # Launch the UI
        launch_gui()
        sys.exit(0)

    # CLI mode
    if not args.prompt:
        print("Error: --prompt is required in CLI mode. Or use --gui to launch the UI.")
        sys.exit(2)

    seed_val = None if args.seed is None or args.seed < 0 else args.seed

    generate_once(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        output=args.output,
        device=args.device,
        seed=seed_val,
        guidance_scale=args.guidance_scale,
    )
