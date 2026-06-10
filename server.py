#!/usr/bin/env python3
"""MCP server for HiDream O1 image generation and editing via ComfyUI."""

import asyncio
import base64
import json
import os
import random
import time
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
OUTPUT_DIR = Path(os.environ.get("COMFYUI_OUTPUT_DIR", "/home/tyrel/ComfyUI/output"))
POLL_INTERVAL = float(os.environ.get("COMFYUI_POLL_INTERVAL", "2.0"))
POLL_TIMEOUT = float(os.environ.get("COMFYUI_POLL_TIMEOUT", "300.0"))
MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "8200"))

mcp = FastMCP(
    "hidream-o1-image-gen",
    host=MCP_HOST,
    port=MCP_PORT,
    stateless_http=True,
    streamable_http_path="/mcp",
)


def _text2img_workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    filename_prefix: str,
) -> dict:
    return {
        "6": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "hidream_o1_image_dev_fp8_scaled.safetensors"},
        },
        "124": {
            "class_type": "ModelNoiseScale",
            "inputs": {"model": ["6", 0], "noise_scale": 7.6},
        },
        "125": {
            "class_type": "SamplerLCM",
            "inputs": {"s_noise": 1.0, "s_noise_end": 1.0, "noise_clip_std": 2.5},
        },
        "112": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["124", 0],
                "scheduler": "normal",
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "110": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["6", 1], "text": prompt},
        },
        "188": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["6", 1], "text": negative_prompt},
        },
        "156": {
            "class_type": "EmptyHiDreamO1LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "108": {
            "class_type": "SamplerCustom",
            "inputs": {
                "model": ["124", 0],
                "add_noise": True,
                "noise_seed": seed,
                "cfg": 1.0,
                "positive": ["110", 0],
                "negative": ["188", 0],
                "sampler": ["125", 0],
                "sigmas": ["112", 0],
                "latent_image": ["156", 0],
            },
        },
        "105": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["108", 0], "vae": ["6", 2]},
        },
        "227": {
            "class_type": "SaveImage",
            "inputs": {"images": ["105", 0], "filename_prefix": filename_prefix},
        },
    }


def _img2img_workflow(
    prompt: str,
    negative_prompt: str,
    image_filename: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    filename_prefix: str,
) -> dict:
    return {
        "6": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "hidream_o1_image_dev_fp8_scaled.safetensors"},
        },
        "124": {
            "class_type": "ModelNoiseScale",
            "inputs": {"model": ["6", 0], "noise_scale": 7.6},
        },
        "125": {
            "class_type": "SamplerLCM",
            "inputs": {"s_noise": 1.0, "s_noise_end": 1.0, "noise_clip_std": 2.5},
        },
        "112": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["124", 0],
                "scheduler": "normal",
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "213": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename, "upload": "image"},
        },
        "110": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["6", 1], "text": prompt},
        },
        "188": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["6", 1], "text": negative_prompt},
        },
        "104": {
            "class_type": "HiDreamO1ReferenceImages",
            "inputs": {
                "positive": ["110", 0],
                "negative": ["188", 0],
                "images.image_1": ["213", 0],
            },
        },
        "172": {
            "class_type": "EmptyHiDreamO1LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "108": {
            "class_type": "SamplerCustom",
            "inputs": {
                "model": ["124", 0],
                "add_noise": True,
                "noise_seed": seed,
                "cfg": 1.0,
                "positive": ["104", 0],
                "negative": ["104", 1],
                "sampler": ["125", 0],
                "sigmas": ["112", 0],
                "latent_image": ["172", 0],
            },
        },
        "105": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["108", 0], "vae": ["6", 2]},
        },
        "227": {
            "class_type": "SaveImage",
            "inputs": {"images": ["105", 0], "filename_prefix": filename_prefix},
        },
    }


async def _upload_image(client: httpx.AsyncClient, image_path: str) -> tuple[str, int, int]:
    """Upload image to ComfyUI, return (filename, width, height)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    from PIL import Image as PILImage
    with PILImage.open(path) as img:
        orig_w, orig_h = img.size

    with open(path, "rb") as f:
        data = f.read()

    suffix = path.suffix.lower() or ".png"
    mime = "image/png" if suffix == ".png" else "image/jpeg"

    response = await client.post(
        f"{COMFYUI_URL}/upload/image",
        files={"image": (path.name, data, mime)},
        data={"overwrite": "true"},
    )
    response.raise_for_status()
    result = response.json()
    return result["name"], orig_w, orig_h


async def _queue_prompt(client: httpx.AsyncClient, workflow: dict) -> str:
    """Submit workflow to ComfyUI, return prompt_id."""
    payload = {"prompt": workflow, "client_id": "mcp-hidream-o1"}
    response = await client.post(f"{COMFYUI_URL}/prompt", json=payload)
    response.raise_for_status()
    return response.json()["prompt_id"]


async def _wait_for_result(client: httpx.AsyncClient, prompt_id: str) -> list[dict]:
    """Poll until prompt finishes, return list of output image info dicts."""
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        await asyncio.sleep(POLL_INTERVAL)
        response = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
        response.raise_for_status()
        history = response.json()
        if prompt_id not in history:
            continue
        entry = history[prompt_id]
        if entry.get("status", {}).get("completed"):
            outputs = []
            for node_output in entry["outputs"].values():
                for img in node_output.get("images", []):
                    outputs.append(img)
            return outputs
        if entry.get("status", {}).get("status_str") == "error":
            msgs = entry.get("status", {}).get("messages", [])
            raise RuntimeError(f"ComfyUI error: {msgs}")
    raise TimeoutError(f"Prompt {prompt_id} did not complete within {POLL_TIMEOUT}s")


async def _fetch_image(client: httpx.AsyncClient, img_info: dict) -> bytes:
    """Download image data from ComfyUI output."""
    params = {
        "filename": img_info["filename"],
        "subfolder": img_info.get("subfolder", ""),
        "type": img_info.get("type", "output"),
    }
    response = await client.get(f"{COMFYUI_URL}/view", params=params)
    response.raise_for_status()
    return response.content


def _round32(n: int) -> int:
    return (n // 32) * 32


def _build_view_url(img_info: dict) -> str:
    from urllib.parse import urlencode
    params = {
        "filename": img_info["filename"],
        "subfolder": img_info.get("subfolder", ""),
        "type": img_info.get("type", "output"),
    }
    return f"{COMFYUI_URL}/view?{urlencode(params)}"


def _make_response(
    img_info: dict,
    image_data: bytes,
    prompt_id: str,
    seed: int,
    include_image: bool,
) -> list[TextContent | ImageContent]:
    output_path = OUTPUT_DIR / img_info.get("subfolder", "") / img_info["filename"]
    meta = json.dumps({
        "prompt_id": prompt_id,
        "seed": seed,
        "filename": img_info["filename"],
        "path": str(output_path),
        "url": _build_view_url(img_info),
    })
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=meta)]
    if include_image:
        b64 = base64.b64encode(image_data).decode()
        content.append(ImageContent(type="image", data=b64, mimeType="image/png"))
    return content


@mcp.tool()
async def generate_image(
    prompt: str,
    width: int = 2048,
    height: int = 2048,
    steps: int = 28,
    seed: int = -1,
    negative_prompt: str = "",
    filename_prefix: str = "hidream_o1",
    include_image: bool = True,
) -> list[TextContent | ImageContent]:
    """Generate an image from a text prompt using HiDream O1.

    Supported resolutions (width x height):
      Square:    2048x2048
      Landscape: 2304x1728, 2304x1792, 2496x1664, 2560x1440, 3104x1312
      Portrait:  1728x2304, 1792x2304, 1664x2496, 1440x2560, 1312x3104

    Args:
        prompt: Detailed image description.
        width: Output image width in pixels (default 2048).
        height: Output image height in pixels (default 2048).
        steps: Number of sampling steps (default 28).
        seed: Random seed; -1 for random.
        negative_prompt: Things to avoid in the image (usually left empty).
        filename_prefix: Prefix for saved output filename.
        include_image: If True (default), returns the image inline as ImageContent
            so clients can render it. Set False to receive only the JSON metadata
            (path + ComfyUI URL) and keep context size small.

    Returns:
        TextContent with JSON metadata (prompt_id, seed, filename, path, url),
        and optionally an ImageContent with the rendered image.
    """
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)

    width = _round32(width)
    height = _round32(height)

    workflow = _text2img_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
        filename_prefix=filename_prefix,
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        prompt_id = await _queue_prompt(client, workflow)
        outputs = await _wait_for_result(client, prompt_id)

        if not outputs:
            raise RuntimeError("No output images returned from ComfyUI")

        img_info = outputs[0]
        image_data = await _fetch_image(client, img_info) if include_image else b""

    return _make_response(img_info, image_data, prompt_id, seed, include_image)


@mcp.tool()
async def edit_image(
    prompt: str,
    image_path: str,
    steps: int = 28,
    seed: int = -1,
    negative_prompt: str = "",
    filename_prefix: str = "hidream_o1_edit",
    include_image: bool = True,
) -> list[TextContent | ImageContent]:
    """Edit or transform an existing image using HiDream O1.

    The model uses the provided image as a reference and generates a new image
    guided by both the reference and the text prompt. Useful for style transfer,
    background replacement, or instructed edits.

    Args:
        prompt: Description of the desired output (e.g. "Transform the background into a rainy neon city").
        image_path: Absolute path to the input image (PNG or JPEG).
        steps: Number of sampling steps (default 28).
        seed: Random seed; -1 for random.
        negative_prompt: Things to avoid in the image (usually left empty).
        filename_prefix: Prefix for saved output filename.
        include_image: If True (default), returns the image inline as ImageContent
            so clients can render it. Set False to receive only the JSON metadata
            (path + ComfyUI URL) and keep context size small.

    Returns:
        TextContent with JSON metadata (prompt_id, seed, filename, path, url),
        and optionally an ImageContent with the rendered image.
    """
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)

    async with httpx.AsyncClient(timeout=60.0) as client:
        image_filename, orig_w, orig_h = await _upload_image(client, image_path)

        # Scale to ~4MP then round dimensions to multiples of 32
        mp = orig_w * orig_h
        target_mp = 4_000_000
        if mp > target_mp:
            scale = (target_mp / mp) ** 0.5
            w = _round32(int(orig_w * scale))
            h = _round32(int(orig_h * scale))
        else:
            w = _round32(orig_w)
            h = _round32(orig_h)

        w = max(w, 64)
        h = max(h, 64)

        workflow = _img2img_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_filename=image_filename,
            width=w,
            height=h,
            steps=steps,
            seed=seed,
            filename_prefix=filename_prefix,
        )

        prompt_id = await _queue_prompt(client, workflow)
        outputs = await _wait_for_result(client, prompt_id)

        if not outputs:
            raise RuntimeError("No output images returned from ComfyUI")

        img_info = outputs[0]
        image_data = await _fetch_image(client, img_info) if include_image else b""

    return _make_response(img_info, image_data, prompt_id, seed, include_image)


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        app = mcp.streamable_http_app()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
        uvicorn.run(app, host=MCP_HOST, port=MCP_PORT, log_level="info")
    else:
        mcp.run(transport="stdio")
