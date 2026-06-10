# hidream-o1-image-gen

MCP server for image generation and editing using the HiDream O1 model via [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Tools

- **`generate_image`** — text-to-image generation. Supports the resolutions natively supported by HiDream O1 (square, landscape, and portrait variants up to ~5MP).
- **`edit_image`** — edits/transforms an existing image using a text prompt and the image as a reference, via `HiDreamO1ReferenceImages`.

Both tools poll ComfyUI until the prompt finishes and return JSON metadata (path, seed, ComfyUI view URL); optionally they also return the rendered image inline.

## Requirements

- ComfyUI running and reachable (default `http://127.0.0.1:8188`)
- Checkpoint `hidream_o1_image_dev_fp8_scaled.safetensors` installed
- HiDream O1 custom nodes (`EmptyHiDreamO1LatentImage`, `HiDreamO1ReferenceImages`, `ModelNoiseScale`, `SamplerLCM`)
- Python 3.11+, dependencies in `pyproject.toml` (`mcp`, `httpx`, `pillow`)

## Configuration (env vars)

| Variable | Default | Description |
| --- | --- | --- |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | Base URL of ComfyUI instance |
| `COMFYUI_OUTPUT_DIR` | `/home/tyrel/ComfyUI/output` | Path to ComfyUI output directory (for resolving file paths in responses) |
| `COMFYUI_POLL_INTERVAL` | `2.0` | Seconds between history polls |
| `COMFYUI_POLL_TIMEOUT` | `300.0` | Max seconds to wait for a prompt to complete |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `streamable-http` |
| `MCP_HOST` | `0.0.0.0` | Host to bind when using `streamable-http` |
| `MCP_PORT` | `8200` | Port to bind when using `streamable-http` |

## Running

### As a stdio MCP server (per-session)

Register in your MCP client config (e.g. `~/.claude/.mcp.json`):

```json
{
  "mcpServers": {
    "hidream-o1-image-gen": {
      "command": "python3",
      "args": ["server.py"],
      "cwd": "/home/tyrel/projects/hidream-o1-image-gen"
    }
  }
}
```

### As a standalone HTTP server (systemd)

A user systemd unit is provided at `~/.config/systemd/user/hidream-o1-mcp.service`, which runs the server with `MCP_TRANSPORT=streamable-http` on port `8200`, exposing the MCP endpoint at `http://localhost:8200/mcp`.

```bash
systemctl --user daemon-reload
systemctl --user enable --now hidream-o1-mcp.service
systemctl --user status hidream-o1-mcp.service
```
