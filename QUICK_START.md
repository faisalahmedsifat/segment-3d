# Segment3D Quick Start

## Prerequisites

- Linux desktop session with a working graphical display
- Python 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A Modal account for cloud-based SAM inference

## 1. Install the project

```bash
git clone https://github.com/faisalahmedsifat/segment-3d.git
cd segment-3d
git checkout optimization
uv sync
```

## 2. Authenticate with Modal

```bash
uv run modal setup
```

Complete the browser authentication flow.

## 3. Deploy the SAM inference service

```bash
uv run modal deploy sam_modal.py
```

The first deployment builds the inference image, installs PyTorch and Segment Anything, and downloads the SAM ViT-B checkpoint. This may take several minutes. Inference uses a Modal T4 GPU and is subject to Modal usage charges.

Confirm the app is deployed:

```bash
uv run modal app list
```

## 4. Launch the desktop application

```bash
uv run python src/main.py
```

Select an STL file, continue to the 3D viewer, and right-click the model to request a segmentation.

## Troubleshooting

- **Qt cannot load `xcb`:** use the `opencv-python-headless` dependency already declared in `pyproject.toml`; avoid installing `opencv-python` alongside PyQt5.
- **Modal function not found:** authenticate with `uv run modal setup`, then deploy with `uv run modal deploy sam_modal.py`.
- **Segmentation request fails:** inspect the Modal app logs in the Modal dashboard and confirm the `sam-segmentation` app is deployed.

For the full setup and architecture notes, see [README.md](README.md).
