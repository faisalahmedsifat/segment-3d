"""Modal deployment for the Segment Anything Model (SAM) inference service.

Deploy with:
    uv run modal deploy sam_modal.py

The SAM ViT-B checkpoint is downloaded while building the Modal image. The
predictor is initialized lazily and cached for the lifetime of each warm
container, avoiding a model reload for every request handled by that container.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import modal

APP_NAME = "sam-segmentation"
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "/opt/models/sam_vit_b_01ec64.pth"
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/"
    "sam_vit_b_01ec64.pth"
)

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "git")
    .pip_install(
        "torch",
        "torchvision",
        "opencv-python-headless",
        "pycocotools",
        "matplotlib",
        "numpy",
        "pillow",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/segment-anything.git /opt/segment-anything && pip install --no-deps /opt/segment-anything",
        "mkdir -p /opt/models",
        f"curl -fL {CHECKPOINT_URL} -o {CHECKPOINT_PATH}",
    )
)

# A Modal container can handle multiple requests. Cache the loaded predictor
# in its process so the model is not reloaded on every invocation.
_predictor: Any | None = None


def _get_predictor() -> Any:
    global _predictor
    if _predictor is None:
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device=device)
        _predictor = SamPredictor(model)
        print(f"SAM {MODEL_TYPE} loaded on {device}")
    return _predictor


@app.function(image=image, gpu="T4", timeout=600)
def segment_image_simple(
    image_base64: str,
    points: list[list[float]],
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Run point-prompted SAM segmentation and return JSON-compatible data."""
    try:
        from PIL import Image
        import numpy as np

        image_bytes = base64.b64decode(image_base64)
        with Image.open(io.BytesIO(image_bytes)) as source:
            image_array = np.asarray(source.convert("RGB"))

        point_array = np.asarray(points, dtype=np.float32)
        if point_array.ndim != 2 or point_array.shape[1] != 2 or len(point_array) == 0:
            raise ValueError("points must be a non-empty list of [x, y] coordinates")

        label_array = np.asarray(
            labels if labels is not None else [1] * len(point_array),
            dtype=np.int32,
        )
        if len(label_array) != len(point_array):
            raise ValueError("labels and points must contain the same number of entries")

        predictor = _get_predictor()
        predictor.set_image(image_array)
        masks, scores, logits = predictor.predict(
            point_coords=point_array,
            point_labels=label_array,
            multimask_output=True,
        )

        return {
            "success": True,
            "masks": masks.tolist(),
            "scores": scores.tolist(),
            "logits": logits.tolist(),
            "image_shape": list(image_array.shape),
            "points_used": point_array.tolist(),
            "message": "Segmentation successful",
        }
    except Exception as exc:
        print(f"SAM segmentation error: {exc}")
        return {"success": False, "error": str(exc), "message": "Segmentation failed"}


@app.function()
def health_check() -> dict[str, str]:
    """Confirm that the deployed SAM inference service is reachable."""
    return {"status": "healthy", "message": "SAM Modal deployment is running"}
