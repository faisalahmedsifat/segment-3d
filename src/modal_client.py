"""Client for the deployed Modal SAM inference function."""

import base64
import io
from typing import Tuple

import cv2
import modal
import numpy as np
from PIL import Image


class ModalSAMClient:
    """Call the serverless SAM deployment and adapt its result for the GUI."""

    def __init__(self) -> None:
        self.app_name = "sam-segmentation"
        self.segment_func = None
        self._connect()

    def _connect(self) -> None:
        """Create a handle for the deployed SAM inference function."""
        try:
            print("Connecting to Modal SAM deployment...")
            self.segment_func = modal.Function.from_name(
                self.app_name,
                "segment_image_simple",
            )
            print("Modal SAM function reference initialized")
        except Exception as exc:
            print(f"Failed to initialize Modal function: {exc}")
            print("Deploy the service with: uv run modal deploy sam_modal.py")
            raise

    def _image_to_base64(self, image_path: str) -> str:
        """Read an image and encode it as JPEG base64 for the Modal call."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read screenshot: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        buffer = io.BytesIO()
        Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def predict(
        self,
        image_path: str,
        click_pos: np.ndarray,
        label: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return SAM masks, scores, and logits as NumPy arrays."""
        try:
            if self.segment_func is None:
                raise RuntimeError("Modal SAM function handle was not initialized")

            print(f"Sending segmentation request to Modal for image: {image_path}")
            image_base64 = self._image_to_base64(image_path)
            result = self.segment_func.remote(
                image_base64=image_base64,
                points=click_pos.tolist(),
                labels=label.tolist(),
            )

            if not result.get("success"):
                raise RuntimeError(
                    f"Modal segmentation failed: {result.get('error', 'Unknown error')}"
                )

            masks = np.asarray(result["masks"])
            scores = np.asarray(result["scores"])
            logits = np.asarray(result["logits"])
            print(f"Modal segmentation successful; masks shape: {masks.shape}")
            return masks, scores, logits
        except Exception as exc:
            print(f"Error in Modal prediction: {exc}")
            raise

    def health_check(self) -> bool:
        """Check that the deployed Modal service responds successfully."""
        try:
            health_func = modal.Function.from_name(self.app_name, "health_check")
            result = health_func.remote()
            return result.get("status") == "healthy"
        except Exception as exc:
            print(f"Health check failed: {exc}")
            return False


_modal_sam_client = None


def get_modal_sam_client() -> ModalSAMClient:
    """Return the process-wide Modal SAM client instance."""
    global _modal_sam_client
    if _modal_sam_client is None:
        _modal_sam_client = ModalSAMClient()
    return _modal_sam_client
