import modal
import base64
import cv2
import numpy as np
from PIL import Image
import io
import asyncio
import threading
from typing import List, Tuple, Optional

class ModalSAMClient:
    """
    Client to interact with Modal SAM deployment
    Replaces the local SAM model with cloud-based processing
    """
    
    def __init__(self):
        self.app_name = "sam-segmentation"
        self.app = None
        self.segmentor_cls = None
        self.segmentor_instance = None
        self._connect()
    
    def _connect(self):
        """Connect to the Modal SAM deployment"""
        try:
            print("Connecting to Modal SAM deployment...")
            self.app = modal.App.lookup(self.app_name, create_if_missing=False)
            self.segmentor_cls = modal.Cls.lookup(self.app_name, "SAMSegmentor")
            print("Successfully connected to Modal SAM deployment")
        except Exception as e:
            print(f"Failed to connect to Modal deployment: {e}")
            print("Make sure you've deployed with: modal deploy sam_modal.py")
            raise
    
    def _get_segmentor(self):
        """Get or create segmentor instance"""
        if self.segmentor_instance is None:
            # Create instance but don't store it - let Modal handle lifecycle
            pass
        return self.segmentor_cls()
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string"""
        try:
            # Read image using cv2 (matching original code)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=95)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return image_base64
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            raise
    
    def predict(self, image_path: str, click_pos: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict segmentation masks using Modal SAM
        
        Args:
            image_path: Path to the image file
            click_pos: Click positions as numpy array
            label: Labels for the click positions
            
        Returns:
            Tuple of (masks, scores, logits) matching original SAM output format
        """
        try:
            print(f"Sending segmentation request to Modal for image: {image_path}")
            print(f"Click positions: {click_pos}")
            print(f"Labels: {label}")
            
            # Convert image to base64
            image_base64 = self._image_to_base64(image_path)
            
            # Convert numpy arrays to lists for API call
            points_list = click_pos.tolist()
            labels_list = label.tolist()
            
            # Use the simple function wrapper instead of class
            segment_func = modal.Function.lookup(self.app_name, "segment_image_simple")
            
            # Make the Modal API call
            result = segment_func.remote(
                image_base64=image_base64,
                points=points_list,
                labels=labels_list
            )
            
            if not result["success"]:
                raise Exception(f"Modal segmentation failed: {result.get('error', 'Unknown error')}")
            
            print(f"Modal segmentation successful! Message: {result['message']}")
            
            # Convert results back to numpy arrays (matching original format)
            masks = np.array(result["masks"])
            scores = np.array(result["scores"])
            logits = np.array(result["logits"])
            
            print(f"Received masks shape: {masks.shape}")
            print(f"Received scores: {scores}")
            
            return masks, scores, logits
            
        except Exception as e:
            print(f"Error in Modal prediction: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Modal deployment is healthy"""
        try:
            health_func = modal.Function.lookup(self.app_name, "health_check")
            result = health_func.remote()
            return result["status"] == "healthy"
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

# Singleton instance for global use
_modal_sam_client = None

def get_modal_sam_client() -> ModalSAMClient:
    """Get singleton Modal SAM client instance"""
    global _modal_sam_client
    if _modal_sam_client is None:
        _modal_sam_client = ModalSAMClient()
    return _modal_sam_client 