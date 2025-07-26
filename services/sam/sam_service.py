"""
Updated Modal SAM service for Segment3D Platform
Enhanced version of the original sam_modal.py for web integration
"""

import modal
import base64
import cv2
import numpy as np
from PIL import Image
import io
from typing import List, Dict, Any, Optional

# Create Modal app
app = modal.App("segment3d-sam-platform")

# Enhanced Docker image with latest dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "torch>=2.0.0",
    "torchvision>=0.15.0", 
    "segment-anything>=1.0",
    "opencv-python>=4.8.0",
    "pillow>=10.1.0",
    "numpy>=1.24.0",
    "fastapi>=0.104.0",
    "pydantic>=2.5.0"
]).run_commands([
    "apt-get update && apt-get install -y curl",
    "curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o /sam_vit_b_01ec64.pth",
    "curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -o /sam_vit_l_0b3195.pth",
    "curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o /sam_vit_h_4b8939.pth"
])

# Enhanced SAM segmentor with multiple model support
@app.cls(
    gpu="T4",
    image=image,
    container_idle_timeout=300,  # Keep warm for 5 minutes
    allow_concurrent_inputs=10,  # Handle more concurrent requests
    timeout=300  # 5 minute timeout for processing
)
class SAMSegmentor:
    """Enhanced SAM segmentor with multiple model sizes and advanced features"""
    
    def __enter__(self):
        """Load SAM models when container starts"""
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        
        print("Loading SAM models on Modal...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.predictors = {}
        
        # Load multiple model sizes for different use cases
        model_configs = {
            "vit_b": "/sam_vit_b_01ec64.pth",      # Fast, good for real-time
            "vit_l": "/sam_vit_l_0b3195.pth",      # Balanced performance
            "vit_h": "/sam_vit_h_4b8939.pth"       # Best quality, slower
        }
        
        for model_type, checkpoint in model_configs.items():
            try:
                sam = sam_model_registry[model_type](checkpoint=checkpoint)
                sam.to(device=self.device)
                self.models[model_type] = sam
                self.predictors[model_type] = SamPredictor(sam)
                print(f"✅ {model_type.upper()} model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load {model_type} model: {e}")
        
        print(f"SAM models loaded on {self.device}!")
        return self
    
    @modal.method()
    async def segment_image(
        self, 
        image_base64: str, 
        points: List[List[float]], 
        labels: Optional[List[int]] = None,
        model_size: str = "vit_b",
        multimask_output: bool = True,
        return_logits: bool = False
    ) -> Dict[str, Any]:
        """
        Enhanced image segmentation with multiple model support
        
        Args:
            image_base64: Base64 encoded image
            points: List of [x, y] coordinates where user clicked
            labels: List of 1s (positive) or 0s (negative) for each point
            model_size: Model to use ("vit_b", "vit_l", "vit_h")
            multimask_output: Whether to return multiple mask candidates
            return_logits: Whether to return mask logits
        
        Returns:
            Dictionary with segmentation results
        """
        import numpy as np
        from PIL import Image
        
        try:
            # Validate model size
            if model_size not in self.predictors:
                model_size = "vit_b"  # Fallback to base model
            
            predictor = self.predictors[model_size]
            
            # Decode and process image
            image_data = base64.b64decode(image_base64)
            image_pil = Image.open(io.BytesIO(image_data))
            image_np = np.array(image_pil.convert('RGB'))
            
            # Set image for SAM predictor
            predictor.set_image(image_np)
            
            # Process points and labels
            input_points = np.array(points, dtype=np.float32)
            input_labels = np.array(labels if labels else [1] * len(points), dtype=np.int32)
            
            # Generate masks
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=multimask_output,
            )
            
            # Process results for JSON serialization
            result = {
                "success": True,
                "masks": [mask.tolist() for mask in masks],
                "scores": scores.tolist(),
                "image_shape": image_np.shape[:2],  # [height, width]
                "points_used": points,
                "labels_used": labels or [1] * len(points),
                "model_used": model_size,
                "mask_count": len(masks),
                "message": f"Segmentation successful using {model_size.upper()} model"
            }
            
            # Optionally include logits
            if return_logits:
                result["logits"] = logits.tolist()
            
            # Add mask statistics
            result["mask_stats"] = []
            for i, (mask, score) in enumerate(zip(masks, scores)):
                mask_area = int(np.sum(mask))
                result["mask_stats"].append({
                    "mask_id": i,
                    "area_pixels": mask_area,
                    "area_percentage": float(mask_area / (image_np.shape[0] * image_np.shape[1]) * 100),
                    "confidence_score": float(score)
                })
            
            return result
            
        except Exception as e:
            print(f"Segmentation error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Segmentation failed with {model_size} model"
            }
    
    @modal.method()
    async def batch_segment(
        self, 
        batch_data: List[Dict[str, Any]],
        model_size: str = "vit_b"
    ) -> List[Dict[str, Any]]:
        """
        Process multiple segmentation requests in batch
        
        Args:
            batch_data: List of segmentation requests
            model_size: Model to use for all requests
        
        Returns:
            List of segmentation results
        """
        results = []
        
        for item in batch_data:
            try:
                result = await self.segment_image(
                    image_base64=item.get("image_base64"),
                    points=item.get("points", []),
                    labels=item.get("labels"),
                    model_size=model_size,
                    multimask_output=item.get("multimask_output", True),
                    return_logits=item.get("return_logits", False)
                )
                result["batch_id"] = item.get("id", len(results))
                results.append(result)
                
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e),
                    "batch_id": item.get("id", len(results)),
                    "message": "Batch item processing failed"
                })
        
        return results
    
    @modal.method()
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            "available_models": list(self.models.keys()),
            "device": self.device,
            "model_details": {
                "vit_b": {
                    "name": "ViT-Base",
                    "description": "Fast model, good for real-time annotation",
                    "parameters": "91M",
                    "speed": "Fast"
                },
                "vit_l": {
                    "name": "ViT-Large", 
                    "description": "Balanced performance and quality",
                    "parameters": "308M",
                    "speed": "Medium"
                },
                "vit_h": {
                    "name": "ViT-Huge",
                    "description": "Best quality, slower processing",
                    "parameters": "636M", 
                    "speed": "Slow"
                }
            }
        }


# Convenience functions for external API calls
@app.function(image=image, gpu="T4", timeout=120)
def segment_image_simple(
    image_base64: str, 
    points: List[List[float]], 
    labels: Optional[List[int]] = None,
    model_size: str = "vit_b"
) -> Dict[str, Any]:
    """Simple function wrapper for one-off segmentation calls"""
    segmentor = SAMSegmentor()
    return segmentor.segment_image(image_base64, points, labels, model_size)


@app.function(image=image)
def health_check() -> Dict[str, str]:
    """Enhanced health check with model status"""
    import torch
    
    return {
        "status": "healthy",
        "service": "segment3d-sam-platform", 
        "version": "2.0.0",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "message": "SAM Platform deployment is running"
    }


@app.function(image=image, gpu="T4")
def benchmark_models() -> Dict[str, Any]:
    """Benchmark all available models for performance comparison"""
    import time
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    import numpy as np
    
    # Create a test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_points = np.array([[256, 256]], dtype=np.float32)
    test_labels = np.array([1], dtype=np.int32)
    
    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_configs = {
        "vit_b": "/sam_vit_b_01ec64.pth",
        "vit_l": "/sam_vit_l_0b3195.pth", 
        "vit_h": "/sam_vit_h_4b8939.pth"
    }
    
    for model_type, checkpoint in model_configs.items():
        try:
            # Load model
            start_time = time.time()
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=device)
            predictor = SamPredictor(sam)
            
            load_time = time.time() - start_time
            
            # Benchmark inference
            predictor.set_image(test_image)
            
            start_time = time.time()
            masks, scores, logits = predictor.predict(
                point_coords=test_points,
                point_labels=test_labels,
                multimask_output=True,
            )
            inference_time = time.time() - start_time
            
            results[model_type] = {
                "load_time_seconds": round(load_time, 3),
                "inference_time_seconds": round(inference_time, 3),
                "mask_count": len(masks),
                "average_score": float(np.mean(scores)),
                "status": "success"
            }
            
        except Exception as e:
            results[model_type] = {
                "status": "error",
                "error": str(e)
            }
    
    return {
        "benchmark_results": results,
        "test_conditions": {
            "image_size": "512x512",
            "device": device,
            "points_tested": 1
        },
        "timestamp": time.time()
    } 