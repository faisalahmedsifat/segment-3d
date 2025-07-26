import modal

# Create Modal app
app = modal.App("sam-segmentation")

# Define the Docker image with SAM dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "torch",
    "torchvision", 
    "segment-anything",
    "opencv-python",
    "pillow",
    "numpy"
]).run_commands([
    # Install curl first, then download SAM model weights
    "apt-get update && apt-get install -y curl",
    "curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o /sam_vit_b_01ec64.pth"
])

# Create a class to hold the model (loads once, reuses across calls) 
@app.cls(
    gpu="T4",  # Fixed deprecated syntax
    image=image,
    container_idle_timeout=300,  # Keep warm for 5 minutes
    allow_concurrent_inputs=5,  # Handle multiple requests concurrently
)
class SAMSegmentor:
    def __enter__(self):
        """Load model when container starts"""
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        
        print("Loading SAM model on Modal...")
        
        # Load SAM model - matching your original setup
        sam_checkpoint = "/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        
        print(f"SAM model loaded successfully on {device}!")
        return self
    
    @modal.enter()
    def load_model(self):
        """Alternative entry point for Modal"""
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        
        print("Loading SAM model on Modal...")
        
        # Load SAM model - matching your original setup
        sam_checkpoint = "/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        
        print(f"SAM model loaded successfully on {device}!")
    
    @modal.method()
    def segment_image(self, image_base64: str, points: list, labels: list = None):
        """
        Segment image using SAM - matches your original generate_mask functionality
        
        Args:
            image_base64: Base64 encoded image
            points: List of [x, y] coordinates where user clicked
            labels: List of 1s (positive) or 0s (negative) for each point
        
        Returns:
            Dictionary with segmentation masks, scores, and logits
        """
        import io
        import base64
        import numpy as np
        from PIL import Image
        
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image_pil = Image.open(io.BytesIO(image_data))
            image_np = np.array(image_pil.convert('RGB'))
            
            # Set image for SAM predictor
            self.predictor.set_image(image_np)
            
            # Convert points to numpy array
            input_points = np.array(points)
            input_labels = np.array(labels if labels else [1] * len(points))
            
            # Generate masks - matching your original predict call
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            
            # Convert numpy arrays to lists for JSON serialization
            masks_list = [mask.tolist() for mask in masks]
            scores_list = scores.tolist()
            logits_list = logits.tolist()
            
            return {
                "success": True,
                "masks": masks_list,
                "scores": scores_list,
                "logits": logits_list,
                "image_shape": image_np.shape,
                "points_used": points,
                "message": "Segmentation successful"
            }
            
        except Exception as e:
            print(f"Segmentation error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Segmentation failed"
            }

# Convenience function for external calls
@app.function(image=image, gpu="T4")
def segment_image_simple(image_base64: str, points: list, labels: list = None):
    """Simple function wrapper for one-off calls"""
    import io
    import base64
    import numpy as np
    from PIL import Image
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    
    try:
        print("Loading SAM model for one-off prediction...")
        
        # Load SAM model
        sam_checkpoint = "/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        
        print(f"SAM model loaded on {device}")
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image_pil = Image.open(io.BytesIO(image_data))
        image_np = np.array(image_pil.convert('RGB'))
        
        # Set image for SAM predictor
        predictor.set_image(image_np)
        
        # Convert points to numpy array
        input_points = np.array(points)
        input_labels = np.array(labels if labels else [1] * len(points))
        
        # Generate masks
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Convert numpy arrays to lists for JSON serialization
        masks_list = [mask.tolist() for mask in masks]
        scores_list = scores.tolist()
        logits_list = logits.tolist()
        
        return {
            "success": True,
            "masks": masks_list,
            "scores": scores_list,
            "logits": logits_list,
            "image_shape": image_np.shape,
            "points_used": points,
            "message": "Segmentation successful"
        }
        
    except Exception as e:
        print(f"Segmentation error: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Segmentation failed"
        }

# Health check function
@app.function(image=image)
def health_check():
    """Simple health check to verify deployment"""
    return {"status": "healthy", "message": "SAM Modal deployment is running"} 