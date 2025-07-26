"""
SAM service implementing business logic for segmentation
Handles Modal integration and result processing
"""

from typing import List, Optional, Dict, Any
from prisma import Prisma
import structlog
import httpx
import asyncio

logger = structlog.get_logger(__name__)


class SAMService:
    """Service class for SAM segmentation operations"""
    
    def __init__(self, db: Optional[Prisma]):
        self.db = db
        self.logger = logger.bind(service="SAMService")
        self.modal_base_url = "https://segment3d-sam-platform--segment-image-simple.modal.run"
        
    async def segment_image(
        self,
        image_base64: str,
        points: List[List[float]],
        labels: Optional[List[int]] = None,
        model_size: str = "vit_b",
        multimask_output: bool = True,
        return_logits: bool = False,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Segment image using Modal SAM deployment
        
        Args:
            image_base64: Base64 encoded image
            points: List of [x, y] click coordinates
            labels: Optional labels for points (1 for positive, 0 for negative)
            model_size: SAM model size to use
            multimask_output: Whether to return multiple candidates
            return_logits: Whether to return mask logits
            user_id: User ID for logging/analytics
            
        Returns:
            Segmentation result dictionary
        """
        try:
            self.logger.info(
                "Starting SAM segmentation",
                user_id=user_id,
                model_size=model_size,
                points_count=len(points)
            )
            
            # Prepare request payload
            payload = {
                "image_base64": image_base64,
                "points": points,
                "labels": labels or [1] * len(points),
                "model_size": model_size
            }
            
            # Call Modal SAM service
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.modal_base_url,
                    json=payload
                )
                
                if response.status_code != 200:
                    raise Exception(f"Modal API error: {response.status_code} - {response.text}")
                
                result = response.json()
                
                if not result.get("success", False):
                    raise Exception(result.get("error", "Unknown SAM error"))
                
                # Log successful segmentation
                self.logger.info(
                    "SAM segmentation completed",
                    user_id=user_id,
                    model_size=model_size,
                    mask_count=len(result.get("masks", [])),
                    average_score=sum(result.get("scores", [])) / len(result.get("scores", [1])) if result.get("scores") else 0
                )
                
                # Enhance result with additional metadata
                enhanced_result = {
                    **result,
                    "processing_metadata": {
                        "user_id": user_id,
                        "model_size": model_size,
                        "points_processed": len(points),
                        "timestamp": asyncio.get_event_loop().time()
                    }
                }
                
                return enhanced_result
                
        except Exception as e:
            self.logger.error(
                "SAM segmentation failed",
                user_id=user_id,
                model_size=model_size,
                error=str(e)
            )
            raise
    
    async def batch_segment(
        self,
        batch_data: List[Dict[str, Any]],
        model_size: str = "vit_b",
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple segmentation requests in batch
        
        Args:
            batch_data: List of segmentation request dictionaries
            model_size: SAM model size for all requests
            user_id: User ID for logging
            
        Returns:
            List of segmentation results
        """
        try:
            self.logger.info(
                "Starting batch SAM segmentation",
                user_id=user_id,
                batch_size=len(batch_data),
                model_size=model_size
            )
            
            # Add batch IDs and ensure consistent format
            enhanced_batch = []
            for i, item in enumerate(batch_data):
                enhanced_item = {
                    "id": item.get("id", f"batch_{i}"),
                    "image_base64": item["image_base64"],
                    "points": item["points"],
                    "labels": item.get("labels"),
                    "model_size": model_size,
                    "multimask_output": item.get("multimask_output", True),
                    "return_logits": item.get("return_logits", False)
                }
                enhanced_batch.append(enhanced_item)
            
            # Call Modal batch endpoint (if available) or process sequentially
            try:
                # Try batch endpoint first
                batch_url = "https://segment3d-sam-platform--batch-segment.modal.run"
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        batch_url,
                        json={
                            "batch_data": enhanced_batch,
                            "model_size": model_size
                        }
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        self.logger.info(
                            "Batch SAM segmentation completed",
                            user_id=user_id,
                            batch_size=len(results)
                        )
                        return results
                        
            except Exception as batch_error:
                self.logger.warning(
                    "Batch endpoint failed, falling back to sequential processing",
                    error=str(batch_error)
                )
            
            # Fallback: process sequentially
            results = []
            for item in enhanced_batch:
                try:
                    result = await self.segment_image(
                        image_base64=item["image_base64"],
                        points=item["points"],
                        labels=item.get("labels"),
                        model_size=model_size,
                        multimask_output=item.get("multimask_output", True),
                        return_logits=item.get("return_logits", False),
                        user_id=user_id
                    )
                    result["batch_id"] = item["id"]
                    results.append(result)
                    
                except Exception as item_error:
                    results.append({
                        "success": False,
                        "error": str(item_error),
                        "batch_id": item["id"]
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(
                "Batch SAM segmentation failed",
                user_id=user_id,
                error=str(e)
            )
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about available SAM models"""
        try:
            model_info_url = "https://segment3d-sam-platform--get-model-info.modal.run"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(model_info_url)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    # Fallback model info
                    return {
                        "available_models": ["vit_b", "vit_l", "vit_h"],
                        "device": "cuda",
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
                    
        except Exception as e:
            self.logger.error("Failed to get model info", error=str(e))
            raise
    
    async def benchmark_models(self) -> Dict[str, Any]:
        """Benchmark all available SAM models"""
        try:
            benchmark_url = "https://segment3d-sam-platform--benchmark-models.modal.run"
            
            async with httpx.AsyncClient(timeout=180.0) as client:  # Longer timeout for benchmarks
                response = await client.get(benchmark_url)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"Benchmark API error: {response.status_code}")
                    
        except Exception as e:
            self.logger.error("Model benchmark failed", error=str(e))
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check SAM service health"""
        try:
            health_url = "https://segment3d-sam-platform--health-check.modal.run"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(health_url)
                
                if response.status_code == 200:
                    result = response.json()
                    result["api_accessible"] = True
                    return result
                else:
                    return {
                        "status": "unhealthy",
                        "api_accessible": False,
                        "error": f"HTTP {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy", 
                "api_accessible": False,
                "error": str(e)
            } 