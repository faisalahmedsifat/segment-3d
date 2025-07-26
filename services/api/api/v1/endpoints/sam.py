"""
SAM (Segment Anything Model) API endpoints
Handles 3D and 2D segmentation requests via Modal
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from prisma import Prisma

from core.database import get_database_dependency
from core.auth import get_current_user
from schemas.user import UserResponse
from services.sam import SAMService

router = APIRouter()


class SegmentationRequest(BaseModel):
    """Schema for segmentation request"""
    image_base64: str = Field(..., description="Base64 encoded image")
    points: List[List[float]] = Field(..., description="List of [x, y] click coordinates")
    labels: Optional[List[int]] = Field(None, description="List of 1s (positive) or 0s (negative) for each point")
    model_size: str = Field("vit_b", description="SAM model size: vit_b, vit_l, or vit_h")
    multimask_output: bool = Field(True, description="Whether to return multiple mask candidates")
    return_logits: bool = Field(False, description="Whether to return mask logits")


class BatchSegmentationRequest(BaseModel):
    """Schema for batch segmentation request"""
    requests: List[Dict[str, Any]] = Field(..., description="List of segmentation requests")
    model_size: str = Field("vit_b", description="SAM model size for all requests")


@router.post("/segment")
async def segment_image(
    request: SegmentationRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: Prisma = Depends(get_database_dependency)
):
    """
    Segment image using SAM model
    
    Processes a single image with click points and returns segmentation masks
    """
    try:
        service = SAMService(db)
        
        result = await service.segment_image(
            image_base64=request.image_base64,
            points=request.points,
            labels=request.labels,
            model_size=request.model_size,
            multimask_output=request.multimask_output,
            return_logits=request.return_logits,
            user_id=current_user.id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}"
        )


@router.post("/batch-segment")
async def batch_segment_images(
    request: BatchSegmentationRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: Prisma = Depends(get_database_dependency)
):
    """
    Process multiple segmentation requests in batch
    
    Efficiently processes multiple images with SAM
    """
    try:
        service = SAMService(db)
        
        results = await service.batch_segment(
            batch_data=request.requests,
            model_size=request.model_size,
            user_id=current_user.id
        )
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch segmentation failed: {str(e)}"
        )


@router.get("/models")
async def get_available_models(
    current_user: UserResponse = Depends(get_current_user),
    db: Prisma = Depends(get_database_dependency)
):
    """
    Get information about available SAM models
    
    Returns details about model sizes, performance characteristics
    """
    try:
        service = SAMService(db)
        return await service.get_model_info()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get("/benchmark")
async def benchmark_models(
    current_user: UserResponse = Depends(get_current_user),
    db: Prisma = Depends(get_database_dependency)
):
    """
    Benchmark all available SAM models
    
    Returns performance metrics for model selection
    """
    try:
        service = SAMService(db)
        return await service.benchmark_models()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model benchmark failed: {str(e)}"
        )


@router.get("/health")
async def sam_health_check():
    """
    Check SAM service health
    
    Verifies that Modal SAM deployment is accessible
    """
    try:
        service = SAMService(None)  # Health check doesn't need DB
        return await service.health_check()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"SAM service unhealthy: {str(e)}"
        ) 