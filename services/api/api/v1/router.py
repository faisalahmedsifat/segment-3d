"""
Main API router for Segment3D Platform
Includes all endpoint modules
"""

from fastapi import APIRouter
from api.v1.endpoints import projects, sam

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(sam.router, prefix="/sam", tags=["sam"]) 