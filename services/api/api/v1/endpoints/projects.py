"""
Project management API endpoints
Implements REST API for project operations
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from prisma import Prisma

from core.database import get_database_dependency
from core.auth import get_current_user
from services.project import ProjectService
from schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse
from schemas.user import UserResponse

router = APIRouter()


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    current_user: UserResponse = Depends(get_current_user),
    db: Prisma = Depends(get_database_dependency)
):
    """Create a new project"""
    service = ProjectService(db)
    return await service.create_project(project_data, current_user.id)


@router.get("/", response_model=List[ProjectResponse])
async def get_user_projects(
    include_archived: bool = Query(False, description="Include archived projects"),
    skip: int = Query(0, ge=0, description="Number of projects to skip"),
    take: int = Query(100, ge=1, le=1000, description="Number of projects to take"),
    current_user: UserResponse = Depends(get_current_user),
    db: Prisma = Depends(get_database_dependency)
):
    """Get projects accessible to current user"""
    service = ProjectService(db)
    return await service.get_user_projects(
        current_user.id, include_archived, skip, take
    )


@router.get("/search", response_model=List[ProjectResponse])
async def search_projects(
    q: str = Query(..., min_length=2, description="Search query"),
    skip: int = Query(0, ge=0, description="Number of projects to skip"),
    take: int = Query(100, ge=1, le=1000, description="Number of projects to take"),
    current_user: UserResponse = Depends(get_current_user),
    db: Prisma = Depends(get_database_dependency)
):
    """Search projects by name or description"""
    service = ProjectService(db)
    return await service.search_projects(q, current_user.id, skip, take)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: UserResponse = Depends(get_current_user),
    db: Prisma = Depends(get_database_dependency)
):
    """Get project by ID"""
    service = ProjectService(db)
    return await service.get_project(project_id, current_user.id)


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    current_user: UserResponse = Depends(get_current_user),
    db: Prisma = Depends(get_database_dependency)
):
    """Update project"""
    service = ProjectService(db)
    return await service.update_project(project_id, project_data, current_user.id)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    current_user: UserResponse = Depends(get_current_user),
    db: Prisma = Depends(get_database_dependency)
):
    """Archive project (soft delete)"""
    service = ProjectService(db)
    await service.delete_project(project_id, current_user.id)


@router.get("/{project_id}/stats")
async def get_project_statistics(
    project_id: str,
    current_user: UserResponse = Depends(get_current_user),
    db: Prisma = Depends(get_database_dependency)
):
    """Get comprehensive project statistics"""
    service = ProjectService(db)
    return await service.get_project_statistics(project_id, current_user.id)


@router.post("/{project_id}/members")
async def add_project_member(
    project_id: str,
    member_user_id: str,
    role: str = "ANNOTATOR",
    current_user: UserResponse = Depends(get_current_user),
    db: Prisma = Depends(get_database_dependency)
):
    """Add member to project"""
    service = ProjectService(db)
    await service.add_project_member(project_id, current_user.id, member_user_id, role)
    return {"message": "Member added successfully"} 