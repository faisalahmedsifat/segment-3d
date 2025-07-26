"""
Pydantic schemas for Project operations
Handles request/response validation and serialization
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ProjectType(str, Enum):
    """Project type enumeration"""
    MIXED = "MIXED"
    IMAGE_2D = "IMAGE_2D"
    MODEL_3D = "MODEL_3D"


class ProjectCreate(BaseModel):
    """Schema for creating a new project"""
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, max_length=1000, description="Project description")
    type: ProjectType = Field(default=ProjectType.MIXED, description="Project type")
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Project settings")


class ProjectUpdate(BaseModel):
    """Schema for updating a project"""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, max_length=1000, description="Project description")
    type: Optional[ProjectType] = Field(None, description="Project type")
    settings: Optional[Dict[str, Any]] = Field(None, description="Project settings")


class ProjectOwner(BaseModel):
    """Schema for project owner information"""
    id: str
    username: str
    email: str
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class ProjectStats(BaseModel):
    """Schema for project statistics"""
    totalFiles: int = 0
    totalAnnotations: int = 0
    completedTasks: int = 0
    pendingTasks: int = 0
    avgQualityScore: Optional[float] = None
    lastUpdated: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ProjectResponse(BaseModel):
    """Schema for project response"""
    id: str
    name: str
    description: Optional[str] = None
    type: ProjectType
    settings: Dict[str, Any] = Field(default_factory=dict)
    isArchived: bool = False
    createdAt: datetime
    updatedAt: datetime
    owner: ProjectOwner
    stats: Optional[ProjectStats] = None
    
    # Additional computed fields
    datasetsCount: Optional[int] = None
    tasksCount: Optional[int] = None
    membersCount: Optional[int] = None
    
    model_config = ConfigDict(from_attributes=True)


class ProjectSummary(BaseModel):
    """Schema for project summary (lightweight version)"""
    id: str
    name: str
    description: Optional[str] = None
    type: ProjectType
    isArchived: bool = False
    createdAt: datetime
    updatedAt: datetime
    ownerName: str
    datasetsCount: int = 0
    tasksCount: int = 0
    
    model_config = ConfigDict(from_attributes=True) 