"""
Pydantic schemas for User operations
Handles request/response validation and serialization
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from enum import Enum


class UserRole(str, Enum):
    """User role enumeration"""
    ADMIN = "ADMIN"
    PROJECT_MANAGER = "PROJECT_MANAGER"
    ANNOTATOR = "ANNOTATOR"
    REVIEWER = "REVIEWER"
    VIEWER = "VIEWER"


class UserCreate(BaseModel):
    """Schema for creating a new user"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    firstName: Optional[str] = Field(None, max_length=100)
    lastName: Optional[str] = Field(None, max_length=100)
    password: str = Field(..., min_length=8, max_length=128)
    role: UserRole = Field(default=UserRole.ANNOTATOR)


class UserUpdate(BaseModel):
    """Schema for updating a user"""
    firstName: Optional[str] = Field(None, max_length=100)
    lastName: Optional[str] = Field(None, max_length=100)
    avatar: Optional[str] = Field(None, max_length=500)
    role: Optional[UserRole] = None
    isActive: Optional[bool] = None


class UserResponse(BaseModel):
    """Schema for user response (excludes sensitive data)"""
    id: str
    email: str
    username: str
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    avatar: Optional[str] = None
    role: UserRole
    isActive: bool
    createdAt: datetime
    updatedAt: datetime
    
    model_config = ConfigDict(from_attributes=True)


class UserLogin(BaseModel):
    """Schema for user login"""
    username: str
    password: str


class UserRegister(BaseModel):
    """Schema for user registration"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    firstName: Optional[str] = Field(None, max_length=100)
    lastName: Optional[str] = Field(None, max_length=100)
    password: str = Field(..., min_length=8, max_length=128)
    confirmPassword: str = Field(..., min_length=8, max_length=128)

    def validate_passwords_match(self):
        """Validate that passwords match"""
        if self.password != self.confirmPassword:
            raise ValueError("Passwords do not match")
        return self


class TokenResponse(BaseModel):
    """Schema for authentication token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse 