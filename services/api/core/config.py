"""
Configuration management for Segment3D Platform API
Uses Pydantic settings for type-safe configuration
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Basic app config
    APP_NAME: str = "Segment3D Platform API"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    
    # Database
    DATABASE_URL: str = Field(..., description="PostgreSQL database URL")
    DATABASE_ECHO: bool = False
    
    # Security
    SECRET_KEY: str = Field(..., description="Secret key for JWT tokens")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    
    # File Storage
    STORAGE_TYPE: str = "local"  # "local" or "s3"
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # AWS S3 (if using S3 storage)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: Optional[str] = None
    
    # Redis (for caching and sessions)
    REDIS_URL: str = "redis://localhost:6379"
    
    # Modal AI
    MODAL_TOKEN_ID: Optional[str] = None
    MODAL_TOKEN_SECRET: Optional[str] = None
    
    # Background Jobs
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


class DevelopmentSettings(Settings):
    """Development environment settings"""
    DEBUG: bool = True
    DATABASE_ECHO: bool = True
    LOG_LEVEL: str = "DEBUG"


class ProductionSettings(Settings):
    """Production environment settings"""
    DEBUG: bool = False
    DATABASE_ECHO: bool = False
    LOG_LEVEL: str = "WARNING"


class TestSettings(Settings):
    """Test environment settings"""
    DATABASE_URL: str = "sqlite:///./test.db"
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"


@lru_cache()
def get_settings() -> Settings:
    """Get application settings based on environment"""
    import os
    
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "test":
        return TestSettings()
    else:
        return DevelopmentSettings() 