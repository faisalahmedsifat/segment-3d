"""
FastAPI main application for Segment3D Platform
Implements clean architecture with service repository pattern
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import structlog

from core.config import get_settings
from api.v1.router import api_router
from core.database import init_db
from core.logging import setup_logging

# Setup structured logging
setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting Segment3D API...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    yield
    
    logger.info("Shutting down Segment3D API...")


# Create FastAPI app
app = FastAPI(
    title="Segment3D Platform API",
    description="SAM-Powered 3D/2D Annotation Platform - Backend API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Get settings
settings = get_settings()

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS
)

# CORS middleware  
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "segment3d-api",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Segment3D Platform API",
        "docs": "/docs",
        "health": "/health"
    } 