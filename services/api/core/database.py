"""
Database configuration and connection management
Uses Prisma ORM for database operations
"""

from prisma import Prisma
from prisma.models import User, Project, Dataset, File, Annotation
import structlog
from typing import Optional

from core.config import get_settings

logger = structlog.get_logger(__name__)

# Global Prisma client instance
prisma_client: Optional[Prisma] = None


async def get_db() -> Prisma:
    """Get database client instance"""
    global prisma_client
    
    if prisma_client is None:
        prisma_client = Prisma()
        await prisma_client.connect()
        logger.info("Database connection established")
    
    return prisma_client


async def init_db():
    """Initialize database connection"""
    settings = get_settings()
    
    try:
        db = await get_db()
        logger.info("Database initialized successfully")
        return db
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise


async def close_db():
    """Close database connection"""
    global prisma_client
    
    if prisma_client:
        await prisma_client.disconnect()
        prisma_client = None
        logger.info("Database connection closed")


# Database dependency for FastAPI
async def get_database_dependency() -> Prisma:
    """FastAPI dependency to get database client"""
    return await get_db() 