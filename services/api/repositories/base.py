"""
Base repository class for implementing repository pattern
Provides common CRUD operations with Prisma ORM
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Dict, Any
from prisma import Prisma
import structlog

logger = structlog.get_logger(__name__)

# Generic type for model
ModelType = TypeVar('ModelType')
CreateSchemaType = TypeVar('CreateSchemaType')
UpdateSchemaType = TypeVar('UpdateSchemaType')


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType], ABC):
    """Base repository class with common CRUD operations"""
    
    def __init__(self, db: Prisma):
        self.db = db
        self.logger = logger.bind(repository=self.__class__.__name__)
    
    @property
    @abstractmethod
    def model_class(self):
        """Return the Prisma model class"""
        pass
    
    async def get(self, id: str, include: Optional[Dict[str, Any]] = None) -> Optional[ModelType]:
        """Get a single record by ID"""
        try:
            return await self.model_class.find_unique(
                where={"id": id},
                include=include
            )
        except Exception as e:
            self.logger.error("Failed to get record", id=id, error=str(e))
            raise
    
    async def get_many(
        self, 
        skip: int = 0, 
        take: int = 100,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[Dict[str, Any]] = None,
        order_by: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """Get multiple records with pagination and filtering"""
        try:
            return await self.model_class.find_many(
                skip=skip,
                take=take,
                where=where or {},
                include=include,
                order=order_by
            )
        except Exception as e:
            self.logger.error("Failed to get records", error=str(e))
            raise
    
    async def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        """Count records matching criteria"""
        try:
            return await self.model_class.count(where=where or {})
        except Exception as e:
            self.logger.error("Failed to count records", error=str(e))
            raise
    
    async def create(self, data: CreateSchemaType, include: Optional[Dict[str, Any]] = None) -> ModelType:
        """Create a new record"""
        try:
            # Convert Pydantic model to dict if needed
            if hasattr(data, 'model_dump'):
                create_data = data.model_dump(exclude_unset=True)
            else:
                create_data = data
            
            return await self.model_class.create(
                data=create_data,
                include=include
            )
        except Exception as e:
            self.logger.error("Failed to create record", data=data, error=str(e))
            raise
    
    async def update(
        self, 
        id: str, 
        data: UpdateSchemaType, 
        include: Optional[Dict[str, Any]] = None
    ) -> Optional[ModelType]:
        """Update a record by ID"""
        try:
            # Convert Pydantic model to dict if needed
            if hasattr(data, 'model_dump'):
                update_data = data.model_dump(exclude_unset=True)
            else:
                update_data = data
            
            return await self.model_class.update(
                where={"id": id},
                data=update_data,
                include=include
            )
        except Exception as e:
            self.logger.error("Failed to update record", id=id, data=data, error=str(e))
            raise
    
    async def delete(self, id: str) -> bool:
        """Delete a record by ID"""
        try:
            deleted = await self.model_class.delete(where={"id": id})
            return deleted is not None
        except Exception as e:
            self.logger.error("Failed to delete record", id=id, error=str(e))
            raise
    
    async def exists(self, id: str) -> bool:
        """Check if a record exists by ID"""
        try:
            record = await self.model_class.find_unique(where={"id": id})
            return record is not None
        except Exception as e:
            self.logger.error("Failed to check record existence", id=id, error=str(e))
            raise
    
    async def get_by_field(
        self, 
        field: str, 
        value: Any, 
        include: Optional[Dict[str, Any]] = None
    ) -> Optional[ModelType]:
        """Get a record by a specific field"""
        try:
            return await self.model_class.find_first(
                where={field: value},
                include=include
            )
        except Exception as e:
            self.logger.error("Failed to get record by field", field=field, value=value, error=str(e))
            raise
    
    async def get_many_by_field(
        self, 
        field: str, 
        value: Any,
        skip: int = 0,
        take: int = 100,
        include: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """Get multiple records by a specific field"""
        try:
            return await self.model_class.find_many(
                where={field: value},
                skip=skip,
                take=take,
                include=include
            )
        except Exception as e:
            self.logger.error("Failed to get records by field", field=field, value=value, error=str(e))
            raise 