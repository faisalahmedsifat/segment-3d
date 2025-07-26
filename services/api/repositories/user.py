"""
User repository for database operations
Implements repository pattern for User model
"""

from typing import Optional, List
from prisma.models import User
from prisma import Prisma

from repositories.base import BaseRepository
from schemas.user import UserCreate, UserUpdate


class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    """Repository for User model operations"""
    
    @property
    def model_class(self):
        return self.db.user
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address"""
        return await self.get_by_field("email", email)
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return await self.get_by_field("username", username)
    
    async def get_active_users(self, skip: int = 0, take: int = 100) -> List[User]:
        """Get all active users"""
        return await self.get_many(
            skip=skip,
            take=take,
            where={"isActive": True}
        )
    
    async def get_users_by_role(self, role: str, skip: int = 0, take: int = 100) -> List[User]:
        """Get users by role"""
        return await self.get_many(
            skip=skip,
            take=take,
            where={"role": role}
        )
    
    async def search_users(self, query: str, skip: int = 0, take: int = 100) -> List[User]:
        """Search users by username, email, or name"""
        return await self.get_many(
            skip=skip,
            take=take,
            where={
                "OR": [
                    {"username": {"contains": query, "mode": "insensitive"}},
                    {"email": {"contains": query, "mode": "insensitive"}},
                    {"firstName": {"contains": query, "mode": "insensitive"}},
                    {"lastName": {"contains": query, "mode": "insensitive"}},
                ]
            }
        )
    
    async def deactivate_user(self, user_id: str) -> Optional[User]:
        """Deactivate a user instead of deleting"""
        return await self.update(user_id, {"isActive": False})
    
    async def activate_user(self, user_id: str) -> Optional[User]:
        """Activate a deactivated user"""
        return await self.update(user_id, {"isActive": True}) 