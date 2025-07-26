"""
Project repository for database operations
Implements repository pattern for Project model
"""

from typing import Optional, List
from prisma.models import Project
from prisma import Prisma

from repositories.base import BaseRepository
from schemas.project import ProjectCreate, ProjectUpdate


class ProjectRepository(BaseRepository[Project, ProjectCreate, ProjectUpdate]):
    """Repository for Project model operations"""
    
    @property
    def model_class(self):
        return self.db.project
    
    async def get_with_stats(self, project_id: str) -> Optional[Project]:
        """Get project with statistics"""
        return await self.get(
            project_id,
            include={
                "owner": True,
                "stats": True,
                "members": {
                    "include": {"user": True}
                },
                "_count": {
                    "datasets": True,
                    "tasks": True,
                    "labelSchema": True
                }
            }
        )
    
    async def get_user_projects(
        self, 
        user_id: str, 
        include_archived: bool = False,
        skip: int = 0, 
        take: int = 100
    ) -> List[Project]:
        """Get projects where user is owner or member"""
        where_clause = {
            "OR": [
                {"ownerId": user_id},
                {"members": {"some": {"userId": user_id, "isActive": True}}}
            ]
        }
        
        if not include_archived:
            where_clause["isArchived"] = False
        
        return await self.get_many(
            skip=skip,
            take=take,
            where=where_clause,
            include={
                "owner": True,
                "stats": True,
                "_count": {
                    "datasets": True,
                    "tasks": True,
                    "members": True
                }
            }
        )
    
    async def get_owned_projects(
        self, 
        owner_id: str, 
        include_archived: bool = False,
        skip: int = 0, 
        take: int = 100
    ) -> List[Project]:
        """Get projects owned by a specific user"""
        where_clause = {"ownerId": owner_id}
        
        if not include_archived:
            where_clause["isArchived"] = False
        
        return await self.get_many(
            skip=skip,
            take=take,
            where=where_clause,
            include={
                "stats": True,
                "_count": {
                    "datasets": True,
                    "tasks": True,
                    "members": True
                }
            }
        )
    
    async def search_projects(
        self, 
        query: str, 
        user_id: Optional[str] = None,
        skip: int = 0, 
        take: int = 100
    ) -> List[Project]:
        """Search projects by name or description"""
        where_clause = {
            "AND": [
                {
                    "OR": [
                        {"name": {"contains": query, "mode": "insensitive"}},
                        {"description": {"contains": query, "mode": "insensitive"}},
                    ]
                },
                {"isArchived": False}
            ]
        }
        
        # If user_id provided, only search projects user has access to
        if user_id:
            where_clause["AND"].append({
                "OR": [
                    {"ownerId": user_id},
                    {"members": {"some": {"userId": user_id, "isActive": True}}}
                ]
            })
        
        return await self.get_many(
            skip=skip,
            take=take,
            where=where_clause,
            include={
                "owner": True,
                "_count": {
                    "datasets": True,
                    "tasks": True
                }
            }
        )
    
    async def get_projects_by_type(
        self, 
        project_type: str,
        skip: int = 0, 
        take: int = 100
    ) -> List[Project]:
        """Get projects by type (MIXED, IMAGE_2D, MODEL_3D)"""
        return await self.get_many(
            skip=skip,
            take=take,
            where={
                "type": project_type,
                "isArchived": False
            },
            include={
                "owner": True,
                "_count": {
                    "datasets": True,
                    "tasks": True
                }
            }
        )
    
    async def archive_project(self, project_id: str) -> Optional[Project]:
        """Archive a project instead of deleting"""
        return await self.update(project_id, {"isArchived": True})
    
    async def unarchive_project(self, project_id: str) -> Optional[Project]:
        """Unarchive a project"""
        return await self.update(project_id, {"isArchived": False})
    
    async def get_project_stats(self, project_id: str) -> dict:
        """Get comprehensive project statistics"""
        try:
            # Get basic counts
            datasets_count = await self.db.dataset.count(where={"projectId": project_id})
            tasks_count = await self.db.task.count(where={"projectId": project_id})
            completed_tasks = await self.db.task.count(
                where={"projectId": project_id, "status": "COMPLETED"}
            )
            annotations_count = await self.db.annotation.count(
                where={"task": {"projectId": project_id}}
            )
            
            # Get file statistics
            files_result = await self.db.file.aggregate(
                where={"dataset": {"projectId": project_id}},
                _count=True,
                _sum={"size": True}
            )
            
            return {
                "datasets": datasets_count,
                "tasks": tasks_count,
                "completed_tasks": completed_tasks,
                "pending_tasks": tasks_count - completed_tasks,
                "annotations": annotations_count,
                "files": files_result["_count"] or 0,
                "total_size": files_result["_sum"]["size"] or 0,
                "completion_rate": (completed_tasks / tasks_count * 100) if tasks_count > 0 else 0
            }
        except Exception as e:
            self.logger.error("Failed to get project stats", project_id=project_id, error=str(e))
            return {} 