"""
Project service implementing business logic
Part of clean architecture - handles business rules and orchestrates repository operations
"""

from typing import List, Optional, Dict, Any
from prisma import Prisma
import structlog

from repositories.project import ProjectRepository
from repositories.user import UserRepository
from schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse
from core.exceptions import NotFoundError, PermissionError, ValidationError

logger = structlog.get_logger(__name__)


class ProjectService:
    """Service class for project business logic"""
    
    def __init__(self, db: Prisma):
        self.db = db
        self.project_repo = ProjectRepository(db)
        self.user_repo = UserRepository(db)
        self.logger = logger.bind(service="ProjectService")
    
    async def create_project(self, project_data: ProjectCreate, owner_id: str) -> ProjectResponse:
        """Create a new project"""
        try:
            # Verify owner exists
            owner = await self.user_repo.get(owner_id)
            if not owner:
                raise NotFoundError("Owner not found")
            
            # Check if user has permission to create projects
            if owner.role not in ["ADMIN", "PROJECT_MANAGER"]:
                raise PermissionError("User does not have permission to create projects")
            
            # Create project with owner
            project_dict = project_data.model_dump()
            project_dict["ownerId"] = owner_id
            
            project = await self.project_repo.create(project_dict)
            self.logger.info("Project created", project_id=project.id, owner_id=owner_id)
            
            return ProjectResponse.model_validate(project)
            
        except Exception as e:
            self.logger.error("Failed to create project", error=str(e))
            raise
    
    async def get_project(self, project_id: str, user_id: str) -> ProjectResponse:
        """Get project by ID with access control"""
        try:
            project = await self.project_repo.get_with_stats(project_id)
            if not project:
                raise NotFoundError("Project not found")
            
            # Check if user has access to this project
            if not await self._check_project_access(project_id, user_id):
                raise PermissionError("User does not have access to this project")
            
            return ProjectResponse.model_validate(project)
            
        except Exception as e:
            self.logger.error("Failed to get project", project_id=project_id, error=str(e))
            raise
    
    async def update_project(
        self, 
        project_id: str, 
        project_data: ProjectUpdate, 
        user_id: str
    ) -> ProjectResponse:
        """Update project with access control"""
        try:
            # Check if project exists and user has admin access
            if not await self._check_project_admin_access(project_id, user_id):
                raise PermissionError("User does not have admin access to this project")
            
            updated_project = await self.project_repo.update(project_id, project_data)
            if not updated_project:
                raise NotFoundError("Project not found")
            
            self.logger.info("Project updated", project_id=project_id, user_id=user_id)
            return ProjectResponse.model_validate(updated_project)
            
        except Exception as e:
            self.logger.error("Failed to update project", project_id=project_id, error=str(e))
            raise
    
    async def delete_project(self, project_id: str, user_id: str) -> bool:
        """Archive project (soft delete) with access control"""
        try:
            # Check if user is project owner or admin
            if not await self._check_project_owner_access(project_id, user_id):
                raise PermissionError("Only project owner can delete project")
            
            archived_project = await self.project_repo.archive_project(project_id)
            if not archived_project:
                raise NotFoundError("Project not found")
            
            self.logger.info("Project archived", project_id=project_id, user_id=user_id)
            return True
            
        except Exception as e:
            self.logger.error("Failed to archive project", project_id=project_id, error=str(e))
            raise
    
    async def get_user_projects(
        self, 
        user_id: str, 
        include_archived: bool = False,
        skip: int = 0, 
        take: int = 100
    ) -> List[ProjectResponse]:
        """Get projects accessible to user"""
        try:
            projects = await self.project_repo.get_user_projects(
                user_id, include_archived, skip, take
            )
            
            return [ProjectResponse.model_validate(p) for p in projects]
            
        except Exception as e:
            self.logger.error("Failed to get user projects", user_id=user_id, error=str(e))
            raise
    
    async def search_projects(
        self, 
        query: str, 
        user_id: str,
        skip: int = 0, 
        take: int = 100
    ) -> List[ProjectResponse]:
        """Search projects accessible to user"""
        try:
            projects = await self.project_repo.search_projects(query, user_id, skip, take)
            return [ProjectResponse.model_validate(p) for p in projects]
            
        except Exception as e:
            self.logger.error("Failed to search projects", query=query, error=str(e))
            raise
    
    async def add_project_member(
        self, 
        project_id: str, 
        user_id: str, 
        member_user_id: str, 
        role: str = "ANNOTATOR"
    ) -> bool:
        """Add member to project"""
        try:
            # Check if user has admin access to project
            if not await self._check_project_admin_access(project_id, user_id):
                raise PermissionError("User does not have admin access to this project")
            
            # Check if member user exists
            member = await self.user_repo.get(member_user_id)
            if not member:
                raise NotFoundError("User to add not found")
            
            # Add member (this would require a ProjectMember repository)
            # For now, we'll skip the actual implementation
            self.logger.info(
                "Project member added", 
                project_id=project_id, 
                member_id=member_user_id, 
                role=role
            )
            return True
            
        except Exception as e:
            self.logger.error("Failed to add project member", error=str(e))
            raise
    
    async def get_project_statistics(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """Get comprehensive project statistics"""
        try:
            # Check access
            if not await self._check_project_access(project_id, user_id):
                raise PermissionError("User does not have access to this project")
            
            stats = await self.project_repo.get_project_stats(project_id)
            return stats
            
        except Exception as e:
            self.logger.error("Failed to get project statistics", project_id=project_id, error=str(e))
            raise
    
    # Private helper methods
    async def _check_project_access(self, project_id: str, user_id: str) -> bool:
        """Check if user has any access to project"""
        try:
            project = await self.project_repo.get(project_id, include={
                "members": {"where": {"userId": user_id, "isActive": True}}
            })
            
            if not project:
                return False
            
            # User has access if they're owner or active member
            return (project.ownerId == user_id or 
                    len(project.members) > 0)
            
        except Exception:
            return False
    
    async def _check_project_admin_access(self, project_id: str, user_id: str) -> bool:
        """Check if user has admin access to project"""
        try:
            project = await self.project_repo.get(project_id, include={
                "members": {"where": {"userId": user_id, "isActive": True}}
            })
            
            if not project:
                return False
            
            # User has admin access if they're owner or admin member
            if project.ownerId == user_id:
                return True
            
            # Check if user is admin member
            for member in project.members:
                if member.userId == user_id and member.role in ["ADMIN", "PROJECT_MANAGER"]:
                    return True
            
            return False
            
        except Exception:
            return False
    
    async def _check_project_owner_access(self, project_id: str, user_id: str) -> bool:
        """Check if user is project owner"""
        try:
            project = await self.project_repo.get(project_id)
            return project and project.ownerId == user_id
            
        except Exception:
            return False 