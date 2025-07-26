"""
Custom exception classes for Segment3D Platform API
Provides structured error handling
"""

from typing import Any, Dict, Optional


class Segment3DException(Exception):
    """Base exception class for Segment3D Platform"""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        super().__init__(self.message)


class NotFoundError(Segment3DException):
    """Raised when a requested resource is not found"""
    
    def __init__(self, message: str = "Resource not found", **kwargs):
        super().__init__(message, error_code="NOT_FOUND", **kwargs)


class PermissionError(Segment3DException):
    """Raised when user lacks required permissions"""
    
    def __init__(self, message: str = "Permission denied", **kwargs):
        super().__init__(message, error_code="PERMISSION_DENIED", **kwargs)


class ValidationError(Segment3DException):
    """Raised when input validation fails"""
    
    def __init__(self, message: str = "Validation failed", **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)


class AuthenticationError(Segment3DException):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, error_code="AUTHENTICATION_ERROR", **kwargs)


class ExternalServiceError(Segment3DException):
    """Raised when external service call fails"""
    
    def __init__(self, message: str = "External service error", service: str = "unknown", **kwargs):
        details = kwargs.get("details", {})
        details["service"] = service
        super().__init__(message, error_code="EXTERNAL_SERVICE_ERROR", details=details, **kwargs)


class DatabaseError(Segment3DException):
    """Raised when database operation fails"""
    
    def __init__(self, message: str = "Database operation failed", **kwargs):
        super().__init__(message, error_code="DATABASE_ERROR", **kwargs)


class FileProcessingError(Segment3DException):
    """Raised when file processing fails"""
    
    def __init__(self, message: str = "File processing failed", **kwargs):
        super().__init__(message, error_code="FILE_PROCESSING_ERROR", **kwargs)


class SAMError(Segment3DException):
    """Raised when SAM segmentation fails"""
    
    def __init__(self, message: str = "SAM segmentation failed", **kwargs):
        super().__init__(message, error_code="SAM_ERROR", **kwargs) 