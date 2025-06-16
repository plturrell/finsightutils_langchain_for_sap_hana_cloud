"""
Standardized authentication models for the SAP HANA Cloud LangChain Integration API.

This module provides consistent models for authentication, authorization, and
user management across the API.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from .base_standardized import BaseAPIModel


class User(BaseAPIModel):
    """User model for authentication and authorization."""
    
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="Email address")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    is_active: bool = Field(True, description="Whether the user is active")
    is_admin: bool = Field(False, description="Whether the user is an administrator")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    
    class Config:
        """Model configuration."""
        schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "johndoe@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "is_active": True,
                "is_admin": False,
                "roles": ["viewer", "editor"],
                "permissions": ["read:documents", "write:documents"]
            }
        }


class Token(BaseAPIModel):
    """Token model for authentication responses."""
    
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    refresh_token: Optional[str] = Field(None, description="JWT refresh token")
    scope: Optional[str] = Field(None, description="Token scope")
    
    class Config:
        """Model configuration."""
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "scope": "read write"
            }
        }


class TokenPayload(BaseAPIModel):
    """Token payload model for JWT token decoding."""
    
    sub: str = Field(..., description="Subject (username)")
    exp: int = Field(..., description="Expiration timestamp")
    iat: Optional[int] = Field(None, description="Issued at timestamp")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    is_admin: bool = Field(False, description="Whether the user is an administrator")


class APIKey(BaseAPIModel):
    """API key model for API key management."""
    
    key: str = Field(..., description="API key")
    name: str = Field(..., description="API key name")
    created_at: str = Field(..., description="Creation timestamp")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")
    is_active: bool = Field(True, description="Whether the API key is active")
    scopes: List[str] = Field(default_factory=list, description="API key scopes")
    user_id: Optional[str] = Field(None, description="User ID associated with this API key")
    
    class Config:
        """Model configuration."""
        schema_extra = {
            "example": {
                "key": "sk_test_123456789",
                "name": "Production API Key",
                "created_at": "2023-01-01T12:00:00Z",
                "expires_at": "2024-01-01T12:00:00Z",
                "is_active": True,
                "scopes": ["read:vectors", "write:vectors"],
                "user_id": "user_123456789"
            }
        }


class LoginRequest(BaseAPIModel):
    """Login request model for username/password authentication."""
    
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    
    class Config:
        """Model configuration."""
        schema_extra = {
            "example": {
                "username": "johndoe",
                "password": "password123"
            }
        }


class ChangePasswordRequest(BaseAPIModel):
    """Change password request model."""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., description="New password")
    
    @validator("new_password")
    def validate_password_strength(cls, v: str) -> str:
        """Validate password strength."""
        min_length = 8
        if len(v) < min_length:
            raise ValueError(f"Password must be at least {min_length} characters long")
        
        # Check for at least one uppercase letter
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        
        # Check for at least one lowercase letter
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        
        # Check for at least one digit
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        
        return v


class CreateAPIKeyRequest(BaseAPIModel):
    """Request model for creating a new API key."""
    
    name: str = Field(..., description="API key name")
    expires_in_days: Optional[int] = Field(
        None, description="API key expiration in days", ge=1, le=365
    )
    scopes: List[str] = Field(default_factory=list, description="API key scopes")
    
    class Config:
        """Model configuration."""
        schema_extra = {
            "example": {
                "name": "Production API Key",
                "expires_in_days": 365,
                "scopes": ["read:vectors", "write:vectors"]
            }
        }


class ArrowFlightAuthRequest(BaseAPIModel):
    """Authentication request model for Arrow Flight."""
    
    username: str = Field(..., description="Username")
    password: Optional[str] = Field(None, description="Password")
    api_key: Optional[str] = Field(None, description="API key")
    
    @validator("api_key", "password")
    def validate_auth_method(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate that either password or API key is provided."""
        if "username" not in values:
            raise ValueError("Username is required")
        
        if values.get("password") is None and values.get("api_key") is None:
            raise ValueError("Either password or API key must be provided")
        
        return v
    
    class Config:
        """Model configuration."""
        schema_extra = {
            "example": {
                "username": "johndoe",
                "password": "password123"
            }
        }


# Permission models
class Role(BaseAPIModel):
    """Role model for role-based access control."""
    
    name: str = Field(..., description="Role name")
    description: Optional[str] = Field(None, description="Role description")
    permissions: List[str] = Field(default_factory=list, description="Role permissions")
    
    class Config:
        """Model configuration."""
        schema_extra = {
            "example": {
                "name": "editor",
                "description": "Can edit documents and vectors",
                "permissions": ["read:documents", "write:documents", "read:vectors", "write:vectors"]
            }
        }


class Permission(BaseAPIModel):
    """Permission model for permission-based access control."""
    
    name: str = Field(..., description="Permission name")
    description: Optional[str] = Field(None, description="Permission description")
    
    class Config:
        """Model configuration."""
        schema_extra = {
            "example": {
                "name": "read:documents",
                "description": "Can read documents from the database"
            }
        }