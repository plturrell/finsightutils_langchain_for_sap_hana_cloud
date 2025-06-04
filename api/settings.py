"""
Settings module for platform-specific configurations.

This module provides settings models for different deployment platforms
and a function to get the appropriate settings based on the current environment.
"""

import os
from typing import Dict, List, Optional, Any, Union
from functools import lru_cache
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Base settings model with common configurations."""
    
    # Basic information
    app_name: str = "SAP HANA Cloud LangChain Integration"
    version: str = "0.2.1"
    environment: str = "development"
    backend_platform: str = "local"
    
    # Database configuration
    hana_host: str = ""
    hana_port: int = 443
    hana_user: str = ""
    hana_password: str = ""
    hana_encrypt: bool = True
    hana_ssl_validate: bool = True
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    enable_cors: bool = True
    cors_origins: List[str] = ["*"]
    
    # Embedding configuration
    embedding_model_name: str = "all-MiniLM-L6-v2"
    use_internal_embedding: bool = False
    internal_embedding_model_id: str = "SAP_NEB.20240715"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class TogetherAISettings(Settings):
    """Settings for Together.ai platform."""
    
    backend_platform: str = "together_ai"
    
    # Together.ai specific settings
    together_api_key: str = ""
    together_model_name: str = "togethercomputer/llama-2-7b"
    together_api_base: str = "https://api.together.xyz/v1"
    use_together_embeddings: bool = True
    
    class Config:
        env_prefix = ""


class NvidiaLaunchPadSettings(Settings):
    """Settings for NVIDIA LaunchPad platform."""
    
    backend_platform: str = "nvidia_launchpad"
    
    # NVIDIA LaunchPad specific settings
    triton_server_url: str = "localhost:8000"
    use_triton_inference: bool = True
    triton_model_name: str = "all-MiniLM-L6-v2"
    triton_timeout: int = 60
    enable_tensorrt: bool = True
    
    class Config:
        env_prefix = ""


class SAPBTPSettings(Settings):
    """Settings for SAP BTP platform."""
    
    backend_platform: str = "sap_btp"
    
    # SAP BTP specific settings
    cf_api_url: str = ""
    cf_org: str = ""
    cf_space: str = ""
    xsuaa_url: str = ""
    destination_service: str = ""
    use_destination_service: bool = False
    
    class Config:
        env_prefix = ""


class VercelSettings(Settings):
    """Settings for Vercel platform."""
    
    backend_platform: str = "vercel"
    
    # Vercel specific settings
    vercel_region: str = ""
    vercel_edge_config: str = ""
    use_vercel_edge: bool = False
    use_vercel_kv: bool = False
    vercel_cron_enabled: bool = False
    
    class Config:
        env_prefix = ""


@lru_cache()
def get_settings():
    """
    Get platform-specific settings based on the environment.
    
    Returns:
        Union[TogetherAISettings, NvidiaLaunchPadSettings, SAPBTPSettings, VercelSettings, Settings]:
            Platform-specific settings model.
    """
    # Detect platform from environment variable
    platform = os.getenv("BACKEND_PLATFORM", "local").lower()
    
    if platform == "together_ai":
        return TogetherAISettings()
    elif platform == "nvidia_launchpad":
        return NvidiaLaunchPadSettings()
    elif platform == "sap_btp":
        return SAPBTPSettings()
    elif platform == "vercel":
        return VercelSettings()
    else:
        return Settings()