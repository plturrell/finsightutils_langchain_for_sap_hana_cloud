"""
RateLimitMiddleware for the SAP HANA LangChain Integration API.

This middleware implements rate limiting to protect the API from abuse and ensure
fair usage among clients. It supports different rate limiting strategies and can
be configured per-endpoint or globally.
"""

import time
import logging
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Union, Callable, Tuple

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from ..config_standardized import get_standardized_settings
from ..utils.standardized_exceptions import RateLimitExceededException

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)


class RateLimiter:
    """Base class for rate limiters."""
    
    def __init__(self):
        """Initialize the rate limiter."""
        pass
    
    def is_allowed(self, key: str) -> Tuple[bool, Dict[str, Union[int, float]]]:
        """
        Check if a request is allowed based on the rate limit.
        
        Args:
            key: Identifier for the client
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        raise NotImplementedError("Subclasses must implement this method")


class FixedWindowRateLimiter(RateLimiter):
    """Fixed window rate limiter."""
    
    def __init__(self, requests_per_window: int, window_size: int):
        """
        Initialize the fixed window rate limiter.
        
        Args:
            requests_per_window: Maximum number of requests allowed in the window
            window_size: Window size in seconds
        """
        super().__init__()
        self.requests_per_window = requests_per_window
        self.window_size = window_size
        self.windows: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.last_cleanup = time.time()
    
    def is_allowed(self, key: str) -> Tuple[bool, Dict[str, Union[int, float]]]:
        """
        Check if a request is allowed based on the rate limit.
        
        Args:
            key: Identifier for the client
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        # Get current time and window
        now = time.time()
        current_window = int(now / self.window_size)
        
        # Clean up old windows every 10 minutes
        if now - self.last_cleanup > 600:
            self._cleanup_old_windows(current_window)
            self.last_cleanup = now
        
        # Check if the request is allowed
        count = self.windows[key][current_window]
        is_allowed = count < self.requests_per_window
        
        # Update the count if allowed
        if is_allowed:
            self.windows[key][current_window] += 1
        
        # Calculate remaining requests and reset time
        remaining = max(0, self.requests_per_window - count)
        reset = (current_window + 1) * self.window_size
        
        # Return result and rate limit info
        return is_allowed, {
            "limit": self.requests_per_window,
            "remaining": remaining if is_allowed else 0,
            "reset": reset,
            "window_size": self.window_size,
        }
    
    def _cleanup_old_windows(self, current_window: int) -> None:
        """
        Clean up old windows to prevent memory leaks.
        
        Args:
            current_window: Current window index
        """
        for key in list(self.windows.keys()):
            windows = self.windows[key]
            for window in list(windows.keys()):
                if window < current_window - 1:
                    del windows[window]
            if not windows:
                del self.windows[key]


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter."""
    
    def __init__(self, requests_per_window: int, window_size: int):
        """
        Initialize the sliding window rate limiter.
        
        Args:
            requests_per_window: Maximum number of requests allowed in the window
            window_size: Window size in seconds
        """
        super().__init__()
        self.requests_per_window = requests_per_window
        self.window_size = window_size
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.last_cleanup = time.time()
    
    def is_allowed(self, key: str) -> Tuple[bool, Dict[str, Union[int, float]]]:
        """
        Check if a request is allowed based on the rate limit.
        
        Args:
            key: Identifier for the client
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        # Get current time
        now = time.time()
        
        # Clean up old requests every 10 minutes
        if now - self.last_cleanup > 600:
            self._cleanup_old_requests(now)
            self.last_cleanup = now
        
        # Remove requests outside the current window
        window_start = now - self.window_size
        self.requests[key] = [ts for ts in self.requests[key] if ts > window_start]
        
        # Check if the request is allowed
        count = len(self.requests[key])
        is_allowed = count < self.requests_per_window
        
        # Update the count if allowed
        if is_allowed:
            self.requests[key].append(now)
        
        # Calculate remaining requests and reset time
        remaining = max(0, self.requests_per_window - count)
        
        # Calculate reset time (when the oldest request will expire)
        if count > 0 and not is_allowed:
            reset = self.requests[key][0] + self.window_size
        else:
            reset = now + self.window_size
        
        # Return result and rate limit info
        return is_allowed, {
            "limit": self.requests_per_window,
            "remaining": remaining if is_allowed else 0,
            "reset": reset,
            "window_size": self.window_size,
        }
    
    def _cleanup_old_requests(self, now: float) -> None:
        """
        Clean up old requests to prevent memory leaks.
        
        Args:
            now: Current time
        """
        window_start = now - self.window_size
        for key in list(self.requests.keys()):
            self.requests[key] = [ts for ts in self.requests[key] if ts > window_start]
            if not self.requests[key]:
                del self.requests[key]


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter."""
    
    def __init__(self, tokens_per_second: float, max_tokens: int):
        """
        Initialize the token bucket rate limiter.
        
        Args:
            tokens_per_second: Number of tokens added per second
            max_tokens: Maximum number of tokens in the bucket
        """
        super().__init__()
        self.tokens_per_second = tokens_per_second
        self.max_tokens = max_tokens
        self.buckets: Dict[str, Tuple[float, float]] = {}  # (tokens, last_update)
        self.last_cleanup = time.time()
    
    def is_allowed(self, key: str) -> Tuple[bool, Dict[str, Union[int, float]]]:
        """
        Check if a request is allowed based on the rate limit.
        
        Args:
            key: Identifier for the client
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        # Get current time
        now = time.time()
        
        # Clean up old buckets every 10 minutes
        if now - self.last_cleanup > 600:
            self._cleanup_old_buckets(now)
            self.last_cleanup = now
        
        # Initialize bucket if not exists
        if key not in self.buckets:
            self.buckets[key] = (self.max_tokens, now)
        
        # Get current tokens and last update time
        tokens, last_update = self.buckets[key]
        
        # Calculate new tokens
        elapsed = now - last_update
        new_tokens = min(self.max_tokens, tokens + elapsed * self.tokens_per_second)
        
        # Check if the request is allowed
        is_allowed = new_tokens >= 1
        
        # Update tokens if allowed
        if is_allowed:
            new_tokens -= 1
        
        # Update bucket
        self.buckets[key] = (new_tokens, now)
        
        # Calculate time to refill
        time_to_refill = 0 if new_tokens >= self.max_tokens else (self.max_tokens - new_tokens) / self.tokens_per_second
        
        # Return result and rate limit info
        return is_allowed, {
            "limit": self.max_tokens,
            "remaining": int(new_tokens),
            "reset": now + time_to_refill,
            "rate": self.tokens_per_second,
        }
    
    def _cleanup_old_buckets(self, now: float) -> None:
        """
        Clean up old buckets to prevent memory leaks.
        
        Args:
            now: Current time
        """
        # Remove buckets that haven't been used for 1 hour
        for key in list(self.buckets.keys()):
            _, last_update = self.buckets[key]
            if now - last_update > 3600:
                del self.buckets[key]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests."""
    
    def __init__(
        self,
        app: FastAPI,
        enabled: bool = None,
        strategy: str = None,
        limit: int = None,
        window_size: int = None,
        tokens_per_second: float = None,
        max_tokens: int = None,
        exclude_paths: List[str] = None,
        key_func: Callable[[Request], str] = None,
    ):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            enabled: Whether to enable rate limiting
            strategy: Rate limiting strategy ('fixed_window', 'sliding_window', 'token_bucket')
            limit: Request limit for fixed/sliding window strategies
            window_size: Window size in seconds for fixed/sliding window strategies
            tokens_per_second: Number of tokens added per second for token bucket strategy
            max_tokens: Maximum number of tokens for token bucket strategy
            exclude_paths: List of paths to exclude from rate limiting
            key_func: Function to generate a key from a request
        """
        super().__init__(app)
        
        # Set defaults from settings if not provided
        self.enabled = enabled if enabled is not None else settings.rate_limit.enabled
        self.strategy = strategy or settings.rate_limit.strategy
        self.limit = limit or settings.rate_limit.limit
        self.window_size = window_size or settings.rate_limit.window_size
        self.tokens_per_second = tokens_per_second or settings.rate_limit.tokens_per_second
        self.max_tokens = max_tokens or settings.rate_limit.max_tokens
        self.exclude_paths = exclude_paths or settings.rate_limit.exclude_paths
        
        # Add standard excluded paths
        self.exclude_paths.extend([
            "/health",
            "/api/health",
            "/api/v1/health",
            "/api/v2/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        ])
        
        # Initialize rate limiter based on strategy
        if self.strategy == "fixed_window":
            self.rate_limiter = FixedWindowRateLimiter(self.limit, self.window_size)
        elif self.strategy == "sliding_window":
            self.rate_limiter = SlidingWindowRateLimiter(self.limit, self.window_size)
        elif self.strategy == "token_bucket":
            self.rate_limiter = TokenBucketRateLimiter(self.tokens_per_second, self.max_tokens)
        else:
            # Default to sliding window
            self.rate_limiter = SlidingWindowRateLimiter(self.limit, self.window_size)
        
        # Set key function
        self.key_func = key_func or self._default_key_func
        
        logger.info(f"Rate limit middleware initialized (enabled: {self.enabled}, strategy: {self.strategy})")
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and apply rate limiting.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler
            
        Returns:
            Response from the next middleware or route handler
        """
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Check if this path should be excluded
        path = request.url.path
        for excluded_path in self.exclude_paths:
            if path.startswith(excluded_path):
                return await call_next(request)
        
        # Generate key from request
        key = self.key_func(request)
        
        # Check if the request is allowed
        is_allowed, rate_limit_info = self.rate_limiter.is_allowed(key)
        
        # If not allowed, return 429 Too Many Requests
        if not is_allowed:
            # Get request ID from state if available
            request_id = getattr(request.state, "request_id", None)
            
            # Calculate retry after
            retry_after = int(rate_limit_info["reset"] - time.time())
            
            # Create headers
            headers = {
                "X-RateLimit-Limit": str(rate_limit_info["limit"]),
                "X-RateLimit-Remaining": str(rate_limit_info["remaining"]),
                "X-RateLimit-Reset": str(int(rate_limit_info["reset"])),
                "Retry-After": str(max(1, retry_after)),
            }
            
            # Raise rate limit exceeded exception
            raise RateLimitExceededException(
                detail="Rate limit exceeded",
                headers=headers,
                suggestion=f"Please try again in {retry_after} seconds",
                details={
                    "rate_limit": rate_limit_info,
                    "client_id": key,
                    "request_id": request_id,
                }
            )
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(int(rate_limit_info["reset"]))
        
        return response
    
    def _default_key_func(self, request: Request) -> str:
        """
        Generate a key from a request for rate limiting.
        
        The key is based on the client IP address by default, but can be customized
        to include API keys, user IDs, or other identifiers.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Key string for rate limiting
        """
        # Try to get API key
        api_key = request.headers.get(settings.auth.api_key_name)
        
        # If API key is present, use it as the key
        if api_key:
            return f"api_key:{api_key}"
        
        # Try to get user ID from state
        user = getattr(request.state, "user", None)
        user_id = getattr(user, "username", None) if user else None
        
        # If user ID is present, use it as the key
        if user_id:
            return f"user:{user_id}"
        
        # Otherwise, use client IP address
        client_host = request.client.host if request.client else "unknown"
        
        # Get forwarded for header if behind a proxy
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_host = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_host}"


def setup_rate_limit_middleware(
    app: FastAPI,
    enabled: bool = None,
    strategy: str = None,
    limit: int = None,
    window_size: int = None,
    tokens_per_second: float = None,
    max_tokens: int = None,
    exclude_paths: List[str] = None,
    key_func: Callable[[Request], str] = None,
) -> None:
    """
    Configure and add the rate limit middleware to the application.
    
    Args:
        app: FastAPI application
        enabled: Whether to enable rate limiting
        strategy: Rate limiting strategy ('fixed_window', 'sliding_window', 'token_bucket')
        limit: Request limit for fixed/sliding window strategies
        window_size: Window size in seconds for fixed/sliding window strategies
        tokens_per_second: Number of tokens added per second for token bucket strategy
        max_tokens: Maximum number of tokens for token bucket strategy
        exclude_paths: List of paths to exclude from rate limiting
        key_func: Function to generate a key from a request
    """
    app.add_middleware(
        RateLimitMiddleware,
        enabled=enabled,
        strategy=strategy,
        limit=limit,
        window_size=window_size,
        tokens_per_second=tokens_per_second,
        max_tokens=max_tokens,
        exclude_paths=exclude_paths,
        key_func=key_func,
    )