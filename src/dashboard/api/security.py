"""Security middleware and utilities for dashboard API.

This module provides rate limiting, security headers, and other security
features for production deployment.

Security Impact:
    - Rate limiting prevents abuse and DoS attacks
    - Security headers protect against common vulnerabilities
    - Input validation prevents injection attacks
"""

import logging
import time
from collections import defaultdict
from typing import Callable, Dict, Tuple
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests.
    
    Implements a simple in-memory rate limiter using token bucket algorithm.
    For production, consider using Redis-based rate limiting.
    
    Security Impact:
        - Prevents DoS attacks
        - Protects against abuse
        - Configurable per endpoint
    """
    
    def __init__(
        self,
        app,
        default_limit: int = 100,
        default_window: int = 60,
        per_endpoint_limits: Dict[str, Tuple[int, int]] = None
    ):
        """Initialize rate limiter.
        
        Parameters:
            app: FastAPI application
            default_limit: Default requests per window
            default_window: Default window in seconds
            per_endpoint_limits: Dict mapping endpoint paths to (limit, window) tuples
        """
        super().__init__(app)
        self.default_limit = default_limit
        self.default_window = default_window
        self.per_endpoint_limits = per_endpoint_limits or {}
        
        # In-memory storage: {client_ip: {endpoint: [(timestamp, ...)]}}
        self._requests: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        
        # Cleanup old entries every 5 minutes
        self._last_cleanup = time.time()
        self._cleanup_interval = 300
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting.
        
        Parameters:
            request: HTTP request
            
        Returns:
            Client identifier (IP address or forwarded IP)
        """
        # Check for forwarded IP (behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take first IP in chain
            return forwarded_for.split(",")[0].strip()
        
        # Fallback to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_endpoint_key(self, request: Request) -> str:
        """Get endpoint key for rate limiting.
        
        Parameters:
            request: HTTP request
            
        Returns:
            Endpoint path pattern
        """
        # Use path as key
        path = request.url.path
        
        # Check for endpoint-specific limits
        for endpoint_pattern, _ in self.per_endpoint_limits.items():
            if path.startswith(endpoint_pattern):
                return endpoint_pattern
        
        return "default"
    
    def _cleanup_old_entries(self):
        """Remove old entries from rate limit tracking."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        cutoff_time = current_time - 3600  # Keep last hour
        
        for client_requests in self._requests.values():
            for endpoint_requests in client_requests.values():
                endpoint_requests[:] = [
                    ts for ts in endpoint_requests if ts > cutoff_time
                ]
        
        self._last_cleanup = current_time
    
    def _check_rate_limit(self, client_id: str, endpoint_key: str) -> Tuple[bool, int, int]:
        """Check if request is within rate limit.
        
        Parameters:
            client_id: Client identifier
            endpoint_key: Endpoint key
            
        Returns:
            Tuple of (allowed, remaining, reset_after)
        """
        # Get limits for this endpoint
        if endpoint_key in self.per_endpoint_limits:
            limit, window = self.per_endpoint_limits[endpoint_key]
        else:
            limit = self.default_limit
            window = self.default_window
        
        current_time = time.time()
        window_start = current_time - window
        
        # Get requests for this client and endpoint
        client_requests = self._requests[client_id][endpoint_key]
        
        # Remove requests outside the window
        client_requests[:] = [ts for ts in client_requests if ts > window_start]
        
        # Check if limit exceeded
        if len(client_requests) >= limit:
            reset_after = int(window - (current_time - client_requests[0]))
            return False, 0, reset_after
        
        # Add current request
        client_requests.append(current_time)
        
        remaining = limit - len(client_requests)
        reset_after = window
        
        return True, remaining, reset_after
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting.
        
        Parameters:
            request: Incoming HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            Response with rate limit headers
        """
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/api/health", "/api/docs", "/api/redoc", "/api/openapi.json", "/"]:
            return await call_next(request)
        
        # Cleanup old entries periodically
        self._cleanup_old_entries()
        
        # Get client and endpoint identifiers
        client_id = self._get_client_id(request)
        endpoint_key = self._get_endpoint_key(request)
        
        # Check rate limit
        allowed, remaining, reset_after = self._check_rate_limit(client_id, endpoint_key)
        
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {client_id} on {endpoint_key}"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "X-RateLimit-Limit": str(self.default_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + reset_after),
                    "Retry-After": str(reset_after)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.default_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + reset_after)
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers.
    
    Adds security headers to all responses to protect against common
    web vulnerabilities.
    
    Security Impact:
        - Prevents XSS attacks
        - Prevents clickjacking
        - Enforces HTTPS in production
        - Prevents MIME type sniffing
    """
    
    def __init__(self, app, enable_hsts: bool = False):
        """Initialize security headers middleware.
        
        Parameters:
            app: FastAPI application
            enable_hsts: Enable HSTS header (use in production with HTTPS)
        """
        super().__init__(app)
        self.enable_hsts = enable_hsts
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response.
        
        Parameters:
            request: Incoming HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            Response with security headers
        """
        response = await call_next(request)
        
        # Security headers
        security_headers = {
            # Prevent XSS attacks
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            
            # Content Security Policy (adjust for your needs)
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self' ws: wss:; "
                "frame-ancestors 'none'"
            ),
            
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions Policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=()"
            ),
        }
        
        # HSTS (only in production with HTTPS)
        if self.enable_hsts:
            security_headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # Add headers to response
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


def get_rate_limit_config() -> Dict[str, Tuple[int, int]]:
    """Get rate limit configuration per endpoint.
    
    Returns:
        Dictionary mapping endpoint patterns to (limit, window) tuples
    """
    return {
        "/api/metrics": (50, 60),  # 50 requests per minute
        "/api/audit-logs": (30, 60),  # 30 requests per minute
        "/api/redaction-logs": (30, 60),  # 30 requests per minute
        "/api/circuit-breaker": (20, 60),  # 20 requests per minute
        "/ws/realtime": (10, 60),  # 10 WebSocket connections per minute
    }

