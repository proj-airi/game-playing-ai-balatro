"""API Provider Engine for remote API calls."""

import time
import httpx
from typing import Dict, Any, Optional
import asyncio

from .base import BaseEngine, EngineConfig, EngineType
from ..llm.base import ProcessingResult
from ...utils.logger import get_logger

logger = get_logger(__name__)


class APIProviderEngine(BaseEngine):
    """Engine for making HTTP API calls to remote providers."""

    def __init__(
        self,
        name: str = "APIProvider",
        timeout: int = 30,
        max_retries: int = 3,
        **kwargs
    ):
        config = EngineConfig(
            engine_type=EngineType.API_PROVIDER,
            timeout=timeout,
            max_retries=max_retries,
            metadata=kwargs
        )
        super().__init__(name, config)

        self.client = None
        self.async_client = None

    def initialize(self) -> bool:
        """Initialize HTTP clients."""
        try:
            self.client = httpx.Client(
                timeout=self.config.timeout,
                follow_redirects=True
            )

            self.async_client = httpx.AsyncClient(
                timeout=self.config.timeout,
                follow_redirects=True
            )

            self.is_initialized = True
            logger.info(f"APIProviderEngine {self.name} initialized")
            return True

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to initialize APIProviderEngine: {e}")
            return False

    def shutdown(self) -> None:
        """Clean up HTTP clients."""
        try:
            if self.client:
                self.client.close()
            if self.async_client:
                asyncio.create_task(self.async_client.aclose())

            self.is_initialized = False
            logger.info(f"APIProviderEngine {self.name} shut down")

        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")

    def make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        """
        Make HTTP request with retries.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            headers: Request headers
            data: Form data
            json_data: JSON payload

        Returns:
            ProcessingResult with response data
        """
        if not self.is_initialized:
            return ProcessingResult(
                success=False,
                data=None,
                errors=["Engine not initialized"]
            )

        start_time = time.time()

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=data,
                    json=json_data
                )

                processing_time = time.time() - start_time

                # Check if request was successful
                if response.is_success:
                    try:
                        response_data = response.json()
                    except Exception:
                        response_data = {"text": response.text}

                    return ProcessingResult(
                        success=True,
                        data={
                            "response": response_data,
                            "status_code": response.status_code,
                            "headers": dict(response.headers)
                        },
                        processing_time=processing_time,
                        metadata={
                            "attempt": attempt + 1,
                            "url": url,
                            "method": method
                        }
                    )
                else:
                    # HTTP error - might retry depending on status
                    error_msg = f"HTTP {response.status_code}: {response.text}"

                    # Don't retry on client errors (4xx)
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        return ProcessingResult(
                            success=False,
                            data=None,
                            processing_time=processing_time,
                            errors=[error_msg],
                            metadata={"status_code": response.status_code}
                        )

                    # Retry on server errors (5xx) and rate limits (429)
                    if attempt == self.config.max_retries - 1:
                        return ProcessingResult(
                            success=False,
                            data=None,
                            processing_time=processing_time,
                            errors=[error_msg],
                            metadata={
                                "status_code": response.status_code,
                                "attempts": self.config.max_retries
                            }
                        )

                    logger.warning(f"Attempt {attempt + 1} failed with {response.status_code}, retrying...")

            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    processing_time = time.time() - start_time
                    return ProcessingResult(
                        success=False,
                        data=None,
                        processing_time=processing_time,
                        errors=[f"Request failed: {e}"],
                        metadata={"attempts": self.config.max_retries}
                    )

                logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")

            # Exponential backoff
            if attempt < self.config.max_retries - 1:
                time.sleep(2 ** attempt)

    async def make_async_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        """Make async HTTP request with retries."""
        if not self.is_initialized:
            return ProcessingResult(
                success=False,
                data=None,
                errors=["Engine not initialized"]
            )

        start_time = time.time()

        for attempt in range(self.config.max_retries):
            try:
                response = await self.async_client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=data,
                    json=json_data
                )

                processing_time = time.time() - start_time

                if response.is_success:
                    try:
                        response_data = response.json()
                    except Exception:
                        response_data = {"text": response.text}

                    return ProcessingResult(
                        success=True,
                        data={
                            "response": response_data,
                            "status_code": response.status_code,
                            "headers": dict(response.headers)
                        },
                        processing_time=processing_time,
                        metadata={
                            "attempt": attempt + 1,
                            "url": url,
                            "method": method
                        }
                    )
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"

                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        return ProcessingResult(
                            success=False,
                            data=None,
                            processing_time=processing_time,
                            errors=[error_msg],
                            metadata={"status_code": response.status_code}
                        )

                    if attempt == self.config.max_retries - 1:
                        return ProcessingResult(
                            success=False,
                            data=None,
                            processing_time=processing_time,
                            errors=[error_msg],
                            metadata={
                                "status_code": response.status_code,
                                "attempts": self.config.max_retries
                            }
                        )

            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    processing_time = time.time() - start_time
                    return ProcessingResult(
                        success=False,
                        data=None,
                        processing_time=processing_time,
                        errors=[f"Async request failed: {e}"],
                        metadata={"attempts": self.config.max_retries}
                    )

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
