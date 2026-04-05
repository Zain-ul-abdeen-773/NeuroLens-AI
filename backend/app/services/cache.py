"""
NeuroLens Analysis Cache
LRU-based in-memory cache for analysis results.
"""

import time
from typing import Any, Dict, Optional
from collections import OrderedDict

from app.config import settings
from app.utils.logger import logger


class AnalysisCache:
    """
    Thread-safe LRU cache for analysis results.
    Supports TTL-based expiration.
    """

    def __init__(
        self,
        max_size: int = None,
        ttl_seconds: int = None,
    ):
        self.max_size = max_size or settings.CACHE_MAX_SIZE
        self.ttl = ttl_seconds or settings.CACHE_TTL_SECONDS
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._timestamps: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a cached result by key."""
        if key not in self._cache:
            return None

        # Check TTL
        if time.time() - self._timestamps[key] > self.ttl:
            self._remove(key)
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Store a result in cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                self._remove(oldest_key)

        self._cache[key] = value
        self._timestamps[key] = time.time()

    def _remove(self, key: str) -> None:
        """Remove an entry from cache."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._timestamps.clear()
        logger.debug("Cache cleared")

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> Dict[str, int]:
        return {
            "size": self.size,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
        }
