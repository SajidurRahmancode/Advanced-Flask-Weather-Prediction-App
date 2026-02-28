"""
rate_limiter.py — In-memory token-bucket rate limiter.

Two flat tiers (no User model change required):
  anonymous      :  10 req/hour, burst 1
  authenticated  :  60 req/hour, burst 5 concurrent

Thread-safe; uses threading.Lock.
Returns RFC-compliant X-RateLimit-* headers for API consumers.

Usage:
    from backend.rate_limiter import rate_limiter

    allowed, headers = rate_limiter.check(str(user_id), 'authenticated')
    if not allowed:
        return jsonify({"error": "Rate limit exceeded", **headers}), 429
    try:
        ...
    finally:
        rate_limiter.release(str(user_id))
"""

import time
import threading
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Tuple, Deque

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TierConfig:
    requests_per_hour: int
    burst_limit: int        # max concurrent requests per user


TIER_CONFIGS: Dict[str, TierConfig] = {
    "anonymous":     TierConfig(requests_per_hour=10,  burst_limit=1),
    "authenticated": TierConfig(requests_per_hour=60,  burst_limit=5),
}


class TokenBucketRateLimiter:
    """
    Sliding-window rate limiter backed by per-user deques.

    Checks both:
    - Hourly request count (sliding window)
    - Concurrent (burst) requests
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # user_id → deque of timestamps within the last hour
        self._hour_windows: Dict[str, Deque[float]] = defaultdict(deque)
        # user_id → active request count
        self._active: Dict[str, int] = defaultdict(int)
        # stats
        self._total_checks = 0
        self._total_rejected = 0

    # ------------------------------------------------------------------
    def check(
        self,
        user_id: str,
        tier: str = "authenticated",
    ) -> Tuple[bool, Dict]:
        """
        Check whether the request is within rate limits.

        Args:
            user_id: String identifier (str(session['user_id']) or 'anon:<ip>')
            tier:    'anonymous' or 'authenticated'

        Returns:
            (allowed, headers_dict)
            headers_dict always contains X-RateLimit-* keys.
            On rejection it additionally contains 'error' and 'retry_after'.
        """
        config = TIER_CONFIGS.get(tier, TIER_CONFIGS["authenticated"])
        now = time.time()
        hour_ago = now - 3600

        with self._lock:
            self._total_checks += 1

            # --- slide the window ---
            window = self._hour_windows[user_id]
            while window and window[0] < hour_ago:
                window.popleft()

            hour_count = len(window)
            active = self._active[user_id]

            # --- enforce hourly limit ---
            if hour_count >= config.requests_per_hour:
                self._total_rejected += 1
                oldest = window[0]
                retry_after = int(oldest + 3600 - now) + 1
                logger.warning(
                    "Rate limit (hourly) hit for user=%s tier=%s "
                    "count=%d limit=%d",
                    user_id, tier, hour_count, config.requests_per_hour,
                )
                return False, {
                    "error": "Rate limit exceeded",
                    "X-RateLimit-Limit":     config.requests_per_hour,
                    "X-RateLimit-Remaining": 0,
                    "X-RateLimit-Reset":     int(oldest + 3600),
                    "Retry-After":           retry_after,
                    "retry_after_seconds":   retry_after,
                }

            # --- enforce burst limit ---
            if active >= config.burst_limit:
                self._total_rejected += 1
                logger.warning(
                    "Rate limit (burst) hit for user=%s active=%d limit=%d",
                    user_id, active, config.burst_limit,
                )
                return False, {
                    "error": "Too many concurrent requests",
                    "X-RateLimit-Limit":     config.burst_limit,
                    "X-RateLimit-Remaining": 0,
                    "Retry-After":           5,
                    "retry_after_seconds":   5,
                }

            # --- allow ---
            window.append(now)
            self._active[user_id] += 1
            remaining = config.requests_per_hour - hour_count - 1

            return True, {
                "X-RateLimit-Limit":     config.requests_per_hour,
                "X-RateLimit-Remaining": remaining,
                "X-RateLimit-Reset":     int(now + 3600),
            }

    def release(self, user_id: str) -> None:
        """Decrement active-request counter when a request completes."""
        with self._lock:
            if self._active[user_id] > 0:
                self._active[user_id] -= 1

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_checks":    self._total_checks,
                "total_rejected":  self._total_rejected,
                "active_users":    sum(
                    1 for v in self._active.values() if v > 0
                ),
                "tracked_users":   len(self._hour_windows),
                "reject_rate_pct": round(
                    self._total_rejected / self._total_checks * 100, 2
                ) if self._total_checks else 0,
            }

    def reset_user(self, user_id: str) -> None:
        """Admin: clear all limits for a single user."""
        with self._lock:
            self._hour_windows.pop(user_id, None)
            self._active[user_id] = 0


# Global singleton — imported directly by routes.py
rate_limiter = TokenBucketRateLimiter()
