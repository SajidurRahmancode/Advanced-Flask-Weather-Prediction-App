"""
circuit_breaker.py — Circuit breaker protection for AI services.

States:
    CLOSED   → normal, every call passes through
    OPEN     → too many failures, calls rejected immediately (or fall back)
    HALF_OPEN → testing recovery, one probe call allowed

Named breakers with tuned thresholds:
    lm_studio   — 3 failures / 30 s reset  (fast fail, expensive service)
    rag_service — 5 failures / 60 s reset
    langgraph   — 3 failures / 45 s reset
    ensemble    — 5 failures / 90 s reset  (has internal fallbacks)

IMPORTANT: LMStudioService.generate_chat() returns a warning *string*
(not an exception) on timeout → the lm_studio breaker detects this.

Usage:
    from backend.circuit_breaker import circuit_breakers

    # Protect a call, with optional fallback
    result, used_fallback = circuit_breakers.lm_studio.call(
        lambda: lm_studio_service.generate_chat(messages),
        fallback=lambda: statistical_result,
    )

    # Status for monitoring
    circuit_breakers.get_all_status()
"""

import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The string lmstudio_service.generate_chat() returns on timeout
LM_STUDIO_TIMEOUT_PREFIX = "⚠️ Prediction generation timed out"


class CircuitState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


@dataclass
class BreakerConfig:
    failure_threshold: int    # failures before opening
    success_threshold: int    # successes (from HALF_OPEN) to close
    timeout_seconds:   float  # seconds before HALF_OPEN retry


@dataclass
class _Stats:
    total_calls:    int = 0
    successes:      int = 0
    failures:       int = 0
    rejections:     int = 0
    state_changes:  List[Dict] = field(default_factory=list)


class CircuitBreaker:
    """
    Thread-safe circuit breaker for a single AI service.
    """

    def __init__(self, name: str, config: BreakerConfig) -> None:
        self.name   = name
        self.config = config
        self._lock  = threading.Lock()

        self._state:            CircuitState   = CircuitState.CLOSED
        self._failure_count:    int            = 0
        self._success_count:    int            = 0
        self._last_failure_at:  Optional[float] = None
        self._last_changed_at:  datetime        = datetime.now()
        self._stats:            _Stats          = _Stats()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        return self._state

    def call(
        self,
        fn: Callable[[], Any],
        fallback: Optional[Callable[[], Any]] = None,
    ) -> Tuple[Any, bool]:
        """
        Execute *fn* with circuit-breaker protection.

        Args:
            fn:       Zero-argument callable (the protected AI call).
            fallback: Zero-argument callable used when circuit is OPEN.

        Returns:
            (result, used_fallback: bool)

        Raises:
            CircuitBreakerOpenError if OPEN and no fallback provided.
        """
        with self._lock:
            self._stats.total_calls += 1
            current_state = self._maybe_transition()

        if current_state == CircuitState.OPEN:
            with self._lock:
                self._stats.rejections += 1
            remaining = self._seconds_until_retry()
            logger.warning(
                "⛔ [%s] OPEN — rejecting call (retry in %.0fs)",
                self.name, remaining,
            )
            if fallback:
                result = fallback()
                return result, True
            raise CircuitBreakerOpenError(
                f"Circuit breaker [{self.name}] OPEN. "
                f"Retry in {remaining:.0f}s."
            )

        # CLOSED or HALF_OPEN — execute
        try:
            result = fn()
            self._handle_result(result)
            return result, False

        except Exception as exc:
            self._record_failure(repr(exc))
            if fallback:
                fallback_result = fallback()
                return fallback_result, True
            raise

    def record_failure(self, reason: str = "external") -> None:
        """Manually record a failure (e.g. detected by caller from return value)."""
        self._record_failure(reason)

    def reset(self) -> None:
        """Admin: manually reset to CLOSED."""
        with self._lock:
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_at = None
            self._change_state(CircuitState.CLOSED, "manual reset")

    def get_status(self) -> Dict:
        with self._lock:
            total = self._stats.total_calls
            return {
                "name":               self.name,
                "state":              self._state.value,
                "failure_count":      self._failure_count,
                "failure_threshold":  self.config.failure_threshold,
                "total_calls":        total,
                "success_rate_pct":   round(
                    self._stats.successes / total * 100, 2
                ) if total else 0,
                "rejection_rate_pct": round(
                    self._stats.rejections / total * 100, 2
                ) if total else 0,
                "last_changed_at":    self._last_changed_at.isoformat(),
                "recent_transitions": self._stats.state_changes[-5:],
                "seconds_until_retry": (
                    self._seconds_until_retry()
                    if self._state == CircuitState.OPEN
                    else None
                ),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_transition(self) -> CircuitState:
        """Check OPEN → HALF_OPEN timeout; return current state."""
        if (
            self._state == CircuitState.OPEN
            and self._last_failure_at is not None
            and (time.time() - self._last_failure_at) >= self.config.timeout_seconds
        ):
            self._change_state(CircuitState.HALF_OPEN, "timeout elapsed")
        return self._state

    def _handle_result(self, result: Any) -> None:
        """Check for string-encoded timeouts from LM Studio."""
        # Detect lm_studio timeout-string as a failure
        if (
            isinstance(result, str)
            and result.startswith(LM_STUDIO_TIMEOUT_PREFIX)
        ):
            self._record_failure("lm_studio_timeout_string")
            return
        self._record_success()

    def _record_success(self) -> None:
        with self._lock:
            self._stats.successes += 1
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._change_state(
                        CircuitState.CLOSED,
                        f"recovered after {self._success_count} successes",
                    )
                    self._success_count = 0

    def _record_failure(self, reason: str) -> None:
        with self._lock:
            self._stats.failures += 1
            self._failure_count += 1
            self._last_failure_at = time.time()
            self._success_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._change_state(
                    CircuitState.OPEN,
                    f"failed during half-open: {reason[:80]}",
                )
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.config.failure_threshold
            ):
                self._change_state(
                    CircuitState.OPEN,
                    f"threshold reached ({self._failure_count} failures): {reason[:80]}",
                )

    def _change_state(self, new_state: CircuitState, reason: str) -> None:
        old = self._state
        self._state = new_state
        self._last_changed_at = datetime.now()
        entry = {
            "from":      old.value,
            "to":        new_state.value,
            "reason":    reason,
            "timestamp": self._last_changed_at.isoformat(),
        }
        self._stats.state_changes.append(entry)
        logger.warning(
            "⚡ CircuitBreaker [%s]: %s → %s (%s)",
            self.name, old.value, new_state.value, reason,
        )

    def _seconds_until_retry(self) -> float:
        if self._last_failure_at is None:
            return 0.0
        elapsed = time.time() - self._last_failure_at
        return max(0.0, self.config.timeout_seconds - elapsed)


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a circuit breaker is OPEN and no fallback is supplied."""


# ------------------------------------------------------------------
# Central registry — one instance shared across the whole app
# ------------------------------------------------------------------

class AIServiceCircuitBreakers:
    """Registry of all AI-service circuit breakers."""

    def __init__(self) -> None:
        self.lm_studio  = CircuitBreaker(
            "lm_studio",
            BreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=30.0),
        )
        self.rag_service = CircuitBreaker(
            "rag_service",
            BreakerConfig(failure_threshold=5, success_threshold=2, timeout_seconds=60.0),
        )
        self.langgraph  = CircuitBreaker(
            "langgraph",
            BreakerConfig(failure_threshold=3, success_threshold=1, timeout_seconds=45.0),
        )
        self.ensemble   = CircuitBreaker(
            "ensemble",
            BreakerConfig(failure_threshold=5, success_threshold=2, timeout_seconds=90.0),
        )

        self._all = {
            "lm_studio":   self.lm_studio,
            "rag_service": self.rag_service,
            "langgraph":   self.langgraph,
            "ensemble":    self.ensemble,
        }

    def get(self, name: str) -> Optional[CircuitBreaker]:
        return self._all.get(name)

    def get_all_status(self) -> Dict:
        return {
            name: cb.get_status()
            for name, cb in self._all.items()
        }

    def reset_all(self) -> None:
        for cb in self._all.values():
            cb.reset()


# Global singleton
circuit_breakers = AIServiceCircuitBreakers()
