"""Small transport-security primitives for the single-process API."""

from __future__ import annotations

import asyncio
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_IDENTITY_HMAC_KEY = secrets.token_bytes(32)


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, identity: str) -> tuple[bool, float]:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[identity]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False, max(0.0, self.window_seconds - (now - events[0]))
            events.append(now)
            if len(self._events) > 10_000:
                stale = [
                    key
                    for key, values in self._events.items()
                    if not values or values[-1] <= cutoff
                ]
                for key in stale[:1000]:
                    self._events.pop(key, None)
            return True, 0.0


def opaque_client_identity(client_host: str | None) -> str:
    """Return a stable, process-local limiter key without retaining the raw host."""
    material = b"\x00" if client_host is None else b"\x01" + client_host.encode("utf-8")
    return hmac.digest(_IDENTITY_HMAC_KEY, material, "sha256").hex()


class BodyLimitMiddleware:
    """Reject oversized bodies, including chunked requests, before JSON parsing."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        messages: list[Message] = []
        size = 0
        while True:
            message = await receive()
            messages.append(message)
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                payload = b'{"detail":"request body exceeds the configured byte limit"}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(payload)).encode("ascii")),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": payload})
                return
            if not message.get("more_body", False):
                break
        iterator = iter(messages)

        async def replay() -> Message:
            try:
                return next(iterator)
            except StopIteration:
                await asyncio.sleep(0)
                return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay, send)


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def secure_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"cache-control", b"no-store"),
                    ]
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, secure_send)
