from starlette.types import ASGIApp, Receive, Scope, Send

from src.state import request as request_state


class RequestStateMiddleware:
    """
    Tracks current request in the context state.
    Uses pure ASGI to avoid BaseHTTPMiddleware response buffering.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from fastapi import Request

        request = Request(scope, receive)
        request_state.set(request)
        try:
            await self.app(scope, receive, send)
        finally:
            request_state.set(None)
