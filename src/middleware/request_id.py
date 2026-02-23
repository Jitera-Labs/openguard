import re
import uuid
from contextvars import ContextVar
from typing import Any, MutableMapping

from starlette.types import ASGIApp, Receive, Scope, Send

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

_REQUEST_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class RequestIDMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        raw = headers.get(b"x-request-id", b"").decode("latin-1")
        if raw and _REQUEST_ID_RE.match(raw):
            request_id = raw
        else:
            request_id = str(uuid.uuid4())

        request_id_var.set(request_id)

        async def send_with_request_id(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers_list = list(message.get("headers", []))
                headers_list.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers_list}
            await send(message)

        await self.app(scope, receive, send_with_request_id)
