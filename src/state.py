from contextvars import ContextVar

from fastapi import Request

request: ContextVar[Request | None] = ContextVar("request", default=None)
