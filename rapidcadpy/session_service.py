"""Shared state proxy for services composed by :class:`CadSession`."""

from __future__ import annotations

from typing import Any


class SessionService:
    """Operate on a session's state without inheriting its public facade."""

    __slots__ = ("_session",)

    def __init__(self, session: Any) -> None:
        object.__setattr__(self, "_session", session)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_session":
            object.__setattr__(self, name, value)
            return
        setattr(self._session, name, value)
