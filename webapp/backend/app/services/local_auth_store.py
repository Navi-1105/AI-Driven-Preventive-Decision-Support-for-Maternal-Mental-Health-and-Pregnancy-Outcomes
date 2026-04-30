import json
import threading
from pathlib import Path
from typing import Any

from app.config import settings

_store_lock = threading.Lock()


def _store_path() -> Path:
    return Path(settings.auth_fallback_store_path)


def _load_users() -> dict[str, dict[str, Any]]:
    path = _store_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as store_file:
            data = json.load(store_file)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(username): user for username, user in data.items() if isinstance(user, dict)}


def _save_users(users: dict[str, dict[str, Any]]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as store_file:
        json.dump(users, store_file, indent=2, sort_keys=True)


def find_user(username: str) -> dict[str, Any] | None:
    with _store_lock:
        return _load_users().get(username)


def create_user(user: dict[str, Any]) -> bool:
    username = str(user["username"])
    with _store_lock:
        users = _load_users()
        if username in users:
            return False
        users[username] = user
        _save_users(users)
        return True
