import asyncio

from pymongo.errors import ServerSelectionTimeoutError

from app.config import settings
from app.models.schemas import UserLoginRequest, UserRegisterRequest
from app.routers import auth as auth_router
from app.services.auth import create_access_token, decode_access_token


def test_jwt_encode_decode_roundtrip():
    token = create_access_token({"sub": "alice", "role": "clinician"})
    payload = decode_access_token(token)
    assert payload["sub"] == "alice"
    assert payload["role"] == "clinician"


def test_register_and_login_fall_back_to_local_store_when_mongo_unavailable(tmp_path, monkeypatch):
    class UnavailableUsers:
        async def find_one(self, query):
            raise ServerSelectionTimeoutError("Mongo is not running")

        async def insert_one(self, user):
            raise ServerSelectionTimeoutError("Mongo is not running")

    class UnavailableDb:
        users = UnavailableUsers()

    monkeypatch.setattr(settings, "auth_fallback_store_path", str(tmp_path / "auth-users.json"))
    monkeypatch.setattr(auth_router, "get_database", lambda: UnavailableDb())

    register_response = asyncio.run(
        auth_router.register(
            UserRegisterRequest(username="clin1", password="StrongPass123", role="clinician")
        )
    )
    login_response = asyncio.run(
        auth_router.login(UserLoginRequest(username="clin1", password="StrongPass123"))
    )

    assert register_response.token_type == "bearer"
    assert login_response.username == "clin1"
    assert decode_access_token(login_response.access_token)["role"] == "clinician"
