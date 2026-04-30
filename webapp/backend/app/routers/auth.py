from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

from app.db.mongo import get_database
from app.deps.auth import get_current_user
from app.models.schemas import AuthTokenResponse, UserLoginRequest, UserRegisterRequest
from app.services.auth import create_access_token, hash_password, verify_password
from app.services import local_auth_store

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthTokenResponse)
async def register(payload: UserRegisterRequest):
    db = get_database()
    user_record = {
        "username": payload.username,
        "password_hash": hash_password(payload.password),
        "role": payload.role,
    }
    try:
        existing = await db.users.find_one({"username": payload.username})
    except ServerSelectionTimeoutError as exc:
        created = local_auth_store.create_user(user_record)
        if not created:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists") from exc
        token = create_access_token({"sub": payload.username, "role": payload.role})
        return AuthTokenResponse(access_token=token, role=payload.role, username=payload.username)
    except PyMongoError as exc:
        raise HTTPException(status_code=500, detail="Database error") from exc
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")

    try:
        await db.users.insert_one(user_record)
    except ServerSelectionTimeoutError as exc:
        created = local_auth_store.create_user(user_record)
        if not created:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists") from exc
    except PyMongoError as exc:
        raise HTTPException(status_code=500, detail="Database error") from exc

    token = create_access_token({"sub": payload.username, "role": payload.role})
    return AuthTokenResponse(access_token=token, role=payload.role, username=payload.username)


@router.post("/login", response_model=AuthTokenResponse)
async def login(payload: UserLoginRequest):
    db = get_database()
    try:
        user = await db.users.find_one({"username": payload.username})
    except ServerSelectionTimeoutError as exc:
        user = local_auth_store.find_user(payload.username)
    except PyMongoError as exc:
        raise HTTPException(status_code=500, detail="Database error") from exc
    if user is None or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token({"sub": payload.username, "role": user["role"]})
    return AuthTokenResponse(access_token=token, role=user["role"], username=payload.username)


@router.get("/me")
async def me(user=Depends(get_current_user)):
    return user
