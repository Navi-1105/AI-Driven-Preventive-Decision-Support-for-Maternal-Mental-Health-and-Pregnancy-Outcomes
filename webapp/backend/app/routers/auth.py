from fastapi import APIRouter, Depends, HTTPException, status

from app.db.mongo import get_database
from app.deps.auth import get_current_user
from app.models.schemas import AuthTokenResponse, UserLoginRequest, UserRegisterRequest
from app.services.auth import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthTokenResponse)
async def register(payload: UserRegisterRequest):
    db = get_database()
    existing = await db.users.find_one({"username": payload.username})
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")

    await db.users.insert_one(
        {
            "username": payload.username,
            "password_hash": hash_password(payload.password),
            "role": payload.role,
        }
    )

    token = create_access_token({"sub": payload.username, "role": payload.role})
    return AuthTokenResponse(access_token=token, role=payload.role, username=payload.username)


@router.post("/login", response_model=AuthTokenResponse)
async def login(payload: UserLoginRequest):
    db = get_database()
    user = await db.users.find_one({"username": payload.username})
    if user is None or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token({"sub": payload.username, "role": user["role"]})
    return AuthTokenResponse(access_token=token, role=user["role"], username=payload.username)


@router.get("/me")
async def me(user=Depends(get_current_user)):
    return user
