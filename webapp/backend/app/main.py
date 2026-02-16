from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.mongo import close_mongo_connection, connect_to_mongo
from app.routers.auth import router as auth_router
from app.routers.clinical import router as clinical_router
from app.routers.governance import router as governance_router

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()


@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()


@app.get("/")
async def root():
    return {"status": "ok", "app": settings.app_name}


app.include_router(clinical_router)
app.include_router(auth_router)
app.include_router(governance_router)
