from motor.motor_asyncio import AsyncIOMotorClient

from app.config import settings


class MongoClient:
    client: AsyncIOMotorClient | None = None


mongo = MongoClient()

def _build_client() -> AsyncIOMotorClient:
    # Fail fast when MongoDB is not running (prevents request hangs on first DB operation).
    return AsyncIOMotorClient(
        settings.mongo_uri,
        serverSelectionTimeoutMS=2000,
        connectTimeoutMS=2000,
        socketTimeoutMS=5000,
    )


def get_database():
    if mongo.client is None:
        mongo.client = _build_client()
    return mongo.client[settings.mongo_db]


async def connect_to_mongo():
    if mongo.client is None:
        mongo.client = _build_client()
    # Trigger server selection during startup so DB issues surface immediately,
    # but don't crash the entire API if Mongo isn't available (auth endpoints will return 503 quickly).
    try:
        await mongo.client.admin.command("ping")
    except Exception:
        mongo.client.close()
        mongo.client = None


async def close_mongo_connection():
    if mongo.client is not None:
        mongo.client.close()
        mongo.client = None
