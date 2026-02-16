from motor.motor_asyncio import AsyncIOMotorClient

from app.config import settings


class MongoClient:
    client: AsyncIOMotorClient | None = None


mongo = MongoClient()


def get_database():
    if mongo.client is None:
        mongo.client = AsyncIOMotorClient(settings.mongo_uri)
    return mongo.client[settings.mongo_db]


async def connect_to_mongo():
    if mongo.client is None:
        mongo.client = AsyncIOMotorClient(settings.mongo_uri)


async def close_mongo_connection():
    if mongo.client is not None:
        mongo.client.close()
        mongo.client = None
