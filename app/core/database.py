from motor.motor_asyncio import AsyncIOMotorClient
from redis import Redis
from .config import settings

class Database:
    client: AsyncIOMotorClient = None
    redis: Redis = None

db = Database()

async def connect_to_mongo():
    db.client = AsyncIOMotorClient(settings.MONGODB_URL)
    print("Connected to MongoDB")

async def close_mongo_connection():
    db.client.close()
    print("Disconnected from MongoDB")

def connect_to_redis():
    db.redis = Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=True
    )
    print("Connected to Redis")

def close_redis_connection():
    if db.redis:
        db.redis.close()
        print("Disconnected from Redis") 