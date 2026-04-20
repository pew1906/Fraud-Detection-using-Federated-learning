"""
MongoDB connection and collection helpers.
Falls back gracefully if MongoDB is not available.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_db = None


async def get_db():
    """Return MongoDB database instance, or None if unavailable."""
    global _db
    if _db is not None:
        return _db

    mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "fedfraud")

    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=2000)
        await client.admin.command("ping")
        _db = client[db_name]
        logger.info(f"MongoDB connected: {mongo_url}/{db_name}")
        return _db
    except Exception as e:
        logger.warning(f"MongoDB unavailable ({e}). Running without persistence.")
        return None


async def save_experiment(db, config: dict, experiment_id: str):
    if db is None:
        return
    try:
        await db.experiments.insert_one({"_id": experiment_id, **config})
    except Exception as e:
        logger.warning(f"Failed to save experiment: {e}")


async def save_round(db, round_data: dict):
    if db is None:
        return
    try:
        await db.rounds.insert_one(round_data)
    except Exception as e:
        logger.warning(f"Failed to save round: {e}")


async def get_experiment_history(db, limit: int = 100):
    if db is None:
        return []
    try:
        cursor = db.rounds.find({}, {"_id": 0}).limit(limit)
        return await cursor.to_list(length=limit)
    except Exception as e:
        logger.warning(f"Failed to fetch history: {e}")
        return []
