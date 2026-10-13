import logging
from typing import Optional
import pymongo
from pymongo.collection import Collection
from pymongo.database import Database
from app.core.config import settings

logger = logging.getLogger(__name__)

_mongo_client: Optional[pymongo.MongoClient] = None

def get_mongo_client() -> pymongo.MongoClient:
    """Returns or creates a singleton MongoClient."""
    global _mongo_client
    if _mongo_client is None:
        try:
            _mongo_client = pymongo.MongoClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=3000
            )
            # Test connection
            _mongo_client.admin.command('ping')
            logger.info(f"[MONGODB] Connected successfully to {settings.MONGODB_URL}")
        except Exception as e:
            logger.error(f"[MONGODB] Connection error: {e}")
            raise e
    return _mongo_client

def get_database(db_name: Optional[str] = None) -> Database:
    """Returns the application MongoDB database instance."""
    client = get_mongo_client()
    database_name = db_name or settings.MONGODB_DATABASE
    return client[database_name]

def get_users_collection(db_name: Optional[str] = None) -> Collection:
    """Returns the users collection with unique index on email ensured."""
    db = get_database(db_name)
    users_col = db["users"]
    try:
        users_col.create_index("email", unique=True)
    except Exception as e:
        logger.warning(f"[MONGODB] Warning ensuring unique email index: {e}")
    return users_col

def get_chat_sessions_collection(db_name: Optional[str] = None) -> Collection:
    """Returns the chat_sessions collection with unique (user_id, session_id) and (user_id, updated_at) indexes ensured."""
    db = get_database(db_name)
    sessions_col = db["chat_sessions"]
    try:
        sessions_col.create_index([("user_id", pymongo.ASCENDING), ("session_id", pymongo.ASCENDING)], unique=True)
        sessions_col.create_index([("user_id", pymongo.ASCENDING), ("updated_at", pymongo.DESCENDING)])
    except Exception as e:
        logger.warning(f"[MONGODB] Warning ensuring chat_sessions indexes: {e}")
    return sessions_col

def get_chat_messages_collection(db_name: Optional[str] = None) -> Collection:
    """Returns the chat_messages collection with (user_id, session_id, created_at) index ensured."""
    db = get_database(db_name)
    messages_col = db["chat_messages"]
    try:
        messages_col.create_index([
            ("user_id", pymongo.ASCENDING),
            ("session_id", pymongo.ASCENDING),
            ("created_at", pymongo.ASCENDING)
        ])
    except Exception as e:
        logger.warning(f"[MONGODB] Warning ensuring chat_messages indexes: {e}")
    return messages_col

def close_mongo_connection():
    """Closes MongoDB client connection."""
    global _mongo_client
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
        logger.info("[MONGODB] Client connection closed.")
