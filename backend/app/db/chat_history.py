import re
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pymongo

from app.db.mongodb import get_chat_sessions_collection, get_chat_messages_collection

logger = logging.getLogger(__name__)


import math
from dataclasses import is_dataclass, asdict


def sanitize_for_mongo(obj: Any) -> Any:
    """
    Recursively transforms arbitrary Python data structures into BSON-safe primitives
    (dict, list, str, int, float, bool, None, datetime).
    Converts Pydantic models (v1 and v2), dataclasses, sets, and custom objects safely.
    Replaces prohibited MongoDB key characters ('.' and leading '$').
    """
    if obj is None:
        return None
    if isinstance(obj, (int, str, bool, datetime, bytes)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj

    # Pydantic models
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return sanitize_for_mongo(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return sanitize_for_mongo(obj.dict())
        except Exception:
            pass

    # Dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return sanitize_for_mongo(asdict(obj))
        except Exception:
            pass

    # Dictionaries
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            str_k = str(k)
            if "." in str_k or str_k.startswith("$"):
                str_k = str_k.replace(".", "_")
                if str_k.startswith("$"):
                    str_k = "_" + str_k[1:]
            sanitized[str_k] = sanitize_for_mongo(v)
        return sanitized

    # Lists, tuples, sets
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_mongo(item) for item in obj]

    # Object with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return sanitize_for_mongo(vars(obj))
        except Exception:
            pass

    # Fallback to string representation
    return str(obj)


def generate_conversation_title(first_message: str) -> str:
    """
    Deterministically generates a concise, readable conversation title (max ~50 chars)
    from the first user query without invoking an LLM.
    """
    if not first_message or not first_message.strip():
        return "New Conversation"

    text = first_message.strip()

    # Strip language directives if present
    text = re.sub(r"^(?:Answer\s+in\s+(?:Telugu|Hindi|English):\s*)", "", text, flags=re.IGNORECASE)

    # Normalize excessive whitespace
    text = " ".join(text.split())

    # Common prefixes to clean for cleaner titles
    prefix_patterns = [
        r"^(?:what\s+are\s+the\s+timings\s+of\s+)",
        r"^(?:what\s+is\s+the\s+weather\s+(?:at|in|for)\s+)",
        r"^(?:what\s+will\s+a\s+visit\s+to\s+)",
        r"^(?:how\s+far\s+is\s+)",
        r"^(?:tell\s+me\s+about\s+)",
        r"^(?:can\s+you\s+tell\s+me\s+about\s+)",
        r"^(?:what\s+are\s+the\s+)",
        r"^(?:what\s+is\s+the\s+)",
        r"^(?:how\s+to\s+reach\s+)",
        r"^(?:plan\s+a\s+)",
    ]

    clean_text = text
    for pat in prefix_patterns:
        m = re.match(pat, clean_text, flags=re.IGNORECASE)
        if m:
            clean_text = clean_text[m.end():].strip()
            break

    # Strip trailing punctuation
    clean_text = clean_text.rstrip("?.! ")

    if not clean_text:
        clean_text = text.rstrip("?.! ")

    # Capitalize first letter
    if clean_text:
        clean_text = clean_text[0].upper() + clean_text[1:]

    # Truncate to 50 chars cleanly at word boundary
    if len(clean_text) > 50:
        truncated = clean_text[:47]
        last_space = truncated.rfind(" ")
        if last_space > 20:
            clean_text = truncated[:last_space] + "..."
        else:
            clean_text = truncated + "..."

    return clean_text or "New Conversation"


def init_chat_indexes(db_name: Optional[str] = None) -> None:
    """
    Initializes database-level unique and query indexes for chat sessions and messages.
    Enforces uniqueness on client_message_id per (user_id, session_id, role) to prevent
    duplicate records and race conditions under true concurrent requests.
    """
    messages_col = get_chat_messages_collection(db_name)
    sessions_col = get_chat_sessions_collection(db_name)

    # 1. Unique index on user messages with client_message_id
    try:
        messages_col.create_index(
            [("user_id", pymongo.ASCENDING), ("session_id", pymongo.ASCENDING), ("client_message_id", pymongo.ASCENDING)],
            unique=True,
            partialFilterExpression={"client_message_id": {"$type": "string"}, "role": "user"}
        )
    except Exception as e:
        logger.debug(f"[CHAT INDEXES] user client_msg index note: {e}")

    # 2. Unique index on assistant replies with reply_to_client_id
    try:
        messages_col.create_index(
            [("user_id", pymongo.ASCENDING), ("session_id", pymongo.ASCENDING), ("reply_to_client_id", pymongo.ASCENDING)],
            unique=True,
            partialFilterExpression={"reply_to_client_id": {"$type": "string"}, "role": "assistant"}
        )
    except Exception as e:
        logger.debug(f"[CHAT INDEXES] assistant reply client_msg index note: {e}")

    # 3. Fast lookup index for session messages
    try:
        messages_col.create_index(
            [("user_id", pymongo.ASCENDING), ("session_id", pymongo.ASCENDING), ("created_at", pymongo.ASCENDING)]
        )
    except Exception as e:
        logger.debug(f"[CHAT INDEXES] session messages lookup index note: {e}")

    # 4. Fast lookup index for user sessions
    try:
        sessions_col.create_index(
            [("user_id", pymongo.ASCENDING), ("updated_at", pymongo.DESCENDING)]
        )
    except Exception as e:
        logger.debug(f"[CHAT INDEXES] user sessions lookup index note: {e}")


def claim_user_message_turn(
    user_id: str,
    session_id: str,
    content: str,
    client_message_id: Optional[str] = None,
    db_name: Optional[str] = None
) -> tuple[bool, Optional[str]]:
    """
    Atomically claims a conversation turn using MongoDB uniqueness constraints.
    Returns (is_owner, message_id):
    - (True, message_id) if this request successfully claimed the turn and should execute the coordinator.
    - (False, message_id) if another concurrent request already claimed this turn.
    """
    if not user_id or not session_id or not content:
        return False, None

    now = datetime.now(timezone.utc)
    sessions_col = get_chat_sessions_collection(db_name)
    messages_col = get_chat_messages_collection(db_name)

    # If client_message_id was supplied, check if already exists
    if client_message_id:
        existing = messages_col.find_one({
            "user_id": user_id,
            "session_id": session_id,
            "client_message_id": client_message_id,
            "role": "user"
        })
        if existing:
            return False, str(existing["_id"])

    title = generate_conversation_title(content)

    # Upsert session document atomically
    sessions_col.update_one(
        {"user_id": user_id, "session_id": session_id},
        {
            "$setOnInsert": {
                "user_id": user_id,
                "session_id": session_id,
                "title": title,
                "created_at": now
            },
            "$set": {
                "updated_at": now
            }
        },
        upsert=True
    )

    doc: Dict[str, Any] = {
        "user_id": user_id,
        "session_id": session_id,
        "role": "user",
        "content": content,
        "agent_name": None,
        "created_at": now
    }
    if client_message_id:
        doc["client_message_id"] = client_message_id

    try:
        res = messages_col.insert_one(doc)
        return True, str(res.inserted_id)
    except pymongo.errors.DuplicateKeyError:
        # Atomic claim lost to concurrent request!
        existing = messages_col.find_one({
            "user_id": user_id,
            "session_id": session_id,
            "client_message_id": client_message_id,
            "role": "user"
        })
        existing_id = str(existing["_id"]) if existing else None
        return False, existing_id


def wait_for_assistant_reply(
    user_id: str,
    session_id: str,
    client_message_id: str,
    timeout_seconds: float = 30.0,
    poll_interval: float = 0.25,
    db_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Safely waits for an in-flight concurrent coordinator execution to save its assistant response.
    Returns the assistant response document or None if timeout expires.
    """
    import time
    start_time = time.monotonic()
    messages_col = get_chat_messages_collection(db_name)

    while (time.monotonic() - start_time) < timeout_seconds:
        doc = messages_col.find_one({
            "user_id": user_id,
            "session_id": session_id,
            "reply_to_client_id": client_message_id,
            "role": "assistant"
        })
        if doc:
            return doc
        time.sleep(poll_interval)
    return None


def save_user_message(
    user_id: str,
    session_id: str,
    content: str,
    client_message_id: Optional[str] = None,
    db_name: Optional[str] = None
) -> Optional[str]:
    """
    Saves a user message to chat_messages and ensures a chat_sessions document exists.
    Idempotent: if a user message with the same (user_id, session_id, client_message_id)
    already exists, returns its message_id without creating a duplicate.
    """
    _, msg_id = claim_user_message_turn(user_id, session_id, content, client_message_id, db_name)
    return msg_id


def save_assistant_message(
    user_id: str,
    session_id: str,
    content: str,
    agent_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    sources: Optional[List[str]] = None,
    confidence_score: Optional[float] = None,
    client_message_id: Optional[str] = None,
    db_name: Optional[str] = None
) -> Optional[str]:
    """
    Saves an assistant response to chat_messages along with structured metadata,
    sources, and confidence score, and updates chat_sessions updated_at timestamp.
    Idempotent: if an assistant message replying to client_message_id exists,
    returns its message_id without inserting a duplicate.
    Sanitizes all metadata into BSON-safe primitives to guarantee persistence.
    """
    if not user_id or not session_id or not content:
        return None

    now = datetime.now(timezone.utc)
    sessions_col = get_chat_sessions_collection(db_name)
    messages_col = get_chat_messages_collection(db_name)

    if client_message_id:
        existing = messages_col.find_one({
            "user_id": user_id,
            "session_id": session_id,
            "reply_to_client_id": client_message_id,
            "role": "assistant"
        })
        if existing:
            return str(existing["_id"])

    # Update session updated_at
    sessions_col.update_one(
        {"user_id": user_id, "session_id": session_id},
        {"$set": {"updated_at": now}}
    )

    # Sanitize metadata so Pydantic models, sets, and custom types never cause InvalidDocument
    sanitized_meta = sanitize_for_mongo(metadata or {})

    doc: Dict[str, Any] = {
        "user_id": user_id,
        "session_id": session_id,
        "role": "assistant",
        "content": content,
        "agent_name": agent_name,
        "metadata": sanitized_meta,
        "sources": sources or [],
        "confidence_score": confidence_score,
        "created_at": now
    }
    if client_message_id:
        doc["reply_to_client_id"] = client_message_id

    try:
        res = messages_col.insert_one(doc)
        return str(res.inserted_id)
    except pymongo.errors.DuplicateKeyError:
        existing = messages_col.find_one({
            "user_id": user_id,
            "session_id": session_id,
            "reply_to_client_id": client_message_id,
            "role": "assistant"
        })
        return str(existing["_id"]) if existing else None


def get_existing_assistant_reply(
    user_id: str,
    session_id: str,
    client_message_id: str,
    db_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieves previously stored assistant response for an idempotent request retry.
    """
    if not user_id or not session_id or not client_message_id:
        return None
    messages_col = get_chat_messages_collection(db_name)
    doc = messages_col.find_one({
        "user_id": user_id,
        "session_id": session_id,
        "reply_to_client_id": client_message_id,
        "role": "assistant"
    })
    return doc


def get_user_sessions(
    user_id: str,
    limit: int = 50,
    db_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Returns all chat sessions belonging to user_id, sorted by updated_at descending.
    """
    if not user_id:
        return []

    sessions_col = get_chat_sessions_collection(db_name)
    cursor = sessions_col.find(
        {"user_id": user_id},
        {"_id": 0, "session_id": 1, "title": 1, "created_at": 1, "updated_at": 1}
    ).sort("updated_at", pymongo.DESCENDING).limit(limit)

    results = []
    for doc in cursor:
        created_at = doc.get("created_at")
        updated_at = doc.get("updated_at")
        results.append({
            "session_id": doc.get("session_id"),
            "title": doc.get("title") or "New Conversation",
            "created_at": created_at.isoformat() if isinstance(created_at, datetime) else str(created_at or ""),
            "updated_at": updated_at.isoformat() if isinstance(updated_at, datetime) else str(updated_at or "")
        })
    return results


def get_session_messages(
    user_id: str,
    session_id: str,
    db_name: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Returns all messages for a given (user_id, session_id) sorted by created_at ascending.
    Returns None if the session does not belong to the user or does not exist.
    Preserves unique MongoDB message_id for stable client-side deduplication.
    """
    if not user_id or not session_id:
        return None

    sessions_col = get_chat_sessions_collection(db_name)
    session_exists = sessions_col.find_one({"user_id": user_id, "session_id": session_id})
    if not session_exists:
        return None

    messages_col = get_chat_messages_collection(db_name)
    cursor = messages_col.find(
        {"user_id": user_id, "session_id": session_id}
    ).sort("created_at", pymongo.ASCENDING)

    messages = []
    for doc in cursor:
        created_at = doc.get("created_at")
        messages.append({
            "message_id": str(doc.get("_id")),
            "client_message_id": doc.get("client_message_id") or doc.get("reply_to_client_id"),
            "role": doc.get("role", "user"),
            "content": doc.get("content", ""),
            "agent_name": doc.get("agent_name"),
            "metadata": doc.get("metadata") or {},
            "sources": doc.get("sources") or [],
            "confidence_score": doc.get("confidence_score"),
            "created_at": created_at.isoformat() if isinstance(created_at, datetime) else str(created_at or "")
        })
    return messages


def delete_user_session(
    user_id: str,
    session_id: str,
    db_name: Optional[str] = None
) -> bool:
    """
    Deletes a conversation session and all its messages for the specified user_id.
    Returns True if found and deleted, False if not found.
    """
    if not user_id or not session_id:
        return False

    sessions_col = get_chat_sessions_collection(db_name)
    res = sessions_col.delete_one({"user_id": user_id, "session_id": session_id})
    if res.deleted_count == 0:
        return False

    messages_col = get_chat_messages_collection(db_name)
    messages_col.delete_many({"user_id": user_id, "session_id": session_id})
    return True
