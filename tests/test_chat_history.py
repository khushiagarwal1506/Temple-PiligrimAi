import pytest
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import httpx

from app.main import app
from app.auth.jwt import create_access_token
from app.db.mongodb import (
    get_database,
    get_users_collection,
    get_chat_sessions_collection,
    get_chat_messages_collection
)
from app.db.chat_history import (
    generate_conversation_title,
    save_user_message,
    save_assistant_message,
    get_user_sessions,
    get_session_messages,
    delete_user_session
)

class SyncTestClient:
    """Synchronous test client wrapper around httpx.AsyncClient for FastAPI ASGI app."""
    def __init__(self, asgi_app):
        self.app = asgi_app

    def post(self, url: str, json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None):
        async def _req():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://testserver") as client:
                return await client.post(url, json=json, headers=headers)
        return asyncio.run(_req())

    def get(self, url: str, headers: Optional[Dict[str, str]] = None):
        async def _req():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://testserver") as client:
                return await client.get(url, headers=headers)
        return asyncio.run(_req())

    def delete(self, url: str, headers: Optional[Dict[str, str]] = None):
        async def _req():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://testserver") as client:
                return await client.delete(url, headers=headers)
        return asyncio.run(_req())

client = SyncTestClient(app)
TEST_DB_NAME = "pilgrim_ai"


def create_mock_user(email: str, name: str = "Test Pilgrim"):
    users_col = get_users_collection(TEST_DB_NAME)
    user_doc = {
        "email": email,
        "name": name,
        "password_hash": "dummyhash",
        "is_verified": True,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    }
    res = users_col.insert_one(user_doc)
    user_id = str(res.inserted_id)
    token = create_access_token({"sub": user_id, "email": email, "role": "user"})
    return user_id, token


# =====================================================================
# 1. Deterministic Title Generation Tests
# =====================================================================

def test_generate_conversation_title():
    assert generate_conversation_title("What are the timings of Yadadri temple?") == "Yadadri temple"
    assert generate_conversation_title("How far is Hyderabad from Yadadri?") == "Hyderabad from Yadadri"
    assert generate_conversation_title("Tell me about Ramappa Temple") == "Ramappa Temple"
    assert generate_conversation_title("Answer in Telugu: What are the timings of Basara?") == "Basara"
    assert generate_conversation_title("") == "New Conversation"
    
    # Truncation test
    long_q = "What are the specific guidelines, rules, dress codes, rituals, and ticket prices for the morning pooja at Srisailam Temple?"
    title = generate_conversation_title(long_q)
    assert len(title) <= 50
    assert title.endswith("...")


# =====================================================================
# 2. Database Persistence Functions Tests
# =====================================================================

def test_save_and_retrieve_user_messages():
    user_id = f"user_test_{uuid.uuid4().hex[:6]}"
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    # 1. Save user message
    save_user_message(
        user_id=user_id,
        session_id=session_id,
        content="What are the timings of Yadadri?",
        db_name=TEST_DB_NAME
    )

    # Verify session created
    sessions = get_user_sessions(user_id=user_id, db_name=TEST_DB_NAME)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id
    assert "Yadadri" in sessions[0]["title"]

    # 2. Save assistant message
    save_assistant_message(
        user_id=user_id,
        session_id=session_id,
        content="Yadadri temple opens at 04:00 AM and closes at 09:30 PM.",
        agent_name="temple_information",
        db_name=TEST_DB_NAME
    )

    # Verify messages
    messages = get_session_messages(user_id=user_id, session_id=session_id, db_name=TEST_DB_NAME)
    assert messages is not None
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What are the timings of Yadadri?"
    assert messages[1]["role"] == "assistant"
    assert "04:00 AM" in messages[1]["content"]
    assert messages[1]["agent_name"] == "temple_information"


def test_sessions_sorted_by_updated_at():
    user_id = f"user_test_sort_{uuid.uuid4().hex[:6]}"
    sess1 = f"session_1_{uuid.uuid4().hex[:6]}"
    sess2 = f"session_2_{uuid.uuid4().hex[:6]}"
    
    # Create Session 1
    save_user_message(user_id, sess1, "Question 1", db_name=TEST_DB_NAME)
    save_assistant_message(user_id, sess1, "Answer 1", db_name=TEST_DB_NAME)
    
    # Create Session 2
    save_user_message(user_id, sess2, "Question 2", db_name=TEST_DB_NAME)
    save_assistant_message(user_id, sess2, "Answer 2", db_name=TEST_DB_NAME)
    
    # Update Session 1 again (making it more recently updated)
    save_user_message(user_id, sess1, "Question 1 follow-up", db_name=TEST_DB_NAME)
    
    sessions = get_user_sessions(user_id=user_id, db_name=TEST_DB_NAME)
    assert len(sessions) == 2
    # sess1 should be first because it was updated most recently
    assert sessions[0]["session_id"] == sess1
    assert sessions[1]["session_id"] == sess2


def test_user_isolation_database():
    user_a = f"user_a_{uuid.uuid4().hex[:6]}"
    user_b = f"user_b_{uuid.uuid4().hex[:6]}"
    sess_a = f"sess_a_{uuid.uuid4().hex[:6]}"
    sess_b = f"sess_b_{uuid.uuid4().hex[:6]}"
    
    save_user_message(user_a, sess_a, "User A question", db_name=TEST_DB_NAME)
    save_user_message(user_b, sess_b, "User B question", db_name=TEST_DB_NAME)

    sessions_a = get_user_sessions(user_id=user_a, db_name=TEST_DB_NAME)
    sessions_b = get_user_sessions(user_id=user_b, db_name=TEST_DB_NAME)

    assert any(s["session_id"] == sess_a for s in sessions_a)
    assert not any(s["session_id"] == sess_b for s in sessions_a)

    assert any(s["session_id"] == sess_b for s in sessions_b)
    assert not any(s["session_id"] == sess_a for s in sessions_b)

    # User A cannot retrieve User B's session messages
    assert get_session_messages(user_id=user_a, session_id=sess_b, db_name=TEST_DB_NAME) is None


def test_delete_user_session():
    user_id = f"user_del_{uuid.uuid4().hex[:6]}"
    sess_id = f"sess_del_{uuid.uuid4().hex[:6]}"

    save_user_message(user_id, sess_id, "Delete me", db_name=TEST_DB_NAME)
    save_assistant_message(user_id, sess_id, "Sure", db_name=TEST_DB_NAME)

    assert get_session_messages(user_id, sess_id, db_name=TEST_DB_NAME) is not None
    deleted = delete_user_session(user_id, sess_id, db_name=TEST_DB_NAME)
    assert deleted is True

    # Check now deleted
    assert get_session_messages(user_id, sess_id, db_name=TEST_DB_NAME) is None
    assert not any(s["session_id"] == sess_id for s in get_user_sessions(user_id, db_name=TEST_DB_NAME))


# =====================================================================
# 3. HTTP API Endpoints Tests
# =====================================================================

def test_api_list_history_unauthenticated():
    res = client.get("/api/chat/history")
    assert res.status_code == 401


def test_api_list_history_authenticated_empty():
    user_id, token = create_mock_user(f"empty_{uuid.uuid4().hex[:6]}@example.com")
    res = client.get("/api/chat/history", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json() == []


def test_api_get_history_and_messages():
    user_id, token = create_mock_user(f"chat_tester_{uuid.uuid4().hex[:6]}@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"

    # Save messages directly in DB for this user
    save_user_message(user_id, session_id, "What are Yadadri timings?")
    save_assistant_message(user_id, session_id, "Yadadri timings are 04:00 AM to 09:30 PM.", "temple_information")

    # 1. GET /api/chat/history
    res = client.get("/api/chat/history", headers=headers)
    assert res.status_code == 200
    history = res.json()
    assert len(history) >= 1
    matching = [h for h in history if h["session_id"] == session_id]
    assert len(matching) == 1
    assert "Yadadri" in matching[0]["title"]

    # 2. GET /api/chat/history/{session_id}
    res_msgs = client.get(f"/api/chat/history/{session_id}", headers=headers)
    assert res_msgs.status_code == 200
    msgs = res_msgs.json()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "What are Yadadri timings?"
    assert msgs[1]["role"] == "assistant"
    assert "04:00 AM" in msgs[1]["content"]

    # 3. DELETE /api/chat/history/{session_id}
    del_res = client.delete(f"/api/chat/history/{session_id}", headers=headers)
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    # Verify 404 after deletion
    res_after = client.get(f"/api/chat/history/{session_id}", headers=headers)
    assert res_after.status_code == 404


def test_api_user_isolation_between_accounts():
    user_a_id, token_a = create_mock_user(f"user_a_{uuid.uuid4().hex[:6]}@example.com")
    user_b_id, token_b = create_mock_user(f"user_b_{uuid.uuid4().hex[:6]}@example.com")

    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    sess_a = f"sess_a_{uuid.uuid4().hex[:8]}"
    save_user_message(user_a_id, sess_a, "User A confidential inquiry")
    save_assistant_message(user_a_id, sess_a, "Answer for A", "temple_information")

    # User A can see their session
    res_a = client.get(f"/api/chat/history/{sess_a}", headers=headers_a)
    assert res_a.status_code == 200

    # User B CANNOT see User A's session -> must return 404
    res_b = client.get(f"/api/chat/history/{sess_a}", headers=headers_b)
    assert res_b.status_code == 404

    # User B CANNOT delete User A's session -> must return 404
    del_b = client.delete(f"/api/chat/history/{sess_a}", headers=headers_b)
    assert del_b.status_code == 404

    # User A's session still exists
    res_a_check = client.get(f"/api/chat/history/{sess_a}", headers=headers_a)
    assert res_a_check.status_code == 200


def test_missing_session_returns_safe_404():
    user_id, token = create_mock_user(f"user_nonexistent_{uuid.uuid4().hex[:6]}@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    fake_session = "non_existent_session_id_999"

    res = client.get(f"/api/chat/history/{fake_session}", headers=headers)
    assert res.status_code == 404
    assert res.json()["detail"] == "Conversation session not found."

    del_res = client.delete(f"/api/chat/history/{fake_session}", headers=headers)
    assert del_res.status_code == 404
    assert del_res.json()["detail"] == "Conversation session not found."


def test_chat_continuation_and_multi_turn_persistence():
    user_id, token = create_mock_user(f"multi_turn_{uuid.uuid4().hex[:6]}@example.com")
    sess_id = f"session_cont_{uuid.uuid4().hex[:8]}"

    # Turn 1
    save_user_message(user_id, sess_id, "What are the timings of Yadadri?", db_name=TEST_DB_NAME)
    save_assistant_message(user_id, sess_id, "Yadadri is open from 04:00 AM to 09:30 PM.", "temple_information", db_name=TEST_DB_NAME)

    # Turn 2 (Continuation of old chat)
    save_user_message(user_id, sess_id, "What about accommodation options nearby?", db_name=TEST_DB_NAME)
    save_assistant_message(user_id, sess_id, "Haritha Hotel and Yogananda Nilayam are available.", "temple_information", db_name=TEST_DB_NAME)

    # Turn 3
    save_user_message(user_id, sess_id, "What is the room tariff?", db_name=TEST_DB_NAME)
    save_assistant_message(user_id, sess_id, "Tariffs range from ₹500 to ₹2500 per night.", "budget", db_name=TEST_DB_NAME)

    messages = get_session_messages(user_id, sess_id, db_name=TEST_DB_NAME)
    assert len(messages) == 6
    assert messages[0]["content"] == "What are the timings of Yadadri?"
    assert messages[2]["content"] == "What about accommodation options nearby?"
    assert messages[4]["content"] == "What is the room tariff?"
    assert messages[5]["agent_name"] == "budget"

