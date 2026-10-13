import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth.jwt import get_current_user
from app.db.chat_history import (
    get_user_sessions,
    get_session_messages,
    delete_user_session
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat/history", tags=["Chat History"])


class SessionSummaryResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str


class MessageItemResponse(BaseModel):
    role: str
    content: str
    agent_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    sources: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    created_at: str


class DeleteSessionResponse(BaseModel):
    status: str
    message: str


@router.get("", response_model=List[SessionSummaryResponse])
async def list_chat_history(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Returns all past chat sessions for the authenticated user, sorted by updated_at descending.
    Enforces strict user isolation using user_id from verified JWT.
    """
    user_id = str(current_user["_id"])
    sessions = get_user_sessions(user_id=user_id)
    return sessions


@router.get("/{session_id}", response_model=List[MessageItemResponse])
async def get_chat_session_messages(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Returns all messages for a specific conversation session belonging to the authenticated user.
    Returns 404 if the session does not exist or does not belong to the user.
    """
    user_id = str(current_user["_id"])
    messages = get_session_messages(user_id=user_id, session_id=session_id)
    if messages is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation session not found."
        )
    return messages


@router.delete("/{session_id}", response_model=DeleteSessionResponse)
async def delete_chat_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Deletes a conversation session and all associated messages for the authenticated user.
    Returns 404 if the session does not exist or does not belong to the user.
    """
    user_id = str(current_user["_id"])
    deleted = delete_user_session(user_id=user_id, session_id=session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation session not found."
        )
    return DeleteSessionResponse(
        status="success",
        message="Conversation deleted successfully."
    )
