import logging
import uuid
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.rag.chat import PilgrimRAGService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

class ChatRequest(BaseModel):
    question: str = Field(..., description="User query about temple details", json_schema_extra={"example": "What are the timings of Basara Temple?"})
    session_id: Optional[str] = Field(default=None, description="Optional conversation session identifier for multi-turn context")
    client_message_id: Optional[str] = Field(default=None, description="Unique client-generated message identifier for idempotency")

class ChatResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: int
    sources: List[str]
    confidence_score: float
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    message_id: Optional[str] = None
    client_message_id: Optional[str] = None

_service_instance: Optional[Any] = None
_coordinator_instance: Optional[Any] = None

def get_rag_service() -> Any:
    global _service_instance
    if _service_instance is None:
        try:
            from app.rag.chat import PilgrimRAGService
            _service_instance = PilgrimRAGService()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Local vector DB or BM25 index not found. Please run ingestion first. Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize local RAG service: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error initializing local RAG service: {str(e)}"
            )
    return _service_instance

def get_coordinator() -> Any:
    global _coordinator_instance
    if _coordinator_instance is None:
        try:
            from app.agents.coordinator import MasterAgentCoordinator
            from app.agents.temple_information_agent import TempleInformationAgent
            from app.agents.comparison_agent import ComparisonAgent
            from app.agents.cross_temple_aggregation_agent import CrossTempleAggregationAgent
            from app.agents.itinerary_planning_agent import ItineraryPlanningAgent
            from app.agents.live_weather_agent import LiveWeatherAgent
            from app.agents.budget_agent import BudgetAgent
            from app.agents.live_navigation_agent import LiveNavigationAgent
            from app.agents.voice_multilingual_agent import VoiceMultilingualAgent, LocalTranslationProvider
            from app.agents.emergency_agent import EmergencyAgent
            from app.agents.recommendation_agent import RecommendationAgent
            from app.agents.memory import ConversationMemoryManager
            from app.agents.router import AgentRouter
            from app.agents.response_aggregator import ResponseAggregator

            memory_manager = ConversationMemoryManager()
            router = AgentRouter()
            aggregator = ResponseAggregator()
            temple_info = TempleInformationAgent()
            comparison = ComparisonAgent()
            cross_temple = CrossTempleAggregationAgent()
            itinerary = ItineraryPlanningAgent()
            weather = LiveWeatherAgent()
            budget = BudgetAgent()
            navigation = LiveNavigationAgent(memory_manager=memory_manager)
            voice_multilingual = VoiceMultilingualAgent(translation_provider=LocalTranslationProvider())
            emergency = EmergencyAgent()
            recommendation = RecommendationAgent(weather_agent=weather)

            _coordinator_instance = MasterAgentCoordinator(
                router=router,
                aggregator=aggregator,
                temple_info_agent=temple_info,
                comparison_agent=comparison,
                cross_temple_agent=cross_temple,
                itinerary_agent=itinerary,
                live_weather_agent=weather,
                budget_agent=budget,
                live_navigation_agent=navigation,
                voice_multilingual_agent=voice_multilingual,
                emergency_agent=emergency,
                recommendation_agent=recommendation,
                memory_manager=memory_manager
            )
        except Exception as e:
            logger.error(f"Failed to initialize MasterAgentCoordinator: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error initializing agent coordinator: {str(e)}"
            )
    return _coordinator_instance

def set_coordinator(coordinator: Optional[Any]) -> None:
    """Sets or overrides the coordinator instance (used for testing / dependency injection)."""
    global _coordinator_instance
    _coordinator_instance = coordinator

from app.auth.jwt import get_optional_current_user
from app.db.chat_history import (
    save_user_message,
    save_assistant_message,
    get_existing_assistant_reply,
    claim_user_message_turn,
    wait_for_assistant_reply
)

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user)
):
    """
    POST /chat endpoint to receive user questions and return AI answers
    routed through the MasterAgentCoordinator multi-agent architecture.
    Optionally associates authenticated user without modifying RAG or response contracts.
    Supports client_message_id for idempotent retry prevention and stable deduplication.
    """
    cleaned_question = request.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question field cannot be empty."
        )

    user_info_str = f" [User: {current_user.get('email')}]" if current_user else " [Anonymous]"
    logger.info(f"[CHAT ENDPOINT]{user_info_str} Received question: {cleaned_question[:80]}")

    session_id = request.session_id.strip() if request.session_id and request.session_id.strip() else str(uuid.uuid4())
    client_msg_id = request.client_message_id.strip() if request.client_message_id and request.client_message_id.strip() else None

    # Idempotency and atomic claim check under true concurrency:
    if current_user and client_msg_id:
        existing_reply = get_existing_assistant_reply(
            user_id=str(current_user["_id"]),
            session_id=session_id,
            client_message_id=client_msg_id
        )
        if existing_reply:
            logger.info(f"[CHAT ENDPOINT] Returning cached idempotent reply for client_message_id={client_msg_id}")
            return ChatResponse(
                question=cleaned_question,
                answer=existing_reply.get("content", ""),
                retrieved_chunks=len(existing_reply.get("sources", [])),
                sources=existing_reply.get("sources", []),
                confidence_score=existing_reply.get("confidence_score") or 1.0,
                agent_name=existing_reply.get("agent_name") or "coordinator",
                session_id=session_id,
                latency_ms=0.0,
                metadata=existing_reply.get("metadata", {}),
                message_id=str(existing_reply.get("_id")),
                client_message_id=client_msg_id
            )

        # Atomic claim: Only the winning concurrent request executes coordinator
        is_owner, _ = claim_user_message_turn(
            user_id=str(current_user["_id"]),
            session_id=session_id,
            content=cleaned_question,
            client_message_id=client_msg_id
        )
        if not is_owner:
            logger.info(f"[CHAT ENDPOINT] Concurrent request in-flight for client_message_id={client_msg_id}; waiting for primary execution.")
            reply_doc = wait_for_assistant_reply(
                user_id=str(current_user["_id"]),
                session_id=session_id,
                client_message_id=client_msg_id,
                timeout_seconds=30.0
            )
            if reply_doc:
                return ChatResponse(
                    question=cleaned_question,
                    answer=reply_doc.get("content", ""),
                    retrieved_chunks=len(reply_doc.get("sources", [])),
                    sources=reply_doc.get("sources", []),
                    confidence_score=reply_doc.get("confidence_score") or 1.0,
                    agent_name=reply_doc.get("agent_name") or "coordinator",
                    session_id=session_id,
                    latency_ms=0.0,
                    metadata=reply_doc.get("metadata", {}),
                    message_id=str(reply_doc.get("_id")),
                    client_message_id=client_msg_id
                )
    elif current_user:
        try:
            save_user_message(
                user_id=str(current_user["_id"]),
                session_id=session_id,
                content=cleaned_question,
                client_message_id=None
            )
        except Exception as e:
            logger.warning(f"[CHAT HISTORY] Could not persist user message: {e}")

    coordinator = get_coordinator()

    try:
        from app.agents.state import AgentState
        state = AgentState(user_query=cleaned_question, session_id=session_id)
        agent_resp = coordinator.process(state)

        # Extract retrieved chunks count safely
        retrieved_chunks = 0
        if "retrieved_chunks" in agent_resp.metadata:
            try:
                retrieved_chunks = int(agent_resp.metadata["retrieved_chunks"])
            except (ValueError, TypeError):
                retrieved_chunks = len(agent_resp.sources)
        elif agent_resp.sources:
            retrieved_chunks = len(agent_resp.sources)

        # Extract confidence score safely
        confidence_score = 1.0
        if "confidence_score" in agent_resp.metadata:
            try:
                confidence_score = float(agent_resp.metadata["confidence_score"])
            except (ValueError, TypeError):
                confidence_score = 1.0
        elif "route_decision" in state.intermediate_results:
            rd = state.intermediate_results["route_decision"]
            if hasattr(rd, "confidence") and rd.confidence is not None:
                confidence_score = float(rd.confidence)

        confidence_score = max(0.0, float(confidence_score))

        logger.info(
            f"[CHAT RESPONSE] session_id={session_id} agent={agent_resp.agent_name} "
            f"constraints={agent_resp.metadata.get('structured_constraints')}"
        )

        # Save assistant response to persistent MongoDB chat history if authenticated
        saved_msg_id = None
        if current_user and agent_resp and agent_resp.answer:
            try:
                saved_msg_id = save_assistant_message(
                    user_id=str(current_user["_id"]),
                    session_id=session_id,
                    content=agent_resp.answer,
                    agent_name=agent_resp.agent_name,
                    metadata=agent_resp.metadata,
                    sources=agent_resp.sources,
                    confidence_score=confidence_score,
                    client_message_id=client_msg_id
                )
            except Exception as e:
                logger.warning(f"[CHAT HISTORY] Could not persist assistant message: {e}")

        return ChatResponse(
            question=cleaned_question,
            answer=agent_resp.answer,
            retrieved_chunks=retrieved_chunks,
            sources=agent_resp.sources,
            confidence_score=confidence_score,
            agent_name=agent_resp.agent_name or "coordinator",
            session_id=session_id,
            latency_ms=agent_resp.latency_ms,
            metadata=agent_resp.metadata,
            message_id=saved_msg_id,
            client_message_id=client_msg_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request through coordinator: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        )
