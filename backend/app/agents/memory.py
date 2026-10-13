"""
PilgrimAI - In-Memory Conversation Memory Manager (Phase I)

Maintains short-lived conversational context across turns within distinct user sessions.
Supports deterministic pronoun and anaphora resolution without external LLM dependencies.

Core Principles:
- 100% Deterministic (pure in-memory rule-based entity tracking and resolution).
- Zero external databases (Redis, Postgres, Mongo are strictly avoided).
- Zero LLM generation / classification calls.
- Zero raw RAG, vector store, or retrieval dependency.
- Strict bounded session capacity (LRU eviction at 1000 sessions).
- Time-to-Live (TTL) expiration per session (default: 30 minutes).
- Complete session isolation (zero cross-session context leakage).
- Sensitive credential and secret sanitization.
"""

import re
import copy
import time
import logging
from collections import OrderedDict
from typing import Dict, Any, List, Optional, Tuple, Callable, Set
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

# Constants
MAX_SESSIONS_DEFAULT = 1000
DEFAULT_TTL_SECONDS = 1800.0  # 30 minutes

# Sensitive keys to redact
_SENSITIVE_KEYS = {
    "api_key", "secret", "password", "token", "access_token",
    "auth", "authorization", "private_key", "credential"
}


def _sanitize_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitizes metadata dictionary removing sensitive keys and truncating long texts."""
    if not meta:
        return {}
    sanitized = {}
    for k, v in meta.items():
        k_str = str(k).lower()
        if any(sens in k_str for sens in _SENSITIVE_KEYS):
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, str) and len(v) > 1000:
            sanitized[k] = v[:1000] + "... [TRUNCATED]"
        elif isinstance(v, dict):
            sanitized[k] = _sanitize_meta(v)
        else:
            sanitized[k] = v
    return sanitized


class MemoryTurn(BaseModel):
    """
    Lightweight structured representation of an individual conversation turn.
    Stores conversational entities (temples, intents) rather than raw document text.
    """
    model_config = ConfigDict(extra="allow")

    session_id: str = Field(..., description="Unique conversation session identifier")
    user_query: str = Field(..., description="Original query as submitted by user")
    resolved_query: str = Field(..., description="Query after anaphora/pronoun resolution")
    detected_intents: List[str] = Field(default_factory=list, description="Classified intents for turn")
    identified_temples: List[str] = Field(default_factory=list, description="Canonical temples identified")
    timestamp: float = Field(..., description="Unix timestamp of the turn")
    response_summary: str = Field(default="", description="Brief summary of response text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Sanitized turn metadata")


def _get_combined_temple_aliases() -> Dict[str, List[str]]:
    """Helper to retrieve unified temple aliases from planner and coordinate registry."""
    from app.rag.query_planner import query_planner
    from app.agents.coordinate_registry import _CANONICAL_TEMPLE_ALIASES
    combined: Dict[str, Set[str]] = {}
    for c, aliases in query_planner.temple_aliases.items():
        combined.setdefault(c, set()).update(aliases)
    for c, aliases in _CANONICAL_TEMPLE_ALIASES.items():
        target_key = c
        for k in combined:
            if k.lower() == c.lower():
                target_key = k
                break
        combined.setdefault(target_key, set()).update(aliases)
    return {k: sorted(list(v), key=len, reverse=True) for k, v in combined.items()}


class ConversationMemoryManager:
    """
    In-memory, session-isolated conversation context manager.
    Enforces LRU session bounds, TTL expiration, and deterministic pronoun resolution.
    """

    def __init__(
        self,
        max_sessions: int = MAX_SESSIONS_DEFAULT,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        time_provider: Optional[Callable[[], float]] = None
    ):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._time_provider = time_provider or time.time
        # OrderedDict used for LRU tracking: key -> session dict
        self._sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def _now(self) -> float:
        return float(self._time_provider())

    def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieves existing session (verifying TTL) or creates a new one.
        Enforces LRU eviction if capacity is reached.
        """
        now = self._now()

        if session_id in self._sessions:
            session = self._sessions[session_id]
            # Check TTL expiration
            if (now - session["last_accessed"]) > self.ttl_seconds:
                # Expired session -> remove and reset
                del self._sessions[session_id]
            else:
                # Refresh access time and move to end (MRU position)
                session["last_accessed"] = now
                self._sessions.move_to_end(session_id)
                return session

        # Enforce LRU eviction if capacity reached
        while len(self._sessions) >= self.max_sessions:
            # Evict least recently used session (first item in OrderedDict)
            evicted_id, _ = self._sessions.popitem(last=False)
            logger.debug(f"Evicted LRU session: {evicted_id}")

        new_session = {
            "session_id": session_id,
            "created_at": now,
            "last_accessed": now,
            "turns": [],
            "pending_clarification": None,
            "session_origin": None,
            "trip_context": {}
        }
        self._sessions[session_id] = new_session
        return new_session

    def set_session_origin(self, session_id: str, origin: str) -> None:
        """Stores the verified starting origin for the session."""
        if not session_id or not origin:
            return
        session = self._get_or_create_session(session_id)
        session["session_origin"] = origin.strip()
        if "trip_context" not in session:
            session["trip_context"] = {}
        session["trip_context"]["origin"] = origin.strip()
        session["last_accessed"] = self._now()
        self._sessions.move_to_end(session_id)

    def get_session_origin(self, session_id: str) -> Optional[str]:
        """Retrieves verified starting origin for the session if available and active."""
        if not session_id or session_id not in self._sessions:
            return None
        session = self._sessions[session_id]
        if (self._now() - session["last_accessed"]) > self.ttl_seconds:
            del self._sessions[session_id]
            return None
        return session.get("session_origin") or session.get("trip_context", {}).get("origin")

    def clear_session_origin(self, session_id: str) -> None:
        """Clears the starting origin for the session."""
        if session_id in self._sessions:
            self._sessions[session_id]["session_origin"] = None
            if "trip_context" in self._sessions[session_id]:
                self._sessions[session_id]["trip_context"].pop("origin", None)

    def get_trip_context(self, session_id: str) -> Dict[str, Any]:
        """Retrieves verified shared trip context for the session."""
        if not session_id or session_id not in self._sessions:
            return {}
        session = self._sessions[session_id]
        if (self._now() - session["last_accessed"]) > self.ttl_seconds:
            del self._sessions[session_id]
            return {}
        return dict(session.get("trip_context", {}))

    def update_trip_context(self, session_id: str, updates: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Updates shared trip context with verified parameters, preserving unchanged values."""
        if not session_id:
            return
        combined = {}
        if updates and isinstance(updates, dict):
            combined.update(updates)
        combined.update(kwargs)
        if not combined:
            return
        session = self._get_or_create_session(session_id)
        if "trip_context" not in session:
            session["trip_context"] = {}
        for k, v in combined.items():
            if v is not None:
                session["trip_context"][k] = v
        if "origin" in combined and combined["origin"]:
            session["session_origin"] = combined["origin"]
        session["last_accessed"] = self._now()
        self._sessions.move_to_end(session_id)

    def replace_trip_context(self, session_id: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Replaces the shared trip context for the session with a fresh trip context."""
        if not session_id:
            return
        combined = {}
        if context and isinstance(context, dict):
            combined.update(context)
        combined.update(kwargs)
        session = self._get_or_create_session(session_id)
        session["trip_context"] = dict(combined)
        if "origin" in combined and combined["origin"] and not combined.get("origin_assumed", True):
            session["session_origin"] = combined["origin"]
        elif combined.get("origin_assumed", True):
            session["session_origin"] = None
        session["last_accessed"] = self._now()
        self._sessions.move_to_end(session_id)

    def clear_trip_context(self, session_id: str) -> None:
        """Clears the shared trip context for the session."""
        if session_id in self._sessions:
            self._sessions[session_id]["trip_context"] = {}

    def set_pending_clarification(
        self,
        session_id: str,
        pending_question: str,
        pending_intent: str = "temple_information",
        pending_sub_intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Stores pending intent context when clarification is asked from user.
        """
        session = self._get_or_create_session(session_id)
        now = self._now()
        safe_meta = _sanitize_meta(metadata)
        session["pending_clarification"] = {
            "pending_intent": pending_intent,
            "pending_sub_intent": pending_sub_intent,
            "pending_slot": "temple",
            "pending_question": pending_question,
            "timestamp": now,
            "metadata": safe_meta
        }
        session["last_accessed"] = now
        self._sessions.move_to_end(session_id)

    def get_pending_clarification(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves defensive copy of pending clarification context for session if active.
        """
        now = self._now()
        if session_id not in self._sessions:
            return None
        session = self._sessions[session_id]
        if (now - session["last_accessed"]) > self.ttl_seconds:
            del self._sessions[session_id]
            return None
        pending = session.get("pending_clarification")
        return copy.deepcopy(pending) if pending else None

    def clear_pending_clarification(self, session_id: str) -> None:
        """
        Clears pending clarification context for session.
        """
        if session_id in self._sessions:
            self._sessions[session_id]["pending_clarification"] = None

    def add_turn(
        self,
        session_id: str,
        user_query: str,
        resolved_query: Optional[str] = None,
        detected_intents: Optional[List[str]] = None,
        identified_temples: Optional[List[str]] = None,
        response_summary: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryTurn:
        """
        Records a completed conversation turn for the given session.
        """
        session = self._get_or_create_session(session_id)
        now = self._now()
        safe_meta = _sanitize_meta(metadata)

        turn = MemoryTurn(
            session_id=session_id,
            user_query=user_query,
            resolved_query=resolved_query or user_query,
            detected_intents=detected_intents or [],
            identified_temples=identified_temples or [],
            timestamp=now,
            response_summary=response_summary[:300] if response_summary else "",
            metadata=safe_meta
        )

        session["turns"].append(turn)
        session["last_accessed"] = now
        self._sessions.move_to_end(session_id)
        return turn

    record_turn = add_turn

    def get_recent_turns(self, session_id: str, limit: int = 5) -> List[MemoryTurn]:
        """
        Returns chronological list of recent turns for session, up to limit.
        Returns defensive copies to prevent caller mutation of internal state.
        """
        now = self._now()
        if session_id not in self._sessions:
            return []

        session = self._sessions[session_id]
        if (now - session["last_accessed"]) > self.ttl_seconds:
            # Expired
            del self._sessions[session_id]
            return []

        # Refresh access time and move to end
        session["last_accessed"] = now
        self._sessions.move_to_end(session_id)

        turns = session["turns"][-limit:] if limit > 0 else []
        return [copy.deepcopy(t) for t in turns]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns a defensive copy of the session state for testing and diagnostics.
        """
        now = self._now()
        if session_id not in self._sessions:
            return None

        session = self._sessions[session_id]
        if (now - session["last_accessed"]) > self.ttl_seconds:
            del self._sessions[session_id]
            return None

        session["last_accessed"] = now
        self._sessions.move_to_end(session_id)
        return copy.deepcopy(session)

    def clear_session(self, session_id: str) -> bool:
        """
        Clears memory for a specific session. Safe no-op if session does not exist.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def clear_all(self) -> None:
        """
        Clears all in-memory sessions.
        """
        self._sessions.clear()

    @staticmethod
    def _detect_entity_only_temple(query: str, temple_aliases: Dict[str, List[str]]) -> Optional[str]:
        """
        Determines if query is an entity-only response providing a temple name.
        """
        q_clean = query.strip()
        if not q_clean:
            return None

        # Remove leading conversational / prepositional phrases
        normalized = re.sub(
            r'^(it\s+is\s+|it\'s\s+|about\s+|at\s+|in\s+|for\s+|to\s+|the\s+|i\s+mean\s+|i\s+want\s+)+',
            '',
            q_clean,
            flags=re.IGNORECASE
        ).strip()

        # Remove trailing polite / temple filler words
        stripped = re.sub(
            r'\s+(temple|temples|devasthanam|kovil|mandir|please|thanks|thank\s+you)+$',
            '',
            normalized,
            flags=re.IGNORECASE
        ).strip()

        # Check word count of stripped entity (must be concise, <= 5 words)
        words = stripped.split()
        if len(words) > 5:
            return None

        # Guard against full question sentences or action verbs
        question_words = [
            r'\b(what|where|how|when|why|who|which|can|could|is|are|do|does|will|would|plan|compare|weather|cost|distance|reach)\b'
        ]
        if any(re.search(pat, stripped, re.IGNORECASE) for pat in question_words):
            return None

        stripped_lower = stripped.lower()
        for canonical, aliases in temple_aliases.items():
            for alias in sorted(aliases, key=len, reverse=True):
                pattern = r'^\b' + re.escape(alias) + r'\b$'
                if re.search(pattern, stripped_lower) or stripped_lower == alias.lower():
                    return canonical

        return None

    @staticmethod
    def _combine_question_with_temple(pending_question: str, temple: str) -> str:
        """
        Deterministically combines an unresolved question with the specified temple.
        """
        base_q = pending_question.strip().rstrip("?. ")
        # If question has placeholder like "at the temple", "in the temple", "at temple"
        if re.search(r'\b(at|in|of)\s+(the\s+)?temple\b', base_q, re.IGNORECASE):
            resolved = re.sub(
                r'\b(at|in|of)\s+(the\s+)?temple\b',
                f'at {temple} temple',
                base_q,
                flags=re.IGNORECASE
            ) + '?'
        elif re.search(r'\b(at|in|of)\b$', base_q, re.IGNORECASE):
            resolved = f"{base_q} {temple} temple?"
        else:
            resolved = f"{base_q} at {temple}?"
        return resolved

    def resolve_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Deterministically resolves pronouns, anaphoric references, and entity follow-ups using recent conversation context.

        Returns:
            Dict containing:
                - "resolved_query": str
                - "is_ambiguous": bool
                - "referenced_temple": Optional[str]
                - "clarification_message": Optional[str]
                - "resolved_from_pending": Optional[bool]
                - "original_pending_question": Optional[str]
        """
        clean_query = query.strip()
        temple_aliases = _get_combined_temple_aliases()

        # 1. Check for active pending clarification in session (e.g. Turn 1 asked for dress code without temple)
        pending_clarification = self.get_pending_clarification(session_id)
        if pending_clarification:
            matched_temple = self._detect_entity_only_temple(clean_query, temple_aliases)
            if matched_temple:
                # Resolve pending question with the identified temple
                pending_q = pending_clarification["pending_question"]
                resolved_q = self._combine_question_with_temple(pending_q, matched_temple)
                self.clear_pending_clarification(session_id)
                return {
                    "resolved_query": resolved_q,
                    "is_ambiguous": False,
                    "referenced_temple": matched_temple,
                    "clarification_message": None,
                    "resolved_from_pending": True,
                    "original_pending_question": pending_q
                }
            else:
                # If user provided a different question or non-temple input, clear pending clarification
                self.clear_pending_clarification(session_id)

        recent_turns = self.get_recent_turns(session_id, limit=3)

        # 2. Check if the query itself explicitly names a registered or unregistered temple
        explicit_temples = []
        q_norm = clean_query.lower()
        for canonical, aliases in temple_aliases.items():
            for alias in aliases:
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, q_norm):
                    explicit_temples.append(canonical)
                    break

        if explicit_temples:
            # Explicit new temple in the current turn overrides all memory
            return {
                "resolved_query": clean_query,
                "is_ambiguous": False,
                "referenced_temple": explicit_temples[0],
                "clarification_message": None
            }

        # 3. Extract temples mentioned across recent turns
        recent_temples_ordered: List[str] = []
        for t in reversed(recent_turns):
            for tm in t.identified_temples:
                if tm and tm not in recent_temples_ordered:
                    recent_temples_ordered.append(tm)

        # 4. Check for pronoun and anaphoric reference triggers
        # Examples: "its", "it", "this temple", "that temple", "the temple", "there", "this trip"
        pronoun_patterns = [
            (r"\b(its|it's)\b", "possessive"),
            (r"\b(this|that|the)\s+(?:temple|trip|visit|tour|itinerary|plan)\b", "demonstrative"),
            (r"\bthere\b", "locative"),
            (r"\bit\b", "neutral_pronoun")
        ]

        has_reference = False
        ref_type = None
        for pat, r_type in pronoun_patterns:
            if re.search(pat, clean_query, re.IGNORECASE):
                has_reference = True
                ref_type = r_type
                break

        # If query has no pronouns, check budget, ellipsis, and temple-topic follow-up patterns
        if not has_reference:
            # Check budget follow-up queries like "what about for 2 people?", "budget for one person", "for 4 people", "change my budget to 3000"
            budget_followup_pattern = (
                r"\b(?:what\s+about\s+(?:for\s+)?|how\s+about\s+(?:for\s+)?|budget\s+for\s+|and\s+for\s+|for\s+)"
                r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(people|person|persons|adults)\b|"
                r"\b(?:change|update|set|make|increase|decrease|reduce|my)?\s*budget\s*(?:to|is|of|as|at)?\s*(?:₹|rs\.?|inr)?\s*\d+\b"
            )
            if re.search(budget_followup_pattern, clean_query, re.IGNORECASE) and recent_temples_ordered:
                has_reference = True
                ref_type = "budget_followup"
            else:
                # Check ellipsis queries like "What about sevas?", "And timings?", "What about the dress code?"
                ellipsis_pattern = r"\b(what\s+about|how\s+about|and|also)\s+(?:the\s+)?(sevas?|timings?|facilities|rooms?|accommodation|accomodation|history|dress\s+code|budget|cost|price|prasadam|poojas?|pujas?|darshan)\b"
                if re.search(ellipsis_pattern, clean_query, re.IGNORECASE) and recent_temples_ordered:
                    has_reference = True
                    ref_type = "ellipsis"
                else:
                    # Check temple topic follow-up queries like "what are the sevas?", "what is the dress code?", "tell me timings"
                    is_weather_or_comp = (
                        re.search(r"\b(weather|temperature|forecast|rain|climate|compare|comparison|versus|vs\.?)\b", clean_query, re.IGNORECASE) or
                        re.search(r"(?:above|below|over|under)\s+\d+", clean_query, re.IGNORECASE) or
                        re.search(r"\b(?:list|which|all)\s+temples\b", clean_query, re.IGNORECASE)
                    )
                    temple_topic_pattern = r"\b(?:what\s+are|what\s+is|tell\s+me\s+about|tell\s+me|show\s+me|give\s+me|are\s+there|is\s+there)?\s*(?:the\s+)?(sevas?|timings?|darshan(?:\s+timings?)?|facilities|rooms?|accommodation|accomodation|history|dress\s+code|prasadam|poojas?|pujas?|harathi|abhishekam|rules|dos\s+and\s+don'ts)\b"
                    if not is_weather_or_comp and re.search(temple_topic_pattern, clean_query, re.IGNORECASE) and recent_temples_ordered:
                        has_reference = True
                        ref_type = "temple_topic_followup"
                    else:
                        return {
                            "resolved_query": clean_query,
                            "is_ambiguous": False,
                            "referenced_temple": None,
                            "clarification_message": None
                        }

        # 4.5 If there is an active multi-temple trip context and the query is a trip-wide adjustment:
        session = self._sessions.get(session_id, {})
        trip_context = session.get("trip_context", {})
        is_trip_adjustment = bool(re.search(r"\b(?:budget|pax|people|persons?|days?|nights?)\b", clean_query, re.IGNORECASE))
        if trip_context and trip_context.get("destinations") and is_trip_adjustment:
            destinations = trip_context["destinations"]
            return {
                "resolved_query": f"{clean_query} for {' and '.join(destinations)}",
                "is_ambiguous": False,
                "referenced_temple": destinations[0],
                "clarification_message": None
            }

        # 5. Handle ambiguity: multiple distinct temples mentioned recently
        if len(recent_temples_ordered) > 1:
            temples_str = " and ".join(recent_temples_ordered[:2])
            return {
                "resolved_query": clean_query,
                "is_ambiguous": True,
                "referenced_temple": None,
                "clarification_message": f"Which temple are you referring to? You recently discussed {temples_str}."
            }

        # If no temples exist in recent memory, cannot resolve pronoun
        if not recent_temples_ordered:
            return {
                "resolved_query": clean_query,
                "is_ambiguous": False,
                "referenced_temple": None,
                "clarification_message": None
            }

        # 6. Deterministically resolve reference to the single recent temple
        target_temple = recent_temples_ordered[0]
        resolved = clean_query

        if ref_type == "possessive":
            # "What are its sevas?" -> "What are Dharmapuri temple's sevas?"
            resolved = re.sub(r"\b(its|it's)\b", f"{target_temple} temple's", resolved, flags=re.IGNORECASE)
        elif ref_type == "demonstrative":
            # "timings of the temple" -> "timings of Dharmapuri temple"
            # "for this 2-day trip" -> "for this 2-day trip to Dharmapuri temple"
            if re.search(r"\b(this|that|the)\s+temple\b", resolved, re.IGNORECASE):
                resolved = re.sub(r"\b(this|that|the)\s+temple\b", f"{target_temple} temple", resolved, flags=re.IGNORECASE)
            elif not re.search(rf"\b{re.escape(target_temple)}\b", resolved, re.IGNORECASE):
                resolved = re.sub(r"\b((?:this|that|the)\s+(?:[\w-]+\s+)*(?:trip|visit|tour|itinerary|plan))\b", rf"\1 to {target_temple} temple", resolved, flags=re.IGNORECASE)
        elif ref_type == "locative":
            # "how to reach there?" -> "how to reach Dharmapuri"
            if re.search(r"\b(reach|go\s+to|travel\s+to|head\s+to|visit)\s+there\b", resolved, re.IGNORECASE):
                resolved = re.sub(r"\b(reach|go\s+to|travel\s+to|head\s+to|visit)\s+there\b", rf"\1 {target_temple}", resolved, flags=re.IGNORECASE)
            elif re.search(r"\bto\s+there\b", resolved, re.IGNORECASE):
                resolved = re.sub(r"\bto\s+there\b", f"to {target_temple}", resolved, flags=re.IGNORECASE)
            else:
                # "Is accommodation available there?" -> "Is accommodation available at Dharmapuri temple?"
                resolved = re.sub(r"\bthere\b", f"at {target_temple} temple", resolved, flags=re.IGNORECASE)
        elif ref_type == "neutral_pronoun":
            # "How can I reach it?" -> "How can I reach Dharmapuri temple?"
            resolved = re.sub(r"\bit\b", f"{target_temple} temple", resolved, flags=re.IGNORECASE)
        elif ref_type == "budget_followup":
            # "what about for 2 people?" -> "what about budget for 2 people at Dharmapuri temple?"
            # "change my budget to ₹3,000" -> "change my budget to ₹3,000 for Dharmapuri temple"
            if not re.search(rf"\b{re.escape(target_temple)}\b", resolved, re.IGNORECASE):
                if not re.search(r"\bbudget\b", resolved, re.IGNORECASE):
                    resolved = f"budget {resolved} at {target_temple} temple"
                else:
                    resolved = f"{resolved} at {target_temple} temple"
        elif ref_type == "ellipsis":
            # "What about sevas?" -> "What about sevas at Dharmapuri temple?"
            resolved = f"{resolved} at {target_temple} temple"
        elif ref_type == "temple_topic_followup":
            # "What are the sevas?" -> "What are the sevas at Yadadri?"
            base_q = resolved.strip().rstrip("?. ")
            resolved = f"{base_q} at {target_temple}?"

        return {
            "resolved_query": resolved,
            "is_ambiguous": False,
            "referenced_temple": target_temple,
            "clarification_message": None
        }
