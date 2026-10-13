import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import pytest
import datetime
from app.db.chat_history import (
    sanitize_for_mongo,
    save_user_message,
    save_assistant_message,
    get_session_messages,
    get_existing_assistant_reply
)
from app.rag.query_planner import query_planner
from app.agents.state import AgentRouteDecision


def test_sanitize_for_mongo_complex_types():
    # 1. Pydantic QueryPlan
    plan = query_planner.plan("timings of Yadadri")
    
    # 2. Pydantic RouteDecision
    rd = AgentRouteDecision(
        selected_agent="temple_information",
        intent="factual",
        confidence=1.0,
        reason="Test",
        metadata={"sub_intents": ["timings"]}
    )

    doc = {
        "plan": plan,
        "route": rd,
        "tags": {"a", "b", "c"},
        "nested": {
            "dot.key": "valid",
            "$dollar_key": "valid_too"
        },
        "tuple_val": (1, 2, 3),
        "nan_val": float("nan")
    }

    sanitized = sanitize_for_mongo(doc)
    
    # Must be encodable by BSON without errors
    from bson import BSON
    raw = BSON.encode(sanitized)
    assert len(raw) > 0
    assert "plan" in sanitized
    assert isinstance(sanitized["tags"], list)
    assert "dot_key" in sanitized["nested"]
    assert "_dollar_key" in sanitized["nested"]


def test_chat_history_idempotency_and_persistence():
    import uuid
    user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    session_id = f"session_lifecycle_{uuid.uuid4().hex}"
    client_msg_id = f"msg_test_{uuid.uuid4().hex[:8]}"

    # 1. First user save
    u_id_1 = save_user_message(
        user_id=user_id,
        session_id=session_id,
        content="plan a 3 day trip to edupayala",
        client_message_id=client_msg_id
    )
    assert u_id_1 is not None

    # 2. Duplicate user save with same client_message_id (e.g. double click or retry)
    u_id_2 = save_user_message(
        user_id=user_id,
        session_id=session_id,
        content="plan a 3 day trip to edupayala",
        client_message_id=client_msg_id
    )
    assert u_id_1 == u_id_2, "Should return existing message ID and avoid duplicate insertion"

    # 3. Save assistant response with complex metadata
    plan = query_planner.plan("plan a 3 day trip to edupayala")
    a_id_1 = save_assistant_message(
        user_id=user_id,
        session_id=session_id,
        content="Here is your 3-day itinerary...",
        agent_name="response_aggregator",
        metadata={"plan": plan, "options": {"hotel_a", "hotel_b"}},
        sources=["edupayala.json"],
        confidence_score=0.95,
        client_message_id=client_msg_id
    )
    assert a_id_1 is not None

    # 4. Duplicate assistant save with same client_message_id
    a_id_2 = save_assistant_message(
        user_id=user_id,
        session_id=session_id,
        content="Here is your 3-day itinerary...",
        agent_name="response_aggregator",
        metadata={"plan": plan},
        client_message_id=client_msg_id
    )
    assert a_id_1 == a_id_2, "Should return existing assistant message ID"

    # 5. Check idempotency lookup
    cached = get_existing_assistant_reply(user_id, session_id, client_msg_id)
    assert cached is not None
    assert cached["content"] == "Here is your 3-day itinerary..."

    # 6. Check get_session_messages
    msgs = get_session_messages(user_id, session_id)
    assert msgs is not None
    assert len(msgs) == 2, "Must contain exactly 1 user message and 1 assistant message"
    assert msgs[0]["role"] == "user"
    assert msgs[0]["message_id"] == u_id_1
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["message_id"] == a_id_1

    # 7. Legitimate repeated question with a NEW client_message_id
    new_client_id = "msg_test_abc_2"
    u_id_3 = save_user_message(
        user_id=user_id,
        session_id=session_id,
        content="plan a 3 day trip to edupayala",
        client_message_id=new_client_id
    )
    assert u_id_3 != u_id_1, "Distinct client_message_id should legitimately create a new turn"
    a_id_3 = save_assistant_message(
        user_id=user_id,
        session_id=session_id,
        content="Here is your updated itinerary...",
        agent_name="response_aggregator",
        client_message_id=new_client_id
    )
    msgs_updated = get_session_messages(user_id, session_id)
    assert len(msgs_updated) == 4, "Should have 2 legitimate question/answer pairs"


def test_comparison_agent_conflict_detection():
    from app.agents.comparison_agent import ComparisonAgent
    agent = ComparisonAgent()

    # Normal festival variations/disclaimers should NOT trigger conflict banner
    normal_data = {
        "Yadadri": {
            "timings": "Morning: 6:00 AM - 9:00 PM. Note: timings are subject to change during festivals."
        },
        "Basara": {
            "timings": "Morning: 4:00 AM - 8:30 PM. Seasonal variations may apply."
        }
    }
    assert agent._detect_conflicts(normal_data) is False

    # Actual contradictory records SHOULD trigger conflict banner
    conflict_data = {
        "Yadadri": {
            "timings": "Timings: Conflicting records exist between temple portal and district handbook."
        },
        "Basara": {
            "timings": "Morning: 4:00 AM - 8:30 PM."
        }
    }
    assert agent._detect_conflicts(conflict_data) is True


def test_concurrent_requests_atomic_claim():
    """
    Verifies that under true concurrent requests with identical (user_id, session_id, client_message_id):
    1. Exactly one request atomically claims ownership to execute the coordinator.
    2. Concurrent sibling requests do not create duplicate user messages or independently execute the coordinator.
    3. Once the assistant response is saved, concurrent requests safely retrieve the single response.
    """
    import concurrent.futures
    import time
    from app.db.chat_history import (
        init_chat_indexes,
        claim_user_message_turn,
        wait_for_assistant_reply,
        get_session_messages
    )

    init_chat_indexes()

    user_id = "test_user_concurrent"
    session_id = f"session_concurrent_{int(time.time()*1000)}"
    client_msg_id = f"client_concurrent_{int(time.time()*1000)}"
    question = "How far is Palvancha to Balkampet?"

    results = []

    def attempt_claim(worker_idx):
        is_owner, msg_id = claim_user_message_turn(
            user_id=user_id,
            session_id=session_id,
            content=question,
            client_message_id=client_msg_id
        )
        return worker_idx, is_owner, msg_id

    # Launch 5 concurrent threads simulating race condition
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(attempt_claim, i) for i in range(5)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # Exactly one thread must have claimed ownership
    owners = [r for r in results if r[1] is True]
    non_owners = [r for r in results if r[1] is False]

    assert len(owners) == 1, f"Expected exactly 1 owner, got {len(owners)}"
    assert len(non_owners) == 4, f"Expected 4 non-owners, got {len(non_owners)}"

    # All non-owners must receive the same message_id as the owner
    owner_msg_id = owners[0][2]
    for non_owner in non_owners:
        assert non_owner[2] == owner_msg_id, "Non-owners must resolve to the owner's message ID"

    # Simulate owner completing execution and persisting single assistant reply
    assistant_reply_id = save_assistant_message(
        user_id=user_id,
        session_id=session_id,
        content="The distance is 300.7 km.",
        agent_name="live_navigation",
        client_message_id=client_msg_id
    )
    assert assistant_reply_id is not None

    # Sibling requests waiting for assistant reply retrieve it immediately
    retrieved = wait_for_assistant_reply(
        user_id=user_id,
        session_id=session_id,
        client_message_id=client_msg_id,
        timeout_seconds=2.0
    )
    assert retrieved is not None
    assert retrieved["content"] == "The distance is 300.7 km."

    # Total session messages must be exactly 2 (1 user, 1 assistant)
    msgs = get_session_messages(user_id, session_id)
    assert len(msgs) == 2, f"Expected exactly 2 messages in session, got {len(msgs)}"


def test_bson_metadata_sanitization_agent_integrity():
    """
    Confirms that BSON metadata sanitization transforms unsupported values (NaN, Inf, sets, custom types)
    without silently dropping or altering meaningful metadata used by response aggregator,
    itinerary, budget, or navigation agents.
    """
    from bson import BSON

    # 1. Navigation Agent metadata
    nav_meta = {
        "origin": "Palvancha",
        "destination": "Balkampet Yellamma Temple, Hyderabad",
        "origin_coordinates": (17.5988, 80.7064),
        "destination_coordinates": (17.447528, 78.451639),
        "distance_km": 300.7,
        "duration_minutes": 348.3,
        "traffic_duration_minutes": None,
        "route_summary": "Hyderabad - Vijayawada Hwy",
        "travel_mode": "driving",
        "polyline": "encoded_polyline_string_abc123",
        "provider": "Google Maps Routes API",
        "route_status": "SUCCESS"
    }

    # 2. Budget Agent metadata
    budget_meta = {
        "total_cost": 4200.0,
        "budget_status": "WITHIN_BUDGET",
        "cost_per_person": 2100.0,
        "num_people": 2,
        "days": 3,
        "mode_of_transport": "car",
        "breakdown": {
            "accommodation": 1800.0,
            "transport": 1400.0,
            "food": 800.0,
            "darshan": 200.0
        },
        "selected_accommodation": {
            "name": "Haritha Hotel",
            "price_per_night": 900.0,
            "contact": "08744-254123"
        }
    }

    # 3. Itinerary Agent metadata
    itinerary_meta = {
        "days": 3,
        "destinations": ["Edupayala", "Chilkur Balaji"],
        "pace": "moderate",
        "daily_plan": [
            {"day": 1, "morning": "Edupayala Darshan", "evening": "Medak Fort"},
            {"day": 2, "morning": "Travel to Hyderabad", "evening": "Rest"},
            {"day": 3, "morning": "Chilkur Balaji", "evening": "Departure"}
        ],
        "summary": "3-day spiritual circuit across Medak and Hyderabad"
    }

    # 4. Aggregator metadata
    aggregator_meta = {
        "agent_name": "response_aggregator",
        "confidence_score": 0.98,
        "sources": ["edupayala.json", "chilkur.json"],
        "retrieved_chunks": 4,
        "intermediate_results": {
            "navigation": nav_meta,
            "budget": budget_meta,
            "itinerary": itinerary_meta
        }
    }

    sanitized = sanitize_for_mongo(aggregator_meta)

    # Validate that BSON encoding succeeds
    encoded = BSON.encode(sanitized)
    assert len(encoded) > 0, "BSON encoding must succeed"

    # Validate Navigation fields
    nav_san = sanitized["intermediate_results"]["navigation"]
    assert nav_san["origin"] == "Palvancha"
    assert nav_san["distance_km"] == 300.7
    assert nav_san["origin_coordinates"] == [17.5988, 80.7064], "Coordinate tuples must serialize cleanly as coordinate lists"
    assert nav_san["destination_coordinates"] == [17.447528, 78.451639]
    assert nav_san["polyline"] == "encoded_polyline_string_abc123"

    # Validate Budget fields
    bud_san = sanitized["intermediate_results"]["budget"]
    assert bud_san["total_cost"] == 4200.0
    assert bud_san["breakdown"]["accommodation"] == 1800.0
    assert bud_san["selected_accommodation"]["contact"] == "08744-254123"

    # Validate Itinerary fields
    itin_san = sanitized["intermediate_results"]["itinerary"]
    assert itin_san["days"] == 3
    assert len(itin_san["daily_plan"]) == 3
    assert itin_san["daily_plan"][0]["morning"] == "Edupayala Darshan"

    # Validate Aggregator top-level fields
    assert sanitized["confidence_score"] == 0.98
    assert sanitized["retrieved_chunks"] == 4

