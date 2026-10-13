"""
PilgrimAI - Comprehensive Regression Test Suite Before P9
Verifies:
1. Itinerary Day Generation: 2.5-day trip has Day 1, Day 2, Day 3 (2.5 days, 2 nights, no Day 4/5, no 2.5 nights label).
2. 2-day trip has Day 1, Day 2, 1 night (no Day 3, no extra day).
3. Explicit daily structure: Morning, Afternoon, Evening (NO evening on final half-day).
4. Shared trip context: Follow-up query modifying budget preserves pax, duration, and destinations.
5. Itinerary routes consistent with navigation: realistic distances and NO self-to-self routes.
6. Budget & accommodation coordination: Unknown origin generates preliminary non-transport estimate with ₹0 transport.
7. Budget nights coordination: 0 nights = ₹0 accommodation; 1-night = 1 night cost without duplicate expense.
8. Response aggregation: Does not wipe out Day headings or daily itinerary blocks; strips raw record counts.
9. Vemulawada factual fix: Opening timing is 6.15 AM to 11.30 AM (no 6.15 PM typo).
10. Arithmetic query: Basic calculation ₹5,000 − ₹1,750 = ₹3,250 evaluated directly.
11. Recommendation limit: Requesting N items returns at most N items.
12. Accessibility validation: Distinguishes confirmed lack of access, documented limitations (e.g. Basara), and missing data.
"""

import sys
from pathlib import Path
import pytest
import json

root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agents.itinerary_planning_agent import ItineraryPlanningAgent
from app.agents.budget_agent import BudgetAgent, extract_budget_parameters
from app.agents.memory import ConversationMemoryManager
from app.agents.multi_intent_pipeline import SequentialMultiIntentPipeline
from app.agents.coordinator import MasterAgentCoordinator, evaluate_arithmetic_query
from app.agents.router import AgentRouter
from app.agents.response_aggregator import ResponseAggregator
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.state import AgentState, AgentResponse
from app.rag.completeness_validator import (
    CompletenessValidator,
    CONFIRMED_AVAILABLE,
    CONFIRMED_UNAVAILABLE,
    DOCUMENTED_LIMITATIONS,
    INFORMATION_NOT_AVAILABLE
)


# =====================================================================
# Assertion 1 & 3: 2.5-Day Trip Generation (Day 1, 2, 3, 2 nights, no evening on Day 3)
# =====================================================================

def test_01_two_and_half_day_trip_structure():
    """
    For a 2.5-day request:
    - Day 1: Full day.
    - Day 2: Full day.
    - Day 3: Final half-day with morning darshan, checkout, afternoon departure.
    - Total duration: 2.5 days, 2 nights.
    - Never Day 4 or Day 5.
    - Never labeled as '2.5 nights'.
    - Day 3 has NO evening section.
    """
    agent = ItineraryPlanningAgent()
    state = AgentState(user_query="Plan a 2.5-day trip to Vemulawada and Kondagattu")
    response = agent.execute(state)
    assert response.success is True

    answer = response.answer
    assert "#### Day 1:" in answer
    assert "#### Day 2:" in answer
    assert "#### Day 3:" in answer
    assert "#### Day 4:" not in answer
    assert "#### Day 5:" not in answer
    assert "2.5 nights" not in answer.lower()
    assert "2 Nights" in answer or "2 nights" in answer

    # Verify Day 3 has morning + checkout + afternoon departure, and NO evening window
    day3_idx = answer.find("#### Day 3:")
    day3_section = answer[day3_idx:]
    assert "Morning Window" in day3_section
    assert "Check-out" in day3_section or "check-out" in day3_section
    assert "Afternoon Window" in day3_section
    assert "Evening Window" not in day3_section


# =====================================================================
# Assertion 2: 2-Day Trip Generation (Day 1, Day 2, 1 night, no Day 3)
# =====================================================================

def test_02_two_day_trip_structure():
    """
    For a 2-day trip request:
    - Day 1: Full day.
    - Day 2: Full day with checkout / return.
    - Total duration: 2 days, 1 night.
    - No Day 3, no Day 4.
    """
    agent = ItineraryPlanningAgent()
    state = AgentState(user_query="Plan a 2 day trip to Yadadri")
    response = agent.execute(state)
    assert response.success is True

    answer = response.answer
    assert "#### Day 1:" in answer
    assert "#### Day 2:" in answer
    assert "#### Day 3:" not in answer
    assert "1 Nights" in answer or "1 night" in answer.lower()


# =====================================================================
# Assertion 4: Shared Trip Context Preservation Across Follow-ups
# =====================================================================

def test_04_shared_trip_context_preservation():
    """
    Follow-up query modifying budget preserves unchanged values (pax, duration, destinations).
    """
    memory = ConversationMemoryManager()
    mock_agents = {
        "itinerary_planning": ItineraryPlanningAgent(),
        "budget": BudgetAgent(),
    }
    pipeline = SequentialMultiIntentPipeline(agents=mock_agents, memory_manager=memory)
    session_id = "test_shared_context_p9"

    # Turn 1: User specifies destination, duration, pax, and budget
    state1 = AgentState(
        session_id=session_id,
        user_query="Plan a 2.5-day trip to Vemulawada and Dharmapuri for 4 people with budget 10000"
    )
    pipeline.execute(state1)

    # Context should be saved in memory
    ctx = memory.get_trip_context(session_id)
    assert ctx["duration_days"] == 2.5
    assert ctx["pax"] == 4
    assert ctx["budget_limit"] == 10000.0
    dest_lower = [d.lower() for d in ctx["destinations"]]
    assert any(x in dest_lower for x in ["vemulawada", "vemulavada"])
    assert "dharmapuri" in dest_lower

    # Turn 2: User only changes budget
    state2 = AgentState(
        session_id=session_id,
        user_query="Change budget to 8000"
    )
    pipeline.execute(state2)

    # Verify updated constraints preserved previous unchanged values
    c2 = state2.structured_constraints
    assert c2["budget_limit"] == 8000.0
    assert c2["duration_days"] == 2.5
    assert c2["pax"] == 4
    temples_lower = [t.lower() for t in c2["temples"]]
    assert any(x in temples_lower for x in ["vemulawada", "vemulavada"])
    assert "dharmapuri" in temples_lower


# =====================================================================
# Assertion 5: Route Distances Consistent and No Self-to-Self Routes
# =====================================================================

def test_05_no_self_to_self_routes_and_realistic_transit():
    """
    Verify multi-temple itineraries do not display self-to-self routes
    (e.g. 'Vemulawada to Vemulawada: 0 km') and use realistic distance.
    """
    agent = ItineraryPlanningAgent()
    state = AgentState(user_query="Plan a 2.5-day trip to Vemulawada and Dharmapuri")
    response = agent.execute(state)
    assert response.success is True

    answer = response.answer
    assert "Vemulawada to Vemulawada" not in answer
    assert "Dharmapuri to Dharmapuri" not in answer
    assert "0 km, 0 mins" not in answer
    # Multi-temple transit should be realistic and present
    assert "road transit" in answer or "onward journey" in answer.lower()


# =====================================================================
# Assertion 6: Unknown Origin Generates Preliminary Non-Transport Estimate
# =====================================================================

def test_06_unknown_origin_preliminary_estimate():
    """
    When origin is unknown, never finalize route-dependent costs using assumed Hyderabad.
    Provide clearly labelled preliminary non-transport estimate with ₹0 transport,
    and prompt user for starting location.
    """
    agent = BudgetAgent()
    # Query has NO starting location
    state = AgentState(user_query="Calculate budget for a 2-day trip to Yadadri for 2 people")
    response = agent.execute(state)
    assert response.success is True

    answer = response.answer
    # Must label as preliminary non-transport estimate
    assert "Preliminary Non-Transport" in answer or "Preliminary" in answer
    assert "Intercity Transit (Starting Location Unspecified)" in answer
    # Must ask user for starting location
    assert "starting location" in answer.lower() or "departure city" in answer.lower()

    # The travel line item must be ₹0
    travel_item = next((it for it in response.metadata["line_items"] if it["category"] == "travel"), None)
    assert travel_item is not None
    assert travel_item["subtotal_min"] == 0.0
    assert travel_item["subtotal_max"] == 0.0


# =====================================================================
# Assertion 7: Budget Nights & Accommodation Coordination
# =====================================================================

def test_07_budget_accommodation_nights_coordination():
    """
    0 nights = ₹0 accommodation.
    2-day/1-night = exactly 1 night accommodation cost. No duplicate hotel expense.
    """
    agent = BudgetAgent()

    # 1-day trip = 0 nights
    state_day = AgentState(user_query="Calculate budget for a 1-day trip to Yadadri for 2 people")
    res_day = agent.execute(state_day)
    assert res_day.success is True
    accom_day = next((it for it in res_day.metadata["line_items"] if it["category"] == "accommodation"), None)
    assert accom_day is not None
    assert accom_day["subtotal_min"] == 0.0

    # 2-day trip = 1 night
    state_2d = AgentState(user_query="Calculate budget for a 2-day trip to Yadadri for 2 people from Hyderabad")
    res_2d = agent.execute(state_2d)
    assert res_2d.success is True
    accom_2d = next((it for it in res_2d.metadata["line_items"] if it["category"] == "accommodation"), None)
    assert accom_2d is not None
    assert "1 night(s)" in accom_2d["quantity_desc"]
    assert accom_2d["subtotal_min"] > 0.0


# =====================================================================
# Assertion 8: Aggregator Preserves Day Headings & Strips Raw Record Counts
# =====================================================================

def test_08_aggregator_preserves_day_headings_and_cleans_artifacts():
    """
    ResponseAggregator deduplication does not eat into daily itinerary sections,
    and strips raw record counts like 'Total 3 records found'.
    """
    agg = ResponseAggregator()

    itin_text = (
        "### Pilgrimage Itinerary: Yadadri (1 Day (Day Trip))\n\n"
        "### 🏛️ Important Temple Information (Verified Records)\n\n"
        "**Yadadri**\n- **Operating Schedule**: 4:00 AM to 9:30 PM\n\n"
        "#### Day 1: Yadadri (Full Day Pilgrimage & Darshan)\n\n"
        "- **Morning Window**: Sanctum darshan 4:00 AM onwards.\n"
        "- **Afternoon Window**: Lunch and rest.\n"
        "- **Evening Window**: Harathi and return.\n\n"
        "### 🏨 Accommodation Summary (Day Trip)\n- Zero room charges."
    )
    info_text = "Yadadri temple is open from 4:00 AM to 9:30 PM daily. Total 3 records found."

    r_itin = AgentResponse(success=True, agent_name="itinerary_planning", answer=itin_text, sources=["yadadri/timings.json"])
    r_info = AgentResponse(success=True, agent_name="temple_information", answer=info_text, sources=["yadadri/overview.json"])

    aggregated = agg.aggregate([r_info, r_itin])
    assert aggregated.success is True
    # Day 1 heading must be preserved!
    assert "#### Day 1:" in aggregated.answer
    # Morning, Afternoon, Evening windows must remain!
    assert "Morning Window" in aggregated.answer
    assert "Afternoon Window" in aggregated.answer
    assert "Evening Window" in aggregated.answer
    # Raw record counts must be stripped!
    assert "Total 3 records found" not in aggregated.answer


# =====================================================================
# Assertion 9: Vemulawada Timing Typo Correction
# =====================================================================

def test_09_vemulawada_timing_typo_fix():
    """
    Verify Vemulawada timing typo '6.15 PM to 11.30 AM' is corrected to '6.15 AM to 11.30 AM'.
    """
    timings_path = root_dir / "vemulavada" / "timings.json"
    timings_str = timings_path.read_text(encoding="utf-8")
    assert "6.15 PM to 11.30 AM" not in timings_str
    assert "6.15 AM to 11.30 AM" in timings_str

    # Check sevas.json
    sevas_path = root_dir / "vemulavada" / "sevas.json"
    sevas_str = sevas_path.read_text(encoding="utf-8")
    assert "6.15 PM to 11.30 AM" not in sevas_str


# =====================================================================
# Assertion 10: General Arithmetic Evaluation
# =====================================================================

def test_10_arithmetic_calculation():
    """
    General arithmetic queries like '₹5,000 − ₹1,750' or '5000 - 1750'
    are evaluated directly without forcing temple intent.
    """
    res1 = evaluate_arithmetic_query("₹5,000 − ₹1,750")
    assert res1 is not None
    assert "3,250" in res1

    res2 = evaluate_arithmetic_query("5000 - 1750")
    assert res2 is not None
    assert "3,250" in res2

    router = AgentRouter()
    decision = router.route("5000 - 1750")
    assert decision.intent == "arithmetic"
    assert decision.selected_agent == "coordinator"


# =====================================================================
# Assertion 11: Recommendation Results Capped at Requested Limit N
# =====================================================================

def test_11_recommendation_limit_cap():
    """
    When asked for N recommendations (e.g. '3 recommendations'),
    RecommendationAgent returns at most N qualifying results.
    """
    agent = RecommendationAgent()
    state = AgentState(user_query="Give me 3 temple recommendations")
    response = agent.execute(state)
    assert response.success is True

    answer = response.answer
    assert "1." in answer
    assert "2." in answer
    assert "3." in answer
    assert "4." not in answer


# =====================================================================
# Assertion 12: Wheelchair Queries Distinguish Documented Limitations
# =====================================================================

def test_12_wheelchair_accessibility_categorization():
    """
    Distinguish confirmed lack of access, documented limitations (e.g. Basara),
    and missing data.
    """
    validator = CompletenessValidator()

    # Basara record has ramp/entrance access but must bring own equipment (DOCUMENTED_LIMITATIONS)
    basara_path = root_dir / "basara" / "elderly_ppl.json"
    with open(basara_path, "r", encoding="utf-8") as f:
        basara_data = json.load(f)
    status_b, ev_b, _ = validator.evaluate_availability(basara_data, "wheelchair")
    assert status_b == DOCUMENTED_LIMITATIONS
    assert "bring their own" in ev_b.lower() or "mobility equipment" in ev_b.lower()

    # Confirmed unavailable
    unavail_data = {
        "wheelchair_accessibility": {
            "is_wheelchair_friendly": False,
            "details": ["No ramps or lifts available; steep stairs only."]
        }
    }
    status_u, _, _ = validator.evaluate_availability(unavail_data, "wheelchair")
    assert status_u == CONFIRMED_UNAVAILABLE

    # Missing data
    missing_data = {}
    status_m, _, _ = validator.evaluate_availability(missing_data, "wheelchair")
    assert status_m == INFORMATION_NOT_AVAILABLE
