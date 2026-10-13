import re
import sys
from pathlib import Path
import pytest

root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agents.state import AgentState
from app.agents.budget_agent import BudgetAgent, extract_budget_parameters
from app.agents.itinerary_planning_agent import ItineraryPlanningAgent
from app.agents.coordinator import MasterAgentCoordinator
from app.agents.router import AgentRouter
from app.agents.multi_intent_pipeline import SequentialMultiIntentPipeline
from app.agents.response_aggregator import ResponseAggregator
from app.agents.tools.rag_tool import RAGTool


@pytest.fixture(scope="module")
def shared_rag_tool():
    return RAGTool()


@pytest.fixture
def budget_agent(shared_rag_tool):
    return BudgetAgent(rag_tool=shared_rag_tool)


@pytest.fixture
def itinerary_agent(shared_rag_tool):
    return ItineraryPlanningAgent(rag_tool=shared_rag_tool)


@pytest.fixture
def multi_intent_pipeline(budget_agent, itinerary_agent):
    router = AgentRouter()
    router.register_agent("itinerary_planning")
    router.register_agent("budget")
    agents = {
        "itinerary_planning": itinerary_agent,
        "budget": budget_agent
    }
    return SequentialMultiIntentPipeline(
        router=router,
        agents=agents,
        aggregator=ResponseAggregator()
    )


# =============================================================================
# P0 & P3: Parameter Extraction & Duration Synchronization
# =============================================================================

def test_extract_budget_parameters_multi_day():
    """Verify 'plan a 3 day trip to Birla Mandir with a budget of 5000' extracts 3 days and 2 nights."""
    query = "plan a 3 day trip to Birla Mandir with a budget of 5000"
    params = extract_budget_parameters(query)

    assert params["days"] == 3
    assert params["nights"] == 2
    assert params["is_day_trip"] is False
    assert params["days_assumed"] is False
    assert params["budget_limit"] == 5000.0
    assert params["accom_requested"] is True
    assert params["pax"] == 1


def test_extract_budget_parameters_day_trip():
    """Verify explicit day trip queries extract 1 day and 0 nights."""
    query1 = "day trip to Birla Mandir with a budget of 1500"
    params1 = extract_budget_parameters(query1)
    assert params1["days"] == 1
    assert params1["nights"] == 0
    assert params1["is_day_trip"] is True

    query2 = "1 day visit to Yadadri"
    params2 = extract_budget_parameters(query2)
    assert params2["days"] == 1
    assert params2["nights"] == 0
    assert params2["is_day_trip"] is True


# =============================================================================
# P2: Local Intra-City Transport Calculations (Hyderabad to Birla Mandir)
# =============================================================================

def test_p2_hyderabad_to_birla_mandir_transit_nonzero(budget_agent):
    """Verify unstated origin excludes intercity transport without fabricating Hyderabad,
    and verify explicit Hyderabad origin calculates intra-city transit scenarios."""
    # Case 1: Unstated origin query must preserve unknown origin and exclude unquantified travel
    state_unstated = AgentState(user_query="plan a 3 day trip to Birla Mandir with a budget of 5000")
    resp_unstated = budget_agent.execute(state_unstated)

    assert resp_unstated.success is True
    params_unstated = resp_unstated.metadata.get("parameters", {})
    assert params_unstated["days"] == 3
    assert params_unstated["nights"] == 2
    assert params_unstated.get("origin_assumed") is True

    # Travel must be categorized as unstated / unquantified (selected_cost == 0.0)
    items_unstated = resp_unstated.metadata.get("selected_plan", {}).get("line_items", [])
    travel_unstated = [it for it in items_unstated if it["category"] == "travel"]
    assert len(travel_unstated) > 0
    assert travel_unstated[0]["selected_cost"] == 0.0
    assert travel_unstated[0]["provenance"] == "UNAVAILABLE"
    assert "Intercity Transit (Starting Location Unspecified)" in travel_unstated[0]["item_name"]
    assert "please provide your starting location" in resp_unstated.answer.lower() or "unstated" in resp_unstated.answer.lower() or "pending starting location" in resp_unstated.answer.lower()

    # Case 2: When origin IS explicitly provided as Hyderabad, calculate intra-city transit scenarios
    state_explicit = AgentState(user_query="plan a 3 day trip from Hyderabad to Birla Mandir with a budget of 5000")
    resp_explicit = budget_agent.execute(state_explicit)
    assert resp_explicit.success is True

    scenarios = resp_explicit.metadata.get("transport_scenarios", {})
    assert scenarios.get("is_intra_city") is True
    assert "public_bus" in scenarios
    assert "local_auto" in scenarios
    assert "private_cab" in scenarios

    # Fares must be non-zero
    p_bus = scenarios["public_bus"]
    assert p_bus["total_min"] >= 50.0  # At least ₹50 round-trip for public bus/metro
    assert p_bus["fare_one_way_per_pax"][0] >= 20.0

    p_auto = scenarios["local_auto"]
    assert p_auto["total_min"] >= 200.0  # Auto round-trip

    p_cab = scenarios["private_cab"]
    assert p_cab["fare_min"] >= 500.0  # Cab round-trip


# =============================================================================
# P3: Budget Synchronization & Accommodation Inclusion
# =============================================================================

def test_p3_budget_includes_accommodation_and_synchronizes(budget_agent):
    """Verify budget includes accommodation for 2 nights, excludes unquantified intercity travel, and synchronizes total."""
    state = AgentState(user_query="plan a 3 day trip to Birla Mandir with a budget of 5000")
    resp = budget_agent.execute(state)

    assert resp.success is True
    selected_plan = resp.metadata.get("selected_plan", {})
    items = selected_plan.get("line_items", [])

    # Accommodation MUST be present in itemized line items
    accom_items = [it for it in items if it["category"] == "accommodation"]
    assert len(accom_items) > 0, "Accommodation must not be excluded for a 3-day trip!"

    # Birla Mandir has verified The Superb Hotels (₹2,002/night)
    chosen_accom = accom_items[0]
    # For 2 nights: 2 * 2,002 = 4,004
    assert chosen_accom["selected_cost"] == 4004.0, f"Expected 4004 for 2 nights, got {chosen_accom['selected_cost']}"

    # Meals must be scaled for 3 days (1 pax * 3 days * 200 = 600)
    meal_items = [it for it in items if it["category"] == "meals"]
    assert len(meal_items) > 0
    assert meal_items[0]["selected_cost"] == 600.0

    # Local transit across 3 days (1 pax * 3 days * 50 = 150)
    local_items = [it for it in items if it["category"] == "local_transit"]
    assert len(local_items) > 0
    assert local_items[0]["selected_cost"] == 150.0

    # Intercity travel: Origin is unstated -> Exclude unquantified intercity transit without fabricating ₹50
    travel_items = [it for it in items if it["category"] == "travel"]
    assert len(travel_items) > 0
    assert travel_items[0]["selected_cost"] == 0.0
    assert travel_items[0]["provenance"] == "UNAVAILABLE"

    # Selected plan total: 0 (travel) + 0 (entry) + 4004 (lodging) + 600 (meals) + 150 (local transit) = 4754 <= 5000
    assert selected_plan["total"] == 4754.0
    assert selected_plan["within_budget"] is True
    assert resp.metadata["budget_status"] == "ESTIMATED_WITHIN_BUDGET"

    # Broader range must cross ₹5,000 due to Amrutha Castle (2 * 2518 = 5036 alone exceeds 5000)
    broader = resp.metadata.get("broader_range", {})
    assert broader["max"] > 5000.0
    assert broader["exceeds_budget"] is True


def test_p3_hard_budget_gate_over_budget_when_limit_too_low(budget_agent):
    """Verify hard budget gate returns OVER_BUDGET when budget limit is too low to cover unavoidable baseline expenses."""
    state = AgentState(user_query="plan a 3 day trip to Birla Mandir with a budget of 500")
    resp = budget_agent.execute(state)

    assert resp.success is True
    # Unavoidable baseline costs (meals ₹600 + local transit ₹150 = ₹750) exceed ₹500
    assert resp.metadata["budget_status"] == "OVER_BUDGET"
    assert resp.metadata["is_over_budget"] is True
    assert resp.metadata.get("shortfall", 0.0) > 0.0
    assert "OVER_BUDGET" in resp.answer or "exceeds" in resp.answer.lower() or "shortfall" in resp.answer.lower()


# =============================================================================
# P1: Realistic, Distinct Days and Non-Repetitive Attractions
# =============================================================================

def test_p1_birla_mandir_3_day_itinerary_distinct_days(itinerary_agent):
    """Verify 3-day Birla Mandir itinerary has distinct themes and zero duplicate attractions across days."""
    state = AgentState(user_query="plan a 3 day trip to Birla Mandir")
    resp = itinerary_agent.execute(state)

    assert resp.success is True
    answer = resp.answer

    # Must contain 3 distinct days
    assert "#### Day 1:" in answer
    assert "#### Day 2:" in answer
    assert "#### Day 3:" in answer

    # Verify Day 1 has arrival and check-in
    day1_idx = answer.find("#### Day 1:")
    day2_idx = answer.find("#### Day 2:")
    day3_idx = answer.find("#### Day 3:")

    day1_text = answer[day1_idx:day2_idx]
    day2_text = answer[day2_idx:day3_idx]
    day3_text = answer[day3_idx:]

    assert "Arrival" in day1_text
    assert "Check-in" in day1_text or "check-in" in day1_text.lower()
    assert "Sandhya Aarti" in day1_text or "Aarti" in day1_text

    # Respects afternoon sanctum break (12:00 PM to 3:00 PM)
    assert "12:00 PM to 3:00 PM" in day1_text or "break" in day1_text.lower()

    # Day 3 has check-out and departure
    assert "Check-out" in day3_text or "check-out" in day3_text.lower()
    assert "Departure" in day3_text or "departure" in day3_text.lower()

    # Verify no duplicate attractions between Day 2 and Day 3
    # Extract nearby attractions listed in Day 2
    d2_attraction_match = re.search(r"Verified Nearby Attractions\*\*:\s*([^\n]+)", day2_text)
    d3_attraction_match = re.search(r"Verified Nearby Attractions\*\*:\s*([^\n]+)", day3_text)

    if d2_attraction_match and d3_attraction_match:
        d2_attrs = [a.strip().lower() for a in d2_attraction_match.group(1).split(",") if a.strip()]
        d3_attrs = [a.strip().lower() for a in d3_attraction_match.group(1).split(",") if a.strip()]

        for d2_a in d2_attrs:
            # None of Day 2 attractions should appear in Day 3 list
            clean_name = d2_a.split("(")[0].strip()
            if len(clean_name) > 3:
                assert clean_name not in [x.split("(")[0].strip() for x in d3_attrs], (
                    f"Duplicate attraction found across days: '{clean_name}' appears in both Day 2 and Day 3!"
                )

    # Must include explanation of temple duration (core darshan takes 1.5–2.5 hours)
    assert "1.5" in answer and "hours" in answer.lower()


# =============================================================================
# P0 & Multi-Intent Pipeline: Shared Trip Plan Synchronization
# =============================================================================

def test_p0_multi_intent_pipeline_shared_trip_plan(multi_intent_pipeline):
    """Verify that multi_intent_pipeline executes itinerary and budget with synchronized 3-day duration."""
    state = AgentState(user_query="plan a 3 day trip to Birla Mandir with a budget of 5000")
    resp = multi_intent_pipeline.execute(state)

    assert resp.success is True
    meta = resp.metadata

    # Verification of shared metadata synchronization
    assert meta.get("duration_days") == 3
    assert meta.get("days") == 3
    assert meta.get("nights") == 2

    params = meta.get("parameters", {})
    assert params.get("days") == 3, f"Expected parameters.days == 3, got {params.get('days')}"
    assert params.get("nights") == 2, f"Expected parameters.nights == 2, got {params.get('nights')}"
    assert params.get("is_day_trip") is False

    # Check shared trip_plan object
    trip_plan = meta.get("trip_plan", {})
    assert trip_plan.get("duration_days") == 3
    assert trip_plan.get("nights") == 2
    assert trip_plan.get("is_day_trip") is False
    assert trip_plan.get("budget_status") == "ESTIMATED_WITHIN_BUDGET"

    # Answer text must contain both Itinerary (Day 1, 2, 3) and Budget sections
    assert "#### Day 1:" in resp.answer
    assert "#### Day 2:" in resp.answer
    assert "#### Day 3:" in resp.answer
    assert "Budget" in resp.answer or "Feasible" in resp.answer


# =============================================================================
# Universal Temple Multi-Day Scaling (Not hardcoded to Birla Mandir)
# =============================================================================

def test_universal_yadadri_2_day_trip(budget_agent, itinerary_agent):
    """Verify Yadadri 2-day trip scales days=2, nights=1 across both agents."""
    q = "plan a 2 day trip to Yadadri with a budget of 6000"

    # Budget Agent
    b_state = AgentState(user_query=q)
    b_resp = budget_agent.execute(b_state)
    assert b_resp.success is True
    assert b_resp.metadata["parameters"]["days"] == 2
    assert b_resp.metadata["parameters"]["nights"] == 1
    assert b_resp.metadata["parameters"]["is_day_trip"] is False

    # Itinerary Agent
    i_state = AgentState(user_query=q)
    i_resp = itinerary_agent.execute(i_state)
    assert i_resp.success is True
    assert i_resp.metadata["duration_days"] == 2
    assert i_resp.metadata["nights"] == 1
    assert "#### Day 1:" in i_resp.answer
    assert "#### Day 2:" in i_resp.answer
