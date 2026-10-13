"""
PilgrimAI - Generalized Destination Resolution & Itinerary Budget Quality Tests

Tests covering:
- Generalized place resolution: cities, towns, railway stations, hotels, and canonical temples.
- Non-temple destinations never silently become temples.
- Duration extraction supporting half-day expressions (e.g. '3 and half days').
- Itinerary meal diversity (lunch and dinner do not recommend identical restaurants).
- Afternoon break formatting (no empty ' to ' periods).
- Honest over-budget calculation and shortfall reporting for multi-day trips with low budgets.
- Response aggregator cleaning of raw record counts and internal JSON file paths.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pytest
import re
from app.agents.live_navigation_agent import LiveNavigationAgent
from app.agents.itinerary_planning_agent import ItineraryPlanningAgent
from app.agents.budget_agent import BudgetAgent, extract_budget_parameters
from app.agents.response_aggregator import ResponseAggregator
from app.agents.state import AgentResponse, AgentState


@pytest.fixture
def nav_agent():
    return LiveNavigationAgent()


@pytest.fixture
def itinerary_agent():
    return ItineraryPlanningAgent()


@pytest.fixture
def budget_agent():
    return BudgetAgent()


@pytest.fixture
def aggregator():
    return ResponseAggregator()


# =====================================================================
# P0: Generalized Destination Resolution
# =====================================================================

def test_city_resolves_as_city_not_temple(nav_agent):
    """General city 'Hyderabad' must resolve cleanly as an address/city, not a temple."""
    disp_name, coords, canon = nav_agent._resolve_location("Hyderabad")
    assert disp_name == "Hyderabad"
    assert coords is None  # Handled as regional city / address
    assert canon is None  # Must NOT be a temple


def test_station_resolves_without_temple(nav_agent):
    """Railway station must not be treated as a temple."""
    disp_name, coords, canon = nav_agent._resolve_location("Secunderabad Railway Station")
    assert "Railway Station" in disp_name
    assert canon is None


def test_hotel_resolves_without_temple(nav_agent):
    """Hotel must not be treated as a temple."""
    disp_name, coords, canon = nav_agent._resolve_location("Novotel Hyderabad")
    assert "Novotel" in disp_name
    assert canon is None


def test_temple_resolves_as_temple(nav_agent):
    """Explicit temple name resolves to the verified canonical temple."""
    disp_name, coords, canon = nav_agent._resolve_location("Birla Mandir")
    assert coords is not None
    assert canon == "Birla Mandir"


def test_temple_alias_resolves_correctly(nav_agent):
    """Canonical temple alias 'Yadagirigutta' resolves to 'Yadadri'."""
    disp_name, coords, canon = nav_agent._resolve_location("Yadagirigutta")
    assert coords is not None
    assert canon == "Yadadri"


# =====================================================================
# P1: Budget Consistency & Duration Extraction
# =====================================================================

def test_half_day_duration_extraction_itinerary(itinerary_agent):
    """'3 and half day trip' must be recognized as 3.5 days without premature integer rounding."""
    days = itinerary_agent._extract_duration_days("plan a 3 and half day trip to srisailam", 1)
    assert days == 3.5


def test_half_day_duration_extraction_budget():
    """'3 and half day trip' must be parsed with days=3.5, nights=3."""
    params = extract_budget_parameters("plan a 3 and half day trip with a budget of 3200 for 2 people")
    assert params["days"] == 3.5
    assert params["nights"] == 3
    assert params["is_day_trip"] is False
    assert params["pax"] == 2
    assert params["budget_limit"] == 3200.0


def test_budget_honest_over_budget_calculation(budget_agent):
    """A multi-day trip with 2 people and ₹3,200 budget must honestly report OVER_BUDGET."""
    query = "plan a 3 and half day trip to srisailam and basara with a budget of 3200 for 2 people"
    state = AgentState(user_query=query)
    resp = budget_agent.execute(state)
    assert resp.success is True
    meta = resp.metadata
    assert meta.get("budget_status") == "OVER_BUDGET"
    assert meta.get("is_over_budget") is True
    assert meta.get("shortfall") is not None
    assert meta["shortfall"] > 0
    # Over-budget details must be clearly communicated in answer
    assert "OVER_BUDGET" in resp.answer or "exceeds" in resp.answer.lower() or "shortfall" in resp.answer.lower()


def test_single_day_budget_no_hotel(budget_agent):
    """Single-day trip requires 0 hotel nights."""
    params = extract_budget_parameters("1 day trip to yadadri with budget 2000 for 2 people")
    assert params["days"] == 1
    assert params["nights"] == 0
    assert params["rooms"] == 0
    assert params["is_day_trip"] is True


# =====================================================================
# P2: Itinerary Quality & Grounding
# =====================================================================

def test_itinerary_distinct_meals_per_day(itinerary_agent):
    """Lunch and dinner recommendations on Day 1 should not recommend the exact same restaurant."""
    mock_data = {
        "temple_name": "Test Temple",
        "timings": {"general_timings": "6:00 AM - 8:00 PM"},
        "restaurants": {
            "all": [
                {"name": "Govinda Veg Mess", "type": "Vegetarian Mess", "pricing": "₹80-₹120", "timings": "7 AM - 10 PM", "distance": "200m", "speciality": "Thali", "address": "Temple Rd", "maps_url": "https://maps.google.com/?q=Govinda"},
                {"name": "Sri Rama Tiffin & Meals", "type": "Casual Dining", "pricing": "₹100-₹180", "timings": "7 AM - 10 PM", "distance": "350m", "speciality": "South Indian Meals", "address": "Main Rd", "maps_url": "https://maps.google.com/?q=SriRama"}
            ]
        }
    }
    lines = itinerary_agent._format_schedule_section("Test Temple", 1, 1, 1, 1, mock_data)
    rendered = "\n".join(lines)
    # Both restaurants should appear, but lunch and dinner must be distinct
    assert "Govinda Veg Mess" in rendered
    assert "Sri Rama Tiffin & Meals" in rendered


def test_afternoon_break_no_empty_break_period(itinerary_agent):
    """When afternoon_break has empty dict values, it should not produce 'to' without times."""
    mock_data = {
        "timings": {
            "general_timings": "6:00 AM - 8:00 PM",
            "afternoon_break": {"from": "", "to": ""}
        }
    }
    parsed = itinerary_agent._parse_timings(mock_data)
    assert parsed["has_break"] is False
    assert parsed["break_period"] != " to "


# =====================================================================
# P3: Response Aggregation Cleaning
# =====================================================================

def test_response_aggregator_cleans_raw_records_and_paths(aggregator):
    """Aggregator must strip raw database record markers and json filenames from user answer."""
    raw_answer = (
        "Here is the comparison:\n"
        "- Srisailam Temple: Famous for Mallikarjuna (and 191 more records)\n"
        "Temple Id: srisailam; (Source: srisailam/temple_info.json)\n"
        "**Sources:**\n- **Record**: srisailam/temple_info.json"
    )
    resp = AgentResponse(
        success=True,
        agent_name="comparison",
        answer=raw_answer,
        sources=["srisailam/temple_info.json"],
        metadata={}
    )
    aggregated = aggregator.aggregate([resp])
    assert "(and 191 more records)" not in aggregated.answer
    assert "srisailam/temple_info.json" not in aggregated.answer
    assert "Temple Id:" not in aggregated.answer


# =====================================================================
# P4: Quality & Grounding Regression Tests
# =====================================================================

def test_fractional_duration_itinerary_generation(itinerary_agent):
    """2.5-day trip generates exactly 2 full days + 1 half-day schedule with 2 nights."""
    state = AgentState(user_query="Plan a 2.5-day trip to Jogulamba Temple")
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    meta = resp.metadata
    assert meta.get("duration_days") == 2.5
    assert meta.get("calendar_days") == 3
    assert meta.get("nights") == 2
    assert meta.get("is_fractional") is True
    # Day 1, Day 2, and Day 3 (Half-Day) must be present in output
    assert "Day 1:" in resp.answer
    assert "Day 2:" in resp.answer
    assert "Day 3:" in resp.answer and "Half-Day" in resp.answer
    # No Day 4
    assert "Day 4:" not in resp.answer
    # Check-out and return transit scheduled on final half-day
    assert "Check-out" in resp.answer
    assert "Return Journey" in resp.answer
    # Display name must be canonical, no internal labels
    assert "Jogulamba(new)" not in resp.answer
    assert "Jogulamba Temple, Alampur" in resp.answer


def test_missing_origin_provisional_budget_notice(budget_agent):
    """When no origin is provided, budget response clearly discloses provisional baseline."""
    state = AgentState(user_query="Plan a 2.5-day trip to Jogulamba with a budget of 2500")
    resp = budget_agent.execute(state)
    assert resp.success is True
    # Must flag origin as provisional / assumed
    assert resp.metadata.get("origin_assumed") is True
    assert "Provisional Origin Baseline" in resp.answer or "provisional" in resp.answer.lower()
    # Must preserve canonical name
    assert "Jogulamba(new)" not in resp.answer
    assert "Jogulamba Temple, Alampur" in resp.answer


def test_costs_not_available_honest_disclosure(budget_agent):
    """Costs not available section must not falsely claim 'None (all costs verified)'."""
    state = AgentState(user_query="budget for 2 days at jogulamba with 3000")
    resp = budget_agent.execute(state)
    assert resp.success is True
    assert "None (all requested categories have verified records)" not in resp.answer
    assert "require confirmation" in resp.answer.lower() or "requires direct" in resp.answer.lower()


def test_departure_terminal_grounded(itinerary_agent):
    """Return journey should derive local terminal from travel data, not hardcoded Ramagundam."""
    state = AgentState(user_query="Plan a 2.5-day trip to Jogulamba Temple")
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    # Jogulamba is in Alampur / Mahbubnagar region; Ramagundam is 350km away in North Telangana!
    assert "Ramagundam" not in resp.answer
    # Should reference local railway/bus terminals (Alampur / Kurnool)
    assert ("Alampur" in resp.answer or "Kurnool" in resp.answer or "terminal" in resp.answer.lower())


def test_response_aggregator_deduplicates_accommodation(aggregator):
    """When both itinerary and budget agents are present, itinerary accommodation stub is stripped."""
    itin_resp = AgentResponse(
        success=True,
        agent_name="itinerary_planning",
        answer=(
            "### 📅 Comprehensive Itinerary\n"
            "### Day 1: Arrival\n- 08:00 AM: Darshan\n\n"
            "### 🏨 Accommodation Summary\n"
            "- Booking: Devasthanam choultries.\n"
            "- Check-in: Standard afternoon."
        ),
        sources=[],
        metadata={"duration_days": 2.5, "nights": 2}
    )
    budget_resp = AgentResponse(
        success=True,
        agent_name="budget",
        answer=(
            "## 💰 Comprehensive Pilgrim Budget Estimate\n"
            "### 🏨 Accommodation Comparison & Verification\n"
            "- Sri Jogulamba Choultry: ₹300 - ₹500/night"
        ),
        sources=[],
        metadata={"budget_limit": 2500.0, "nights": 2}
    )
    aggregated = aggregator.aggregate([itin_resp, budget_resp])
    # The itinerary accommodation stub must be stripped
    assert "### 🏨 Accommodation Summary" not in aggregated.answer
    # The authoritative budget accommodation section must remain
    assert "### 🏨 Accommodation Comparison & Verification" in aggregated.answer
    # Nights and duration must harmonize
    assert aggregated.metadata["trip_plan"]["duration_days"] == 2.5
    assert aggregated.metadata["trip_plan"]["nights"] == 2

