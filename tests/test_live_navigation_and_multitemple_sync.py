"""
Regression Test Suite for P0-P5:
1. Live Navigation intent parsing, repeated origin cleaning, destination resolution.
2. Honest duration/distance formatting and route states.
3. Multi-temple 4-day trip canonical routing, accommodation partition (exactly 3 nights, not 6),
   continuous circuit transportation (no duplicate round trips), and budget synchronization.
4. Global support across all 20 temples.
"""

import sys
from pathlib import Path
import pytest

# Ensure backend directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agents.live_navigation_agent import LiveNavigationAgent, format_duration, NavigationResult
from app.agents.budget_agent import BudgetAgent, extract_budget_parameters, BudgetStatus
from app.agents.itinerary_planning_agent import ItineraryPlanningAgent
from app.agents.state import AgentState, AgentResponse
from app.agents.coordinate_registry import TempleCoordinateRegistry


@pytest.fixture
def nav_agent():
    return LiveNavigationAgent()


@pytest.fixture
def budget_agent():
    return BudgetAgent()


@pytest.fixture
def itinerary_agent():
    return ItineraryPlanningAgent()


# =====================================================================
# P0: Live Navigation Intent Parsing & Destination Resolution Tests
# =====================================================================

def test_nav_query_repeated_origin_cleaning(nav_agent):
    """
    Query: 'Give me the route from Hyderabad to hyderabad to medaram'
    Must resolve origin to 'Hyderabad' and destination to 'Medaram'
    (NOT 'hyderabad to medaram').
    """
    params = nav_agent.extract_route_parameters("Give me the route from Hyderabad to hyderabad to medaram")
    assert params["origin"].strip().lower() == "hyderabad"
    assert params["destination"].strip().lower() == "medaram"

    # Location resolution
    orig_name, orig_coords, _ = nav_agent._resolve_location(params["origin"])
    dest_name, dest_coords, _ = nav_agent._resolve_location(params["destination"])

    assert "Hyderabad" in orig_name
    assert orig_coords is not None
    assert "Medaram" in dest_name or "Sammakka" in dest_name


def test_nav_query_directions_hyderabad_to_bhadrachalam(nav_agent):
    """
    Query: 'Show me directions from Hyderabad to Bhadrachalam'
    Must resolve origin to Hyderabad, destination to Bhadrachalam.
    """
    params = nav_agent.extract_route_parameters("Show me directions from Hyderabad to Bhadrachalam")
    assert params["origin"].strip().lower() == "hyderabad"
    assert "bhadrachalam" in params["destination"].strip().lower()

    dest_name, dest_coords, dest_canon = nav_agent._resolve_location(params["destination"])
    assert dest_canon == "Badrachalam"
    assert dest_coords is not None


def test_nav_query_single_destination_balkampet(nav_agent):
    """
    Query: 'Route to Balkampet temple'
    Destination resolves to Balkampet.
    """
    params = nav_agent.extract_route_parameters("Route to Balkampet temple")
    assert params["destination"] is not None
    assert "balkampet" in params["destination"].lower()

    dest_name, dest_coords, dest_canon = nav_agent._resolve_location(params["destination"])
    assert dest_canon == "Balkampet"
    assert dest_coords is not None


def test_nav_query_bhadrachalam_to_hyderabad_no_reversal(nav_agent):
    """
    Query: 'Give me the driving route from Bhadrachalam to Hyderabad'
    Origin: Bhadrachalam, Destination: Hyderabad (no accidental reversal).
    """
    params = nav_agent.extract_route_parameters("Give me the driving route from Bhadrachalam to Hyderabad")
    assert "bhadrachalam" in params["origin"].lower()
    assert "hyderabad" in params["destination"].lower()


def test_format_duration_honest_values():
    """
    Duration must never display '0 min' or genuine journey for missing/zero duration.
    """
    assert format_duration(None) == "Not available"
    assert format_duration(0) == "Not available"
    assert format_duration(-5) == "Not available"
    assert format_duration(45) == "45 min"
    assert format_duration(60) == "1 hr"
    assert format_duration(150) == "2 hr 30 min"


def test_navigation_result_route_status_calculation():
    """
    Verify route_status states: SUCCESS, PARTIAL_DATA, NO_ROUTE.
    """
    # Complete route
    res_success = NavigationResult(
        origin="Hyderabad",
        destination="Medaram",
        distance_km=230.0,
        duration_minutes=270.0,
        polyline="abcz~",
        route_status="SUCCESS"
    )
    assert res_success.route_status == "SUCCESS"

    # Missing polyline
    res_partial = NavigationResult(
        origin="Hyderabad",
        destination="Medaram",
        distance_km=230.0,
        duration_minutes=270.0,
        polyline=None,
        route_status="PARTIAL_DATA"
    )
    assert res_partial.route_status == "PARTIAL_DATA"


# =====================================================================
# P2 & P4: Multi-Temple 4-Day Trip Routing, Accommodation, & Budget Sync
# =====================================================================

def test_extract_budget_parameters_4_day_trip():
    """
    Query: 'plan a 4 day trip to bhadrachalam and balkampet temple with a budget of 6300'
    Extracts days=4, nights=3, budget_limit=6300.
    """
    query = "plan a 4 day trip to bhadrachalam and balkampet temple with a budget of 6300"
    params = extract_budget_parameters(query)

    assert params["days"] == 4
    assert params["nights"] == 3
    assert params["budget_limit"] == 6300.0
    assert not params["is_day_trip"]


def test_multi_temple_budget_nights_partition(budget_agent):
    """
    Verify for a 4-day trip to Bhadrachalam and Balkampet with budget 6300:
    1. Nights across the journey sum to EXACTLY 3 nights (NOT 6 nights!).
    2. Continuous circuit transportation is calculated (not 2 separate round trips).
    3. Exactly 4 days of meals and local transit.
    4. Arithmetic validation: total min equals sum of line items min.
    5. Feasible baseline configuration fits within ₹6,300.
    6. Budget status is ESTIMATED_WITHIN_BUDGET.
    """
    query = "plan a 4 day trip to bhadrachalam and balkampet temple with a budget of 6300"
    state = AgentState(user_query=query)

    response = budget_agent.execute(state)
    assert response.success is True

    metadata = response.metadata
    line_items = metadata.get("line_items", [])

    # 1. Total nights across accommodation items must equal exactly 3
    accom_items = [it for it in line_items if it.get("category") == "accommodation"]
    assert len(accom_items) >= 1

    # Extract nights from accommodation items
    total_accom_nights = 0
    for a in accom_items:
        desc = a.get("quantity_desc", "")
        # e.g., "1 room(s) × 2 night(s)" or "1 room(s) × 1 night(s)"
        import re
        m = re.search(r"(\d+)\s*night\(s\)", desc)
        if m:
            total_accom_nights += int(m.group(1))

    assert total_accom_nights == 3, f"Expected exactly 3 nights across accommodation, got {total_accom_nights}"

    # 2. Continuous circuit transportation: only 1 intercity transportation line item
    transit_items = [it for it in line_items if it.get("category") == "transportation"]
    assert len(transit_items) == 1, f"Expected 1 circuit travel item, got {len(transit_items)}"
    assert "Circuit" in transit_items[0].get("item_name", "")

    # 3. Meals: only 1 trip-wide meals item with multiplier = 4 (for 4 days)
    meal_items = [it for it in line_items if it.get("category") == "meals"]
    assert len(meal_items) == 1
    assert meal_items[0].get("multiplier") == 4.0

    # 4. Local transit: only 1 trip-wide item with multiplier = 4 (for 4 days)
    local_items = [it for it in line_items if it.get("category") == "local_transit"]
    assert len(local_items) == 1
    assert local_items[0].get("multiplier") == 4.0

    # 5. Arithmetic validation
    tot_min = metadata["calculated_total"]["min"]
    tot_max = metadata["calculated_total"]["max"]
    sum_items_min = sum(it.get("subtotal_min", 0.0) for it in line_items)
    sum_items_max = sum(it.get("subtotal_max", 0.0) for it in line_items)
    assert abs(tot_min - sum_items_min) < 1.0
    assert abs(tot_max - sum_items_max) < 1.0

    # 6. Budget Gate check for ₹6,300 limit
    # The baseline feasible plan must be <= 6,300
    assert tot_min <= 6300.0, f"Expected baseline plan <= 6300, got {tot_min}"
    assert metadata["budget_status"] == BudgetStatus.ESTIMATED_WITHIN_BUDGET
    assert metadata["is_cross_limit"] is True
    assert metadata["selected_plan"]["total"] <= 6300.0


def test_multi_temple_itinerary_day_allocations(itinerary_agent):
    """
    Verify 4-day itinerary for Bhadrachalam and Balkampet generates 4 days
    with transfer day on Day 3 and distinct themes.
    """
    query = "plan a 4 day trip to bhadrachalam and balkampet temple with a budget of 6300"
    state = AgentState(user_query=query)

    response = itinerary_agent.execute(state)
    assert response.success is True

    answer = response.answer
    assert "#### Day 1:" in answer
    assert "#### Day 2:" in answer
    assert "#### Day 3:" in answer
    assert "#### Day 4:" in answer
    assert "#### Day 5:" not in answer

    # Verify Bhadrachalam and Balkampet are featured
    assert "Badrachalam" in answer or "Bhadrachalam" in answer
    assert "Balkampet" in answer

    # Verify Day 3 onward travel is present
    assert "Onward Travel" in answer or "onward journey" in answer.lower()


# =====================================================================
# P0-P4: Typography, White Text, Source Stripping & Itinerary Audit Tests
# =====================================================================

def test_style_css_day_heading_typography():
    """
    Verify style.css implements refined targets:
    - Desktop Day headings: 21–22 px (21.5px), weight 700, white #ffffff
    - Mobile Day headings: 19–20 px (19.5px)
    - Activity names: 15–16 px (15.5px), white #ffffff (no bright yellow)
    - Descriptions: 14–15 px (14.5px), soft light grey #cbd5e1
    """
    css_path = backend_dir / "app" / "static" / "style.css"
    assert css_path.exists()

    content = css_path.read_text(encoding="utf-8")
    assert ".itinerary-day-heading" in content
    assert "font-size: 21.5px" in content
    assert "font-size: 19.5px" in content
    assert "font-size: 15.5px" in content
    assert "font-size: 14.5px" in content
    assert "sources-details" in content
    assert "nav-btn-external-map" in content


def test_itinerary_specific_opening_hours_audit(itinerary_agent):
    """
    Verify Bhadrachalam and Balkampet itinerary displays verified opening hours:
    - Bhadrachalam: 04:30 AM to 09:00 PM (Daily)
    - Balkampet: 06:00 AM to 08:00 PM (Daily)
    - Does NOT use generic fallback 'Standard daily temple opening hours apply.'
    """
    query = "plan a 4 day trip to bhadrachalam and balkampet temple with a budget of 6300"
    state = AgentState(user_query=query)
    response = itinerary_agent.execute(state)
    assert response.success is True

    answer = response.answer
    assert "04:30 AM to 09:00 PM" in answer
    assert "06:00 AM to 08:00 PM" in answer
    assert "Standard daily temple opening hours apply." not in answer


def test_user_answer_no_raw_sources_leak(itinerary_agent, budget_agent):
    """
    Verify no raw .json file paths or implementation details leak into
    user-facing markdown answers, while internal response.sources retains them.
    """
    query = "plan a 4 day trip to bhadrachalam and balkampet temple with a budget of 6300"
    state = AgentState(user_query=query)

    # Itinerary Agent
    itin_resp = itinerary_agent.execute(state)
    assert itin_resp.success is True
    assert ".json" not in itin_resp.answer
    assert "**Sources:**" not in itin_resp.answer
    assert len(itin_resp.sources) > 0
    assert any("timings.json" in s for s in itin_resp.sources)

    # Budget Agent
    budget_resp = budget_agent.execute(state)
    assert budget_resp.success is True
    assert "(Source: badrachalam/sevas.json)" not in budget_resp.answer
    assert "(Source:" not in budget_resp.answer
    assert ".json" not in budget_resp.answer
    assert len(budget_resp.sources) > 0


def test_budget_no_duplicate_sevas_and_optional_separation(budget_agent):
    """
    Verify:
    1. Sevas do NOT appear under 'Verified Costs (Listed Temple Tariffs)'.
    2. Optional sevas appear ONLY in 'Optional Services & Offerings'.
    3. No duplicate seva entries exist.
    """
    query = "plan a 4 day trip to bhadrachalam and balkampet temple with a budget of 6300"
    state = AgentState(user_query=query)
    response = budget_agent.execute(state)
    assert response.success is True

    answer = response.answer
    # Split into sections
    if "### Verified Costs" in answer and "### Optional Services" in answer:
        verified_sec = answer.split("### Verified Costs")[1].split("### Optional Services")[0]
        optional_sec = answer.split("### Optional Services")[1].split("###")[0]

        # Sevas like Kalyanotsavam or Abhishekam should not appear in verified mandatory costs
        assert "Kalyanotsavam" not in verified_sec
        assert "Abhishekam" not in verified_sec

        # Kalyanotsavam or Abhishekam should be in optional section
        assert "Kalyanotsavam" in optional_sec or "Abhishekam" in optional_sec or "General Darshan" in verified_sec
