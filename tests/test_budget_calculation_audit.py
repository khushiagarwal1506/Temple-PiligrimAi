"""
PilgrimAI - Comprehensive Budget Calculation & Hard Gate Audit Suite

Strict regression test suite verifying all 10 P0 criteria:
1. Vemulavada, four people, ₹5,000 (preserves 4 pax, rooms, travel, meals, separates listed vs estimates).
2. Basara from Hyderabad, ₹2,000 day trip (round-trip travel included, stay excluded, optional seva separated).
3. Multi-day trip whose accommodation alone exceeds the budget (triggers OVER_BUDGET, never Within Budget).
4. Trip with missing/unstated origin (discloses benchmark, tags confirmation requirement).
5. Trip with optional paid sevas (optional sevas separated from baseline mandatory total).
6. Free-darshan trip where travel and meals still cost money (total > 0, darshan ₹0).
7. Normal seva-price query that must NOT produce a budget card (is_budget_request is False).
8. Different group sizes, nights, and room requirements (scaling invariant).
9. Budget calculations across all 20 registered temples.
10. Frontend rendering metadata (line items, estimates, totals, status enum).
"""

import sys
import re
from pathlib import Path
import pytest

root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agents.state import AgentState, AgentResponse
from app.agents.router import AgentRouter
from app.agents.coordinator import MasterAgentCoordinator
from app.agents.budget_agent import (
    BudgetAgent,
    CostProvenance,
    BudgetStatus,
    extract_budget_parameters,
    BENCHMARK_DISTANCES_FROM_HYDERABAD,
    TEMPLE_CANONICAL_MAP
)
from app.agents.tools.rag_tool import RAGTool


@pytest.fixture(scope="module")
def rag_tool():
    return RAGTool()


@pytest.fixture(scope="module")
def budget_agent(rag_tool):
    return BudgetAgent(rag_tool=rag_tool)


@pytest.fixture(scope="module")
def active_router():
    r = AgentRouter()
    r.register_agent("temple_information")
    r.register_agent("budget")
    r.register_agent("itinerary_planning")
    r.register_agent("live_weather")
    r.register_agent("live_navigation")
    return r


# =====================================================================
# 1. Vemulavada: 4 people, ₹5,000 budget
# =====================================================================

def test_01_vemulavada_four_people_five_thousand_budget(budget_agent):
    """
    Prompt: "I'm planning a family trip to Vemulavada with four people and a budget of ₹5,000.
    Estimate the expenses for travel, accommodation, meals and temple services, separating listed prices from estimates."
    """
    query = (
        "I'm planning a family trip to Vemulavada with four people and a budget of ₹5,000. "
        "Estimate the expenses for travel, accommodation, meals and temple services, separating listed prices from estimates."
    )
    state = AgentState(user_query=query)
    resp = budget_agent.process(state)

    assert resp.success is True
    meta = resp.metadata

    # 1. Parameter extraction: pax = 4, budget_limit = 5000.0
    params = meta["parameters"]
    assert params["pax"] == 4, f"Expected 4 pax, got {params['pax']}"
    assert params["pax_assumed"] is False
    assert params["budget_limit"] == 5000.0

    # 2. Rooms: 4 pax in standard lodging requires 2 rooms
    assert params["rooms"] >= 2

    # 3. Complete cost model includes travel, meals, stay, entry
    cats = [it["category"] for it in meta["line_items"]]
    assert "travel" in cats
    assert "accommodation" in cats
    assert "meals" in cats
    assert "entry" in cats

    # 4. Total represents complete trip cost (NOT the old partial subtotal ₹1,225–₹2,525)
    tot = meta["calculated_total"]
    assert tot["min"] > 2500.0, f"Expected complete total > ₹2,500, got {tot['min']}"

    # 5. Separation of listed prices vs estimates
    provenances = [it["provenance"] for it in meta["line_items"]]
    assert "LISTED_PRICE" in provenances or CostProvenance.LISTED_PRICE.value in provenances
    assert "ESTIMATE" in provenances or CostProvenance.ESTIMATE.value in provenances

    # 6. Prices requiring confirmation are enumerated
    assert len(meta["confirmation_required"]) > 0

    # 7. Four people retained in text
    assert "4 pilgrim(s)" in resp.answer or "4 person" in resp.answer or "Group of 4" in resp.answer

    # 8. Hard budget gate evaluation & feasible plan selection
    assert meta["budget_status"] == BudgetStatus.ESTIMATED_WITHIN_BUDGET.value
    assert meta["is_cross_limit"] is True
    assert meta["selected_plan"]["total"] == 4140.0
    assert meta["selected_plan"]["total"] <= 5000.0
    assert meta["selected_plan"]["buffer"] == 860.0
    assert meta["selected_plan"]["within_budget"] is True
    assert meta["broader_range"]["min"] == 4140.0
    assert meta["broader_range"]["max"] == 6920.0
    assert meta["broader_range"]["exceeds_budget"] is True
    assert meta["broader_range"]["excess_amount"] == 1920.0
    assert "Selected Feasible Configuration" in resp.answer
    assert "Broader Possible Cost Range" in resp.answer
    assert "Selected Feasible Plan Total" in resp.answer


# =====================================================================
# 2. Basara from Hyderabad: ₹2,000 day trip
# =====================================================================

def test_02_basara_day_trip_from_hyderabad_two_thousand(budget_agent):
    """
    Prompt: "I have ₹2,000 for a day trip to Basara from Hyderabad.
    Help me prepare a budget and tell me which costs need confirmation."
    """
    query = (
        "I have ₹2,000 for a day trip to Basara from Hyderabad. "
        "Help me prepare a budget and tell me which costs need confirmation."
    )
    state = AgentState(user_query=query)
    resp = budget_agent.process(state)

    assert resp.success is True
    meta = resp.metadata

    # 1. Parameter extraction: day trip (0 nights), origin = Hyderabad, budget_limit = 2000.0
    params = meta["parameters"]
    assert params["nights"] == 0
    assert params["is_day_trip"] is True
    assert params["origin"] == "Hyderabad"
    assert params["origin_assumed"] is False
    assert params["budget_limit"] == 2000.0

    # 2. Accommodation is excluded for a day trip
    stay_item = next((it for it in meta["line_items"] if it["category"] == "accommodation"), None)
    assert stay_item is not None
    assert stay_item["subtotal_min"] == 0.0
    assert stay_item["subtotal_max"] == 0.0

    # 3. Travel: Hyderabad to Basara (~200 km) round-trip is included
    travel_item = next((it for it in meta["line_items"] if it["category"] == "travel"), None)
    assert travel_item is not None
    assert travel_item["subtotal_min"] >= 400.0  # At least ₹400 for 400 km RT
    assert travel_item["provenance"] in ("ESTIMATE", CostProvenance.ESTIMATE.value)

    # 4. Meals are included for day trip
    meals_item = next((it for it in meta["line_items"] if it["category"] == "meals"), None)
    assert meals_item is not None
    assert meals_item["subtotal_min"] > 0.0

    # 5. Baseline fits within ₹2,000
    tot = meta["calculated_total"]
    assert tot["min"] <= 2000.0
    assert meta["budget_status"] == BudgetStatus.ESTIMATED_WITHIN_BUDGET.value

    # 6. Never uses ₹250-₹500 Manorath figure as total trip cost
    assert "**Total Estimated Pilgrimage Cost**: **₹250 – ₹500**" not in resp.answer
    assert tot["min"] != 250.0 or tot["max"] != 500.0

    # 7. Confirmation items enumerated
    assert len(meta["confirmation_required"]) > 0


# =====================================================================
# 3. Multi-day trip whose accommodation alone exceeds the budget
# =====================================================================

def test_03_over_budget_multi_day_accommodation(budget_agent):
    """
    Multi-day trip where lodging/stay exceeds tight budget limit -> OVER_BUDGET.
    """
    query = "Plan a budget for 4 days trip to Yadadri for 4 people with a budget of ₹1,000"
    state = AgentState(user_query=query)
    resp = budget_agent.process(state)

    assert resp.success is True
    meta = resp.metadata

    # Accommodation + travel for 4 pax for 4 days easily exceeds ₹1,000
    assert meta["calculated_total"]["min"] > 1000.0
    assert meta["budget_status"] == BudgetStatus.OVER_BUDGET.value
    assert "OVER_BUDGET" in resp.answer or "Over Budget" in resp.answer
    assert "Within Budget" not in resp.answer


# =====================================================================
# 4. Trip with missing / unstated origin
# =====================================================================

def test_04_unstated_origin_benchmark_disclosure(budget_agent):
    """
    When origin is unstated, benchmark from Hyderabad is used and clearly disclosed.
    """
    query = "Estimate the budget for a trip to Dharmapuri for 2 people"
    state = AgentState(user_query=query)
    resp = budget_agent.process(state)

    assert resp.success is True
    meta = resp.metadata
    params = meta["parameters"]

    assert params["origin"] == "Hyderabad"
    assert params["origin_assumed"] is True
    assert "Assumed" in resp.answer or "Benchmark" in resp.answer
    assert any("Transportation fare" in c for c in meta["confirmation_required"])


# =====================================================================
# 5. Trip with optional paid sevas
# =====================================================================

def test_05_optional_sevas_separated_from_baseline(budget_agent):
    """
    Optional paid sevas are listed with prices but NOT added to mandatory baseline total.
    """
    query = "What is the budget for Basara day trip?"
    state = AgentState(user_query=query)
    resp = budget_agent.process(state)

    assert resp.success is True
    # Aksharabhyasam is an optional seva at Basara, should appear in Optional section
    assert "Optional Services" in resp.answer or "Optional Seva" in resp.answer
    assert "Aksharabhyasam" in resp.answer or "Aksharabyasam" in resp.answer


# =====================================================================
# 6. Free-darshan trip where travel and meals still cost money
# =====================================================================

def test_06_free_darshan_nonzero_complete_total(budget_agent):
    """
    General entry is FREE, but travel + meals + local transit ensure trip total > ₹0.
    """
    query = "Calculate complete trip cost for Chilkur Balaji for 2 people"
    state = AgentState(user_query=query)
    resp = budget_agent.process(state)

    assert resp.success is True
    tot = resp.metadata["calculated_total"]

    # Entry is ₹0, but total trip cost must be > 0
    assert tot["min"] > 0.0
    entry_item = next((it for it in resp.metadata["line_items"] if it["category"] == "entry"), None)
    assert entry_item is not None
    assert entry_item["is_free"] is True


# =====================================================================
# 7. Normal seva-price query must NOT produce a budget card
# =====================================================================

def test_07_normal_seva_price_query_no_unsolicited_budget(active_router, budget_agent):
    """
    A simple request for listed seva prices routes to temple_information,
    and has is_budget_request = False so UI does not render a budget card.
    """
    query = "What are the seva prices at Vemulavada?"
    decision = active_router.route(query)

    # Must route to temple_information, NOT budget agent!
    assert decision.selected_agent == "temple_information"
    assert decision.intent != "budget"

    # Even if processed directly by budget_agent, is_budget_request is False
    state = AgentState(user_query=query)
    resp = budget_agent.process(state)
    assert resp.metadata["is_budget_request"] is False


# =====================================================================
# 8. Scaling invariant: different group sizes, nights, and rooms
# =====================================================================

def test_08_group_size_and_nights_scaling(budget_agent):
    """
    Scaling: 1 person vs 4 people scaling on per-person items.
    """
    resp_1 = budget_agent.process(AgentState(user_query="trip budget for Yadadri for 1 person for 1 day"))
    resp_4 = budget_agent.process(AgentState(user_query="trip budget for Yadadri for 4 people for 1 day"))

    tot_1 = resp_1.metadata["calculated_total"]["min"]
    tot_4 = resp_4.metadata["calculated_total"]["min"]

    # 4 people costs significantly more than 1 person
    assert tot_4 > tot_1 * 2


# =====================================================================
# 9. Budget calculations across all 20 registered temples
# =====================================================================

def test_09_all_twenty_temples_budget_calculation(budget_agent):
    """
    Verifies that complete budget calculation executes cleanly for all 20 canonical temples.
    """
    for canonical_name in BENCHMARK_DISTANCES_FROM_HYDERABAD.keys():
        query = f"I want to plan a trip to {canonical_name} with 2 people and a budget of ₹4,000"
        state = AgentState(user_query=query)
        resp = budget_agent.process(state)

        assert resp.success is True, f"Failed for {canonical_name}"
        assert resp.metadata["calculated_total"]["is_quantified"] is True
        assert resp.metadata["calculated_total"]["min"] > 0
        assert resp.metadata["budget_status"] in [
            BudgetStatus.WITHIN_BUDGET.value,
            BudgetStatus.ESTIMATED_WITHIN_BUDGET.value,
            BudgetStatus.OVER_BUDGET.value
        ]


# =====================================================================
# 10. Frontend rendering metadata contract
# =====================================================================

def test_10_frontend_metadata_contract(budget_agent):
    """
    Verifies metadata contains all keys required by app.js:
    parameters, budget_status, calculated_total, line_items, confirmation_required, is_budget_request.
    """
    query = "Family trip to Vemulavada with four people and a budget of ₹5,000"
    state = AgentState(user_query=query)
    resp = budget_agent.process(state)
    meta = resp.metadata

    assert "parameters" in meta
    assert "budget_status" in meta
    assert "calculated_total" in meta
    assert "line_items" in meta
    assert "confirmation_required" in meta
    assert "is_budget_request" in meta

    # Line item schema validation
    for item in meta["line_items"]:
        assert "category" in item
        assert "item_name" in item
        assert "min_cost" in item
        assert "max_cost" in item
        assert "unit" in item
        assert "multiplier" in item
        assert "quantity_desc" in item
        assert "subtotal_min" in item
        assert "subtotal_max" in item
        assert "provenance" in item
        assert "confirmation_required" in item


# =====================================================================
# 11. Regression: Minimum below budget but maximum exceeds (Cross-Limit)
# =====================================================================

def test_11_regression_cross_limit_feasible_plan_selection(budget_agent):
    """
    Scenario: Minimum is below budget (₹4,140 <= ₹5,000) but maximum exceeds it (₹6,920 > ₹5,000).
    Verifies:
    1. is_cross_limit is True.
    2. Constructs and displays specific feasible configuration using disclosed lower-cost options.
    3. Calculates every line item for that configuration and verifies total <= budget (₹4,140 <= ₹5,000).
    4. Displays broader possible cost range separately and explicitly discloses higher options exceed limit.
    5. Returns ESTIMATED_WITHIN_BUDGET with selected plan disclosure.
    """
    query = "I'm planning a family trip to Vemulavada with four people. Our hard budget is 5000 rupees. Can we afford this?"
    state = AgentState(user_query=query)
    resp = budget_agent.process(state)

    assert resp.success is True
    meta = resp.metadata

    # Verification of cross-limit flag
    assert meta["is_cross_limit"] is True

    # Feasible plan verification
    selected_plan = meta["selected_plan"]
    assert selected_plan["within_budget"] is True
    assert selected_plan["total"] <= 5000.0
    assert selected_plan["total"] == 4140.0
    assert selected_plan["buffer"] == 860.0

    # Broader range verification
    broader = meta["broader_range"]
    assert broader["min"] == 4140.0
    assert broader["max"] == 6920.0
    assert broader["exceeds_budget"] is True
    assert broader["excess_amount"] == 1920.0

    # Status verification
    assert meta["budget_status"] == BudgetStatus.ESTIMATED_WITHIN_BUDGET.value

    # Markdown verification
    assert "Selected Feasible Configuration (Fits Within ₹5,000 Hard Limit)" in resp.answer
    assert "Broader Possible Cost Range & Higher-Tier Options (Exceeds Limit)" in resp.answer
    assert "Selected Feasible Plan Total" in resp.answer
    assert "₹4,140" in resp.answer
    assert "₹6,920" in resp.answer
    assert "exceed your ₹5,000 hard limit by up to ₹1,920" in resp.answer

    # Meal benchmark labelled ESTIMATE
    meals_item = next(it for it in meta["line_items"] if it["category"] == "meals")
    assert meals_item["provenance"] == CostProvenance.ESTIMATE.value


# =====================================================================
# 12. Regression: All configurations exceed the budget (OVER_BUDGET)
# =====================================================================

def test_12_regression_all_configurations_exceed_budget(budget_agent):
    """
    Scenario: Multi-day family trip where all possible configurations exceed the limit (min > limit).
    Verifies:
    1. is_cross_limit is False.
    2. Returns strictly OVER_BUDGET (never Within Budget).
    3. Selected plan within_budget is False.
    4. Explains shortfall and gives defensible alternatives.
    """
    query = "Plan a budget for 4 days trip to Yadadri for 4 people with a budget of ₹1,000"
    state = AgentState(user_query=query)
    resp = budget_agent.process(state)

    assert resp.success is True
    meta = resp.metadata

    assert meta["is_cross_limit"] is False
    assert meta["budget_status"] == BudgetStatus.OVER_BUDGET.value
    assert meta["selected_plan"]["within_budget"] is False
    assert meta["calculated_total"]["min"] > 1000.0

    # Text must explicitly state OVER_BUDGET and not claim within budget
    assert "OVER_BUDGET" in resp.answer
    assert "Within Budget" not in resp.answer
    assert "No feasible configuration could be demonstrated" in resp.answer


# =====================================================================
# 13. Regression: Full estimated range fits within budget
# =====================================================================

def test_13_regression_full_estimated_range_fits_budget(budget_agent):
    """
    Scenario: Day trip with a comfortable budget where entire range fits (max <= limit).
    Verifies:
    1. is_cross_limit is False.
    2. Calculated total max <= limit.
    3. Status is ESTIMATED_WITHIN_BUDGET.
    4. Selected plan within_budget is True.
    5. No warning that higher options exceed limit.
    """
    query = "I have ₹2,000 for a day trip to Basara from Hyderabad. Help me prepare a budget."
    state = AgentState(user_query=query)
    resp = budget_agent.process(state)

    assert resp.success is True
    meta = resp.metadata

    assert meta["is_cross_limit"] is False
    assert meta["calculated_total"]["max"] <= 2000.0
    assert meta["budget_status"] == BudgetStatus.ESTIMATED_WITHIN_BUDGET.value
    assert meta["selected_plan"]["within_budget"] is True
    assert meta["broader_range"]["exceeds_budget"] is False

    # Standard breakdown displayed without exceeding warning
    assert "fits within your ₹2,000 limit" in resp.answer
    assert "Exceeds Limit" not in resp.answer
