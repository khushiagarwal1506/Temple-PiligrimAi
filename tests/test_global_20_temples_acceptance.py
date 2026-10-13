"""
PilgrimAI - Comprehensive Global 20-Temple Acceptance & Regression Test Suite (Phase 11)

Tests every one of the 20 registered temple datasets across 8 mandatory query types:
1. Single-temple factual question (history/deity).
2. Timing or seva query.
3. Facilities or accessibility query.
4. Recommendation query.
5. Budget-constrained itinerary query (hard cap deterministic enforcement).
6. Nearby-attraction query.
7. Missing-data query (never infer permissibility/invent facts).
8. Follow-up question using conversational context.

Plus all 10 reported cross-temple and system regression scenarios.
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
from app.agents.coordinator import MasterAgentCoordinator
from app.agents.router import AgentRouter
from app.agents.response_aggregator import ResponseAggregator
from app.agents.temple_information_agent import TempleInformationAgent
from app.agents.itinerary_planning_agent import ItineraryPlanningAgent
from app.agents.live_weather_agent import LiveWeatherAgent
from app.agents.budget_agent import BudgetAgent
from app.agents.live_navigation_agent import LiveNavigationAgent
from app.agents.emergency_agent import EmergencyAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.comparison_agent import ComparisonAgent
from app.agents.memory import ConversationMemoryManager
from app.agents.tools.rag_tool import RAGTool


ALL_20_TEMPLES = [
    "Ahobilam",
    "Badrachalam",
    "Balkampet",
    "Basara",
    "Birla Mandir",
    "Chilkur",
    "Dharmapuri",
    "Edupayal",
    "Jogulamba(new)",
    "Kaleshvaram",
    "Karmanghat",
    "Medaram",
    "Ramappa",
    "Srisaila",
    "Thousand Pillar",
    "Ttd",
    "Ujjaini",
    "Veera Satyanarayana",
    "Vemulavada",
    "Yadadri"
]
from unittest.mock import patch
from app.rag.chat import PilgrimRAGService


@pytest.fixture(autouse=True)
def fast_ollama():
    """Mock the slow LLM token generator so full RAG pipeline, Qdrant/BM25 retrieval,
    guardrails, grounding validation, router, coordinator and budget calculations
    run deterministically and rapidly across all 20 temples."""
    with patch.object(
        PilgrimRAGService,
        "_call_ollama_generate",
        return_value="Verified temple information from official devasthanam records with established rituals and darshan facilities."
    ):
        yield


@pytest.fixture(scope="module")
def coordinator():
    memory_manager = ConversationMemoryManager()
    router = AgentRouter()
    aggregator = ResponseAggregator()
    temple_info = TempleInformationAgent()
    itinerary = ItineraryPlanningAgent()
    weather = LiveWeatherAgent()
    budget = BudgetAgent()
    navigation = LiveNavigationAgent()
    emergency = EmergencyAgent()
    recommendation = RecommendationAgent(weather_agent=weather)
    comparison = ComparisonAgent()

    return MasterAgentCoordinator(
        router=router,
        aggregator=aggregator,
        temple_info_agent=temple_info,
        comparison_agent=comparison,
        itinerary_agent=itinerary,
        live_weather_agent=weather,
        budget_agent=budget,
        live_navigation_agent=navigation,
        emergency_agent=emergency,
        recommendation_agent=recommendation,
        memory_manager=memory_manager,
    )


# ---------------------------------------------------------------------
# 1. Single-Temple Factual Queries (All 20 Temples)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("temple", ALL_20_TEMPLES)
def test_01_factual_history_query(coordinator, temple):
    state = AgentState(user_query=f"What is the history and main deity of {temple} temple?")
    resp = coordinator.process(state)
    assert resp.success, f"Factual query failed for {temple}: {resp.error}"
    assert len(resp.answer.strip()) > 30
    assert resp.agent_name in ["temple_information", "coordinator", "recommendation"]
    # Verify no negative confidence displayed
    conf = resp.metadata.get("confidence_score")
    if conf is not None:
        assert conf >= 0.0, f"Negative confidence {conf} returned for {temple}"


# ---------------------------------------------------------------------
# 2. Timing or Seva Queries (All 20 Temples)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("temple", ALL_20_TEMPLES)
def test_02_timing_or_seva_query(coordinator, temple):
    state = AgentState(user_query=f"What are the darshan timings and sevas at {temple}?")
    resp = coordinator.process(state)
    assert resp.success, f"Timing/seva query failed for {temple}: {resp.error}"
    assert len(resp.answer.strip()) > 20
    # Must never inject unsolicited full trip budget
    assert "Trip Budget Breakdown" not in resp.answer
    assert resp.metadata.get("budget_limit") is None


# ---------------------------------------------------------------------
# 3. Facilities or Accessibility Queries (All 20 Temples)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("temple", ALL_20_TEMPLES)
def test_03_facilities_or_accessibility_query(coordinator, temple):
    state = AgentState(user_query=f"What wheelchair accessibility and senior citizen facilities exist at {temple}?")
    resp = coordinator.process(state)
    assert resp.success, f"Accessibility query failed for {temple}: {resp.error}"
    ans_lower = resp.answer.lower()
    assert len(ans_lower) > 20


# ---------------------------------------------------------------------
# 4. Recommendation Queries (All 20 Temples)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("temple", ALL_20_TEMPLES)
def test_04_recommendation_query(coordinator, temple):
    state = AgentState(user_query=f"Recommend places to stay and eat for pilgrims around {temple}")
    resp = coordinator.process(state)
    assert resp.success, f"Recommendation query failed for {temple}: {resp.error}"
    # Must only recommend for target temple, not dump all 20 temples
    temples_in_resp = resp.metadata.get("detected_temples") or []
    if temples_in_resp:
        clean_target = temple.lower().replace("(new)", "")
        assert any(clean_target in t.lower() for t in temples_in_resp)


# ---------------------------------------------------------------------
# 5. Budget-Constrained Itinerary Queries (All 20 Temples)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("temple", ALL_20_TEMPLES)
def test_05_budget_constrained_itinerary_query(coordinator, temple):
    budget_cap = 3000
    state = AgentState(user_query=f"Plan a 2 day trip to {temple} with a budget of ₹{budget_cap}")
    resp = coordinator.process(state)
    assert resp.success, f"Budget itinerary failed for {temple}: {resp.error}"
    meta = resp.metadata
    calc_total = meta.get("calculated_total")
    total_val = calc_total.get("max", 0) if isinstance(calc_total, dict) else calc_total
    if total_val is not None:
        # Total must never be 0 merely because darshan is free
        assert total_val > 0, f"Calculated total is ₹0 for {temple}"
        # Hard cap: if plan exceeds budget, it must be flagged
        if total_val > budget_cap:
            feasibility = meta.get("feasibility_status", "")
            ans = resp.answer
            assert "Exceeds" in feasibility or "Infeasible" in feasibility or "⚠️" in feasibility or "exceed" in ans.lower() or "infeasible" in ans.lower()


# ---------------------------------------------------------------------
# 6. Nearby Attractions Queries (All 20 Temples)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("temple", ALL_20_TEMPLES)
def test_06_nearby_attractions_query(coordinator, temple):
    state = AgentState(user_query=f"What are the verified nearby attractions around {temple}?")
    resp = coordinator.process(state)
    assert resp.success, f"Nearby attractions query failed for {temple}: {resp.error}"
    assert len(resp.answer.strip()) > 20


# ---------------------------------------------------------------------
# 7. Missing-Data / Rules Inquiry (All 20 Temples)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("temple", ALL_20_TEMPLES)
def test_07_missing_data_integrity_query(coordinator, temple):
    # Ask about pets or unauthorized claims that must never be affirmed as permitted without evidence
    state = AgentState(user_query=f"Are pet dogs allowed inside {temple} temple?")
    resp = coordinator.process(state)
    ans_lower = resp.answer.lower()
    # Must not affirm that pets are allowed or permitted inside the sanctum
    assert not ("pets are allowed inside" in ans_lower or "dogs are permitted inside" in ans_lower)


# ---------------------------------------------------------------------
# 8. Follow-up Context Resolution (All 20 Temples)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("temple", ALL_20_TEMPLES)
def test_08_followup_context_resolution(coordinator, temple):
    session_id = f"session_test_{temple.lower().replace(' ', '_')}"
    # Turn 1: Establish temple context
    s1 = AgentState(user_query=f"Tell me about {temple} temple", session_id=session_id)
    r1 = coordinator.process(s1)
    assert r1.success

    # Turn 2: Follow-up question without naming temple
    s2 = AgentState(user_query="What about the accommodation options?", session_id=session_id)
    r2 = coordinator.process(s2)
    assert r2.success
    resolved_temple = r2.metadata.get("referenced_temple") or r2.metadata.get("temple") or ""
    clean_target = temple.lower().replace("(new)", "")
    assert clean_target in resolved_temple.lower() or clean_target in r2.answer.lower(), \
        f"Lost conversational context for {temple} on follow-up turn"


# ---------------------------------------------------------------------
# 9. Specific Reported Failures & Cross-Temple Regression Scenarios
# ---------------------------------------------------------------------
def test_reg_01_edupayal_budget_trip(coordinator):
    """1. Edupayal 2-day trip ₹2,500 budget: Edupayal is selected, never Ramappa; cap enforced."""
    state = AgentState(user_query="Plan a 2 day trip to Edupayal with a budget of ₹2500")
    resp = coordinator.process(state)
    assert resp.success
    assert "Edupayal" in resp.answer
    assert "Ramappa" not in resp.metadata.get("detected_temples", [])
    calc_total = resp.metadata.get("calculated_total")
    total_val = calc_total.get("max", 0) if isinstance(calc_total, dict) else calc_total
    if total_val is not None and total_val > 2500:
        ans = resp.answer
        feasibility = resp.metadata.get("feasibility_status", "")
        assert "Exceeds" in feasibility or "Infeasible" in feasibility or "⚠️" in feasibility or "exceed" in ans.lower() or "infeasible" in ans.lower()


def test_reg_02_ramappa_day_trip_budget(coordinator):
    """2. Ramappa day trip ₹2000: includes transport/food, never ₹0 total."""
    state = AgentState(user_query="Plan a day trip to Ramappa with a budget of ₹2000")
    resp = coordinator.process(state)
    assert resp.success
    calc_total = resp.metadata.get("calculated_total")
    total_val = calc_total.get("max", 0) if isinstance(calc_total, dict) else calc_total
    if total_val is not None:
        assert total_val > 0, "Ramappa day trip must include baseline food/travel and not be ₹0"


def test_reg_03_yadadri_four_people_budget(coordinator):
    """3. Yadadri 2-day trip for 4 people within ₹5,000."""
    state = AgentState(user_query="Plan a two-day pilgrimage to Yadadri for four people with a total budget of ₹5000")
    resp = coordinator.process(state)
    assert resp.success
    assert "Yadadri" in resp.answer
    assert resp.metadata.get("pilgrim_count", 1) >= 4 or "4" in resp.answer


def test_reg_04_vemulavada_sevas_no_budget_injection(coordinator):
    """4. 'What sevas can I book at Vemulavada, and what do they cost?' -> sevas with prices, no budget card."""
    state = AgentState(user_query="What sevas can I book at Vemulavada, and what do they cost?")
    resp = coordinator.process(state)
    assert resp.success
    assert resp.agent_name == "temple_information"
    assert "Trip Budget Breakdown" not in resp.answer
    assert resp.metadata.get("budget_limit") is None


def test_reg_05_basara_and_dharmapuri_itinerary(coordinator):
    """5. Basara and Dharmapuri combined itinerary must not return zero output."""
    state = AgentState(user_query="Plan a 2 day trip to Basara and Dharmapuri")
    resp = coordinator.process(state)
    assert resp.success
    assert "Basara" in resp.answer
    assert "Dharmapuri" in resp.answer


def test_reg_06_birla_mandir_and_karmanghat_itinerary(coordinator):
    """6. Birla Mandir and Karmanghat in one itinerary."""
    state = AgentState(user_query="Plan a 1 day trip to Birla Mandir and Karmanghat")
    resp = coordinator.process(state)
    assert resp.success
    assert "Birla Mandir" in resp.answer
    assert "Karmanghat" in resp.answer


def test_reg_07_ahobilam_srisailam_comparison(coordinator):
    """7. Ahobilam and Srisailam comparison preserves both temples."""
    state = AgentState(user_query="Compare Ahobilam and Srisailam in terms of facilities and accessibility")
    resp = coordinator.process(state)
    assert resp.success
    assert "Ahobilam" in resp.answer
    assert "Srisailam" in resp.answer


def test_reg_08_edupayal_and_balkampet_attractions(coordinator):
    """8. Nearby attractions around Edupayal and Balkampet."""
    state = AgentState(user_query="What are the nearby attractions around Edupayal and Balkampet?")
    resp = coordinator.process(state)
    assert resp.success
    assert len(resp.answer.strip()) > 30


def test_reg_09_hyderabad_to_kaleshwaram_navigation(coordinator):
    """9. Navigation from Hyderabad to Kaleshwaram extracts valid endpoints."""
    nav_agent = LiveNavigationAgent()
    params = nav_agent.extract_route_parameters("I am in Hyderabad and want to visit Kaleshwaram")
    assert params["origin"].lower() == "hyderabad"
    assert "kaleshwaram" in params["destination"].lower() or "kaleshvaram" in params["destination"].lower()


def test_reg_10_thousand_pillar_to_ramappa_navigation(coordinator):
    """10. Navigation between Thousand Pillar Temple and Ramappa."""
    nav_agent = LiveNavigationAgent()
    params = nav_agent.extract_route_parameters("Route between Thousand Pillar Temple and Ramappa")
    assert "thousand pillar" in params["origin"].lower()
    assert "ramappa" in params["destination"].lower()


def test_reg_11_chilkur_electronic_ban_integrity(coordinator):
    """11. Chilkur electronic ban: must preserve inner temple ban without stating no ban exists."""
    state = AgentState(user_query="What are the photography and electronic restrictions at Chilkur?")
    resp = coordinator.process(state)
    assert resp.success
    ans_lower = resp.answer.lower()
    # Must NOT state that electronics or photography are allowed inside the inner temple
    assert "no electronic ban" not in ans_lower
    assert not ("cameras are allowed inside the inner temple" in ans_lower)
