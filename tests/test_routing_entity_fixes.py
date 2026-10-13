import sys
from pathlib import Path

# Ensure backend is in sys.path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import pytest
from app.agents.router import AgentRouter
from app.agents.memory import ConversationMemoryManager
from app.agents.live_navigation_agent import LiveNavigationAgent
from app.agents.itinerary_planning_agent import ItineraryPlanningAgent
from app.agents.coordinator import MasterAgentCoordinator
from app.agents.temple_information_agent import TempleInformationAgent
from app.agents.comparison_agent import ComparisonAgent
from app.agents.live_weather_agent import LiveWeatherAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.state import AgentState


@pytest.fixture
def router():
    """Initializes router with all active agents."""
    r = AgentRouter()
    r.register_agent("temple_information")
    r.register_agent("comparison")
    r.register_agent("itinerary_planning")
    r.register_agent("live_weather")
    r.register_agent("live_navigation")
    r.register_agent("budget")
    r.register_agent("recommendation")
    return r


@pytest.fixture
def nav_agent():
    return LiveNavigationAgent()


@pytest.fixture
def memory_manager():
    return ConversationMemoryManager()


# ---------------------------------------------------------------------
# Requirement 1: BUG A & B - Itinerary Query Routing & Ujjaini Aliases
# ---------------------------------------------------------------------

def test_01_iternary_typo_with_ujjaini(router):
    """1. 'plan an iternary to go to ujjaini for 2 days' -> itinerary_planning, temple = Ujjaini"""
    decision = router.route("plan an iternary to go to ujjaini for 2 days")
    assert decision.selected_agent == "itinerary_planning"
    assert "Ujjaini" in decision.metadata.get("detected_temples", [])


def test_02_itinerary_with_ujjaini_mahankali(router):
    """2. 'plan an itinerary to go to ujjaini mahankali for 2 days' -> itinerary_planning, temple = Ujjaini"""
    decision = router.route("plan an itinerary to go to ujjaini mahankali for 2 days")
    assert decision.selected_agent == "itinerary_planning"
    assert "Ujjaini" in decision.metadata.get("detected_temples", [])


def test_03_plan_2_day_trip_to_ujjaini(router):
    """3. 'plan a 2 day trip to Ujjaini' -> itinerary_planning"""
    decision = router.route("plan a 2 day trip to Ujjaini")
    assert decision.selected_agent == "itinerary_planning"
    assert "Ujjaini" in decision.metadata.get("detected_temples", [])


def test_04_make_itinerary_for_yadadri(router):
    """4. 'make an itinerary for Yadadri' -> itinerary_planning"""
    decision = router.route("make an itinerary for Yadadri")
    assert decision.selected_agent == "itinerary_planning"
    assert "Yadadri" in decision.metadata.get("detected_temples", [])


def test_04b_itinerary_phrasing_variants(router):
    """Verify other itinerary variants and typos."""
    variants = [
        ("plan an itinerary for Basara", "Basara"),
        ("plan a iternary to Dharmapuri", "Dharmapuri"),
        ("plan a trip to Srisailam", "Srisaila"),
        ("create an itinerary for Yadadri", "Yadadri"),
        ("schedule a trip to Vemulavada", "Vemulavada"),
        ("2 day itinerary for Basara", "Basara"),
        ("3 day trip to Srisailam", "Srisaila"),
        ("visit Yadadri for 2 days", "Yadadri"),
        ("visit for 2 days at Yadadri", "Yadadri"),
    ]
    for query, expected_temple in variants:
        dec = router.route(query)
        assert dec.selected_agent == "itinerary_planning", f"Failed for: {query}"
        assert expected_temple in dec.metadata.get("detected_temples", []), f"Failed temple for: {query}"


def test_04c_ujjaini_alias_resolutions(router):
    """Verify all canonical aliases for Ujjaini resolve cleanly without being flagged unregistered."""
    aliases = [
        "ujjaini",
        "ujjaini mahakali",
        "ujjaini mahankali",
        "sri ujjaini mahakali",
        "sri ujjaini mahakali temple",
        "ujjaini mahakali temple",
        "ujjaini mahankali temple",
    ]
    for alias in aliases:
        query = f"what are the timings of {alias}"
        dec = router.route(query)
        assert dec.selected_agent == "temple_information", f"Failed for alias: {alias}"
        assert "Ujjaini" in dec.metadata.get("detected_temples", []), f"Failed to detect Ujjaini for alias: {alias}"


# ---------------------------------------------------------------------
# Requirement 2: BUG C - Navigation Entity Extraction
# ---------------------------------------------------------------------

def test_05_navigation_bus_routes_extraction(nav_agent):
    """5. 'i want to travel to vemulavada from hyderabad give me bus routes' -> origin=Hyderabad, destination=Vemulavada, mode=transit"""
    query = "i want to travel to vemulavada from hyderabad give me bus routes"
    params = nav_agent.extract_route_parameters(query)
    assert params["origin"].lower() == "hyderabad"
    assert params["destination"].lower() == "vemulavada"
    assert params["travel_mode"] == "transit"


def test_06_how_far_is_hyderabad_from_yadadri(nav_agent):
    """6. 'How far is Hyderabad from Yadadri?' -> origin=Yadadri, destination=Hyderabad"""
    query = "How far is Hyderabad from Yadadri?"
    params = nav_agent.extract_route_parameters(query)
    assert params["origin"].lower() == "yadadri"
    assert params["destination"].lower() == "hyderabad"
    assert params["travel_mode"] == "driving"


def test_07_how_far_is_yadadri_from_hyderabad(nav_agent):
    """7. 'How far is Yadadri from Hyderabad?' -> origin=Hyderabad, destination=Yadadri"""
    query = "How far is Yadadri from Hyderabad?"
    params = nav_agent.extract_route_parameters(query)
    assert params["origin"].lower() == "hyderabad"
    assert params["destination"].lower() == "yadadri"
    assert params["travel_mode"] == "driving"


def test_07b_navigation_phrasing_variants(nav_agent):
    """Verify various directional phrasing structures for navigation."""
    cases = [
        ("travel from Hyderabad to Vemulavada", "hyderabad", "vemulavada", "driving"),
        ("travel to Vemulavada from Hyderabad", "hyderabad", "vemulavada", "driving"),
        ("how do I get from Hyderabad to Vemulavada", "hyderabad", "vemulavada", "driving"),
        ("route from Hyderabad to Vemulavada", "hyderabad", "vemulavada", "driving"),
        ("bus from Hyderabad to Vemulavada", "hyderabad", "vemulavada", "transit"),
        ("give me bus routes from Hyderabad to Vemulavada", "hyderabad", "vemulavada", "transit"),
        ("Directions from Secunderabad to Basara", "secunderabad", "basara", "driving"),
        ("Distance between Basara and Dharmapuri", "basara", "dharmapuri", "driving"),
    ]
    for q, exp_orig, exp_dest, exp_mode in cases:
        p = nav_agent.extract_route_parameters(q)
        assert p["origin"].lower() == exp_orig, f"Origin mismatch for: {q}"
        assert p["destination"].lower() == exp_dest, f"Destination mismatch for: {q}"
        assert p["travel_mode"] == exp_mode, f"Mode mismatch for: {q}"


# ---------------------------------------------------------------------
# Requirement 3: BUG D - Follow-up Context & Memory Resolution
# ---------------------------------------------------------------------

def test_08_timings_of_yadadri(router):
    """8. 'what are the timings of Yadadri' -> temple_information, temple = Yadadri"""
    decision = router.route("what are the timings of Yadadri")
    assert decision.selected_agent == "temple_information"
    assert "Yadadri" in decision.metadata.get("detected_temples", [])


def test_09_followup_context_active_temple(memory_manager, router):
    """9. After active temple = Yadadri: 'what are the sevas' -> resolves to 'what are the sevas at Yadadri?'"""
    session_id = "test_session_followup_09"
    memory_manager.record_turn(
        session_id=session_id,
        user_query="what are the timings of Yadadri",
        resolved_query="what are the timings of Yadadri",
        detected_intents=["temple_information"],
        identified_temples=["Yadadri"],
        response_summary="Yadadri timings are 4:00 AM to 9:30 PM."
    )

    res = memory_manager.resolve_query(session_id, "what are the sevas")
    assert res["is_ambiguous"] is False
    assert res["referenced_temple"] == "Yadadri"
    assert "yadadri" in res["resolved_query"].lower()

    decision = router.route(res["resolved_query"])
    assert decision.selected_agent == "temple_information"
    assert "Yadadri" in decision.metadata.get("detected_temples", [])


def test_09b_pending_clarification_followup(memory_manager, router):
    """Clarification pattern:
    User: 'What is the dress code?' -> Clarification needed
    User: 'Yadadri' -> Resolves to 'What is the dress code at Yadadri?'
    """
    session_id = "test_session_clarify_09b"
    memory_manager.set_pending_clarification(
        session_id=session_id,
        pending_question="What is the dress code?",
        pending_intent="temple_information"
    )

    res = memory_manager.resolve_query(session_id, "Yadadri")
    assert res["is_ambiguous"] is False
    assert res["referenced_temple"] == "Yadadri"
    assert res["resolved_query"] == "What is the dress code at Yadadri?"

    decision = router.route(res["resolved_query"])
    assert decision.selected_agent == "temple_information"
    assert "Yadadri" in decision.metadata.get("detected_temples", [])


def test_09c_no_blind_reuse_for_unrelated(memory_manager):
    """Verify that unrelated follow-up queries do not blindly reuse the active temple."""
    session_id = "test_session_unrelated_09c"
    memory_manager.record_turn(
        session_id=session_id,
        user_query="what are the timings of Yadadri",
        resolved_query="what are the timings of Yadadri",
        detected_intents=["temple_information"],
        identified_temples=["Yadadri"],
        response_summary="Yadadri timings are 4:00 AM to 9:30 PM."
    )

    # Weather query
    res_weather = memory_manager.resolve_query(session_id, "what are the temples whose weather is below 30 degrees")
    assert res_weather["referenced_temple"] is None
    assert res_weather["resolved_query"] == "what are the temples whose weather is below 30 degrees"

    # Comparison query
    res_comp = memory_manager.resolve_query(session_id, "compare Vemulavada and Basara")
    assert res_comp["referenced_temple"] in ("Vemulavada", "Basara")  # explicit in query, overrides previous turn
    assert res_comp["resolved_query"] == "compare Vemulavada and Basara"


# ---------------------------------------------------------------------
# Requirement 4: Weather, Comparison, Recommendation & Edge Cases
# ---------------------------------------------------------------------

def test_10_weather_below_30_degrees(router):
    """10. 'what are the temples whose weather is below 30 degrees' -> live_weather"""
    decision = router.route("what are the temples whose weather is below 30 degrees")
    assert decision.selected_agent == "live_weather"
    assert decision.intent == "weather"


def test_10b_weather_below_40_degrees(router):
    """'tell the temples whose weather is below 40 degrees' -> live_weather"""
    decision = router.route("tell the temples whose weather is below 40 degrees")
    assert decision.selected_agent == "live_weather"
    assert decision.intent == "weather"


def test_11_compare_vemulavada_and_basara(router):
    """11. 'compare Vemulavada and Basara' -> comparison"""
    decision = router.route("compare Vemulavada and Basara")
    assert decision.selected_agent == "comparison"
    assert decision.intent == "comparison"
    assert set(decision.metadata.get("detected_temples", [])) == {"Vemulavada", "Basara"}


def test_12_nearby_attractions_tirumala(router):
    """12. 'give me nearby attractions for Tirumala' -> temple_information"""
    decision = router.route("give me nearby attractions for Tirumala")
    assert decision.selected_agent == "temple_information"
    assert "Ttd" in decision.metadata.get("detected_temples", [])


def test_13_recommend_a_temple_typo(router):
    """13. 'recomend a temple' -> recommendation"""
    decision = router.route("recomend a temple")
    assert decision.selected_agent == "recommendation"
    assert decision.intent == "recommendation"


def test_14_unknown_temple_itinerary(router):
    """14. 'plan a trip to completely unknown place' -> unsupported / no fabrication"""
    decision = router.route("plan a trip to completely unknown place")
    # Should not fabricate or hallucinate
    assert decision.metadata.get("detected_temples") == [] or decision.intent in ("unsupported", "clarification")


def test_15_itinerary_planning_execution_with_ujjaini():
    """Verify ItineraryPlanningAgent synthesizes a verified 2-day itinerary for Ujjaini without failing."""
    agent = ItineraryPlanningAgent()
    state = AgentState(user_query="plan an iternary to go to ujjaini for 2 days")
    response = agent.execute(state)
    assert response.success is True
    assert "Ujjaini" in response.answer
    assert "Day 1" in response.answer
    assert "Day 2" in response.answer
    assert response.metadata.get("duration_days") == 2
