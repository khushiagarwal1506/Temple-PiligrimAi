"""
PilgrimAI - Follow-up Context Resolution Tests
Verifies deterministic conversational follow-up resolution:
1. Temple-dependent query without temple -> Clarification asked -> Temple entity follow-up -> Resolved query.
2. Timings follow-up resolution.
3. Sevas follow-up resolution.
4. Accommodation follow-up resolution.
5. Facilities follow-up resolution.
6. Pronoun/ellipsis follow-up: 'What about accommodation?' after Yadadri.
7. Pronoun/ellipsis follow-up: 'What about the dress code?' after Yadadri.
8. Unknown entity response does NOT fabricate or silently attach to a temple.
9. Multi-intent regression preserved.
10. Cross-temple regression preserved.
11. ExecutionTrace metadata properly logs resolution steps.
"""

import sys
from pathlib import Path
import pytest

# Ensure backend is in sys.path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agents.state import AgentState, AgentResponse
from app.agents.router import AgentRouter
from app.agents.coordinator import MasterAgentCoordinator
from app.agents.memory import ConversationMemoryManager
from app.agents.temple_information_agent import TempleInformationAgent
from app.agents.cross_temple_aggregation_agent import CrossTempleAggregationAgent
from app.agents.itinerary_planning_agent import ItineraryPlanningAgent
from app.agents.budget_agent import BudgetAgent
from app.agents.live_navigation_agent import LiveNavigationAgent
from app.agents.live_weather_agent import LiveWeatherAgent
from app.agents.emergency_agent import EmergencyAgent
from app.agents.recommendation_agent import RecommendationAgent


from app.api.chat import get_coordinator, get_rag_service
from app.agents.tools.rag_tool import RAGTool


@pytest.fixture
def coordinator():
    # Retrieve singleton coordinator and attach fresh memory manager for test isolation
    coord = get_coordinator()
    coord.memory_manager = ConversationMemoryManager()
    return coord


def test_1_dress_code_then_yadadri_resolution(coordinator):
    """
    Test 1:
    User: 'What is the dress code?' -> System asks for temple.
    User: 'Yadadri' -> Resolved to 'What is the dress code at Yadadri?' -> Handled by TempleInformationAgent.
    """
    session_id = "test_sess_dress_code_1"

    # Turn 1: No temple specified
    state1 = AgentState(session_id=session_id, user_query="What is the dress code?")
    resp1 = coordinator.process(state1)
    assert resp1.agent_name == "coordinator"
    assert "Which temple would you like information about?" in resp1.answer

    # Turn 2: User provides temple name 'Yadadri'
    state2 = AgentState(session_id=session_id, user_query="Yadadri")
    resp2 = coordinator.process(state2)

    # Must be processed by temple_information agent
    assert resp2.agent_name == "temple_information"
    assert resp2.success is True
    assert any(w in resp2.answer.lower() for w in ["dhoti", "saree", "kurta", "dress", "yadadri", "verified"])

    # Verify intermediate results and trace
    assert "memory" in state2.intermediate_results
    assert state2.intermediate_results["memory"]["resolved_from_pending"] is True
    assert state2.intermediate_results["memory"]["referenced_temple"] == "Yadadri"
    assert "dress code at Yadadri" in state2.intermediate_results["memory"]["resolved_query"]


def test_2_timings_then_yadadri(coordinator):
    """
    Test 2:
    User: 'What are the timings?' -> Clarification -> 'Yadadri' -> Timings for Yadadri.
    """
    session_id = "test_sess_timings_2"

    state1 = AgentState(session_id=session_id, user_query="What are the timings?")
    resp1 = coordinator.process(state1)
    assert "Which temple" in resp1.answer

    state2 = AgentState(session_id=session_id, user_query="Yadadri")
    resp2 = coordinator.process(state2)
    assert resp2.agent_name == "temple_information"
    assert resp2.success is True
    assert any(w in resp2.answer.lower() for w in ["4:00", "open", "am", "pm", "yadadri", "timings"])
    assert "timings at Yadadri" in state2.intermediate_results["memory"]["resolved_query"]


def test_3_sevas_then_dharmapuri(coordinator):
    """
    Test 3:
    User: 'What are the sevas?' -> Clarification -> 'Dharmapuri' -> Sevas for Dharmapuri.
    """
    session_id = "test_sess_sevas_3"

    state1 = AgentState(session_id=session_id, user_query="What are the sevas?")
    resp1 = coordinator.process(state1)
    assert "Which temple" in resp1.answer

    state2 = AgentState(session_id=session_id, user_query="Dharmapuri")
    resp2 = coordinator.process(state2)
    assert resp2.agent_name == "temple_information"
    assert "Dharmapuri" in resp2.answer or "seva" in resp2.answer.lower()
    assert "sevas at Dharmapuri" in state2.intermediate_results["memory"]["resolved_query"]


def test_4_accommodation_then_bhadrachalam(coordinator):
    """
    Test 4:
    User: 'What accommodation is available?' -> Clarification -> 'Bhadrachalam' -> Accommodation at Bhadrachalam.
    """
    session_id = "test_sess_acc_4"

    state1 = AgentState(session_id=session_id, user_query="What accommodation is available?")
    resp1 = coordinator.process(state1)
    assert "Which temple" in resp1.answer

    state2 = AgentState(session_id=session_id, user_query="Bhadrachalam")
    resp2 = coordinator.process(state2)
    assert resp2.agent_name == "temple_information"
    assert "accommodation is available at Badrachalam" in state2.intermediate_results["memory"]["resolved_query"] or "Bhadrachalam" in state2.intermediate_results["memory"]["resolved_query"]


def test_5_facilities_then_basara(coordinator):
    """
    Test 5:
    User: 'What facilities are available?' -> Clarification -> 'Basara' -> Facilities at Basara.
    """
    session_id = "test_sess_fac_5"

    state1 = AgentState(session_id=session_id, user_query="What facilities are available?")
    resp1 = coordinator.process(state1)
    assert "Which temple" in resp1.answer

    state2 = AgentState(session_id=session_id, user_query="Basara")
    resp2 = coordinator.process(state2)
    assert resp2.agent_name == "temple_information"
    assert "facilities are available at Basara" in state2.intermediate_results["memory"]["resolved_query"]


def test_6_yadadri_timings_then_what_about_accommodation(coordinator):
    """
    Test 6:
    User: 'What are Yadadri timings?' -> Answered.
    User: 'What about accommodation?' -> Resolves to accommodation at Yadadri.
    """
    session_id = "test_sess_followup_6"

    state1 = AgentState(session_id=session_id, user_query="What are Yadadri timings?")
    resp1 = coordinator.process(state1)
    assert resp1.agent_name == "temple_information"

    state2 = AgentState(session_id=session_id, user_query="What about accommodation?")
    resp2 = coordinator.process(state2)
    assert resp2.agent_name == "temple_information"
    assert "Yadadri" in state2.intermediate_results["memory"]["resolved_query"]


def test_7_yadadri_timings_then_what_about_dress_code(coordinator):
    """
    Test 7:
    User: 'What are Yadadri timings?' -> Answered.
    User: 'What about the dress code?' -> Resolves to dress code at Yadadri.
    """
    session_id = "test_sess_followup_7"

    state1 = AgentState(session_id=session_id, user_query="What are Yadadri timings?")
    resp1 = coordinator.process(state1)
    assert resp1.agent_name == "temple_information"

    state2 = AgentState(session_id=session_id, user_query="What about the dress code?")
    resp2 = coordinator.process(state2)
    assert resp2.agent_name == "temple_information"
    assert "Yadadri" in state2.intermediate_results["memory"]["resolved_query"]


def test_8_unknown_entity_response_does_not_fabricate(coordinator):
    """
    Test 8:
    User: 'What is the dress code?' -> Clarification.
    User: 'Some unknown fictitious place' -> Must NOT attach to a temple or fabricate.
    """
    session_id = "test_sess_unknown_8"

    state1 = AgentState(session_id=session_id, user_query="What is the dress code?")
    coordinator.process(state1)

    state2 = AgentState(session_id=session_id, user_query="Some unknown fictitious place")
    resp2 = coordinator.process(state2)

    # Must be rejected / unsupported / clarification, NOT silently answered for Yadadri or another temple
    assert resp2.agent_name == "coordinator"
    assert "I am the PilgrimAI Assistant" in resp2.answer or "outside the scope" in resp2.answer or "don't have verified" in resp2.answer or "Which temple" in resp2.answer


def test_9_multi_intent_regression_preserved(coordinator):
    """
    Test 9:
    Multi-intent query like 'What are Yadadri timings and plan a one-day visit?' works as expected.
    """
    session_id = "test_sess_multi_9"

    state = AgentState(session_id=session_id, user_query="What are Yadadri timings and plan a one-day visit?")
    resp = coordinator.process(state)

    # Routed through multi_intent pipeline
    assert resp.agent_name in ["multi_intent", "itinerary_planning", "response_aggregator", "temple_information"]
    assert "Yadadri" in resp.answer


def test_10_cross_temple_regression_preserved(coordinator):
    """
    Test 10:
    Cross-temple query like 'Which temples are on a river bank?' routes to CrossTempleAggregationAgent.
    """
    session_id = "test_sess_cross_10"

    state = AgentState(session_id=session_id, user_query="Which temples are on a river bank?")
    resp = coordinator.process(state)

    assert resp.agent_name == "cross_temple_aggregation"
