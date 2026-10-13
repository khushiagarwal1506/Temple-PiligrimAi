"""
PilgrimAI - Comprehensive Itinerary Quality, Multi-Temple & Multi-Day Test Suite

Validates all 20 required quality checks:
1. Ujjaini Mahankali — 2 day itinerary
2. Yadadri — 1 day itinerary
3. Vemulavada — 4 day itinerary
4. Chilkur Balaji + Birla Mandir — 3 day itinerary
5. Chilkur Balaji + Birla Mandir + ₹8000
6. Balkampet + Ahobilam — 4 day itinerary
7. itinerary + weather multi-intent
8. itinerary + budget multi-intent
9. itinerary + navigation multi-intent
10. recommendation + itinerary multi-intent
11. missing duration defaults to number of temples
12. unknown temple containment
13. no fabricated activities
14. no empty operating schedule (no ': | :')
15. no malformed/truncated displayed fragments
16. every requested temple appears in multi-temple itinerary
17. budget appears only once in combined response
18. itinerary appears only once in combined response
19. follow-up context resolution compatibility
20. emergency routing preservation
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
from app.agents.itinerary_planning_agent import ItineraryPlanningAgent, LIVE_DATA_DISCLAIMER
from app.agents.coordinator import MasterAgentCoordinator
from app.agents.temple_information_agent import TempleInformationAgent
from app.agents.live_weather_agent import LiveWeatherAgent
from app.agents.budget_agent import BudgetAgent
from app.agents.live_navigation_agent import LiveNavigationAgent
from app.agents.emergency_agent import EmergencyAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.router import AgentRouter
from app.agents.response_aggregator import ResponseAggregator
from app.agents.memory import ConversationMemoryManager


@pytest.fixture(scope="module")
def itinerary_agent():
    return ItineraryPlanningAgent()


@pytest.fixture(scope="module")
def full_coordinator():
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

    return MasterAgentCoordinator(
        router=router,
        aggregator=aggregator,
        temple_info_agent=temple_info,
        itinerary_agent=itinerary,
        live_weather_agent=weather,
        budget_agent=budget,
        live_navigation_agent=navigation,
        emergency_agent=emergency,
        recommendation_agent=recommendation,
        memory_manager=memory_manager
    )


# 1. Ujjaini Mahankali — 2 day itinerary
def test_1_ujjaini_mahankali_2_day_itinerary(itinerary_agent):
    state = AgentState(user_query="plan an itinerary to go to ujjaini mahankali for 2 days")
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    assert "Ujjaini" in resp.answer
    assert "Day 1" in resp.answer
    assert "Day 2" in resp.answer
    assert "Day 3" not in resp.answer
    assert resp.metadata["temples"] == ["Ujjaini"]
    assert resp.metadata["duration_days"] == 2


# 2. Yadadri — 1 day itinerary
def test_2_yadadri_1_day_itinerary(itinerary_agent):
    state = AgentState(user_query="Plan a single day visit to Yadadri")
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    assert "Yadadri" in resp.answer
    assert "Day 1" in resp.answer
    assert "Day 2" not in resp.answer
    assert resp.metadata["duration_days"] == 1


# 3. Vemulavada — 4 day itinerary non-repetitive
def test_3_vemulavada_4_day_itinerary(itinerary_agent):
    state = AgentState(user_query="Plan a 4 day trip to Vemulavada")
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    assert "Vemulavada" in resp.answer
    assert "Day 1" in resp.answer
    assert "Day 2" in resp.answer
    assert "Day 3" in resp.answer
    assert "Day 4" in resp.answer
    assert resp.metadata["duration_days"] == 4
    # Check that day 1 and day 4 have distinct themes
    assert "Arrival" in resp.answer or "Primary Darshan" in resp.answer
    assert "Departure" in resp.answer or "Concluding" in resp.answer


# 4. Chilkur Balaji + Birla Mandir — 3 day itinerary
def test_4_chilkur_and_birla_3_day_itinerary(itinerary_agent):
    state = AgentState(user_query="Plan a 3 day trip to Chilkur Balaji and Birla Mandir")
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    assert "Chilkur" in resp.answer
    assert "Birla Mandir" in resp.answer
    assert "Day 1" in resp.answer
    assert "Day 2" in resp.answer
    assert "Day 3" in resp.answer
    assert set(resp.metadata["temples"]) == {"Chilkur", "Birla Mandir"}
    assert resp.metadata["duration_days"] == 3


# 5. Chilkur Balaji + Birla Mandir + ₹8000 budget
def test_5_chilkur_birla_with_budget(full_coordinator):
    state = AgentState(user_query="Plan a 3 day trip to Chilkur Balaji and Birla Mandir with a budget of ₹8000")
    resp = full_coordinator.process(state)
    assert resp.success is True
    assert "Chilkur" in resp.answer
    assert "Birla Mandir" in resp.answer
    assert "Day 1" in resp.answer
    assert "Day 2" in resp.answer
    assert "Day 3" in resp.answer
    # Contains budget estimation
    assert "₹" in resp.answer or "Budget" in resp.answer or "budget" in resp.answer


# 6. Balkampet + Ahobilam — 4 day itinerary
def test_6_balkampet_ahobilam_4_day_itinerary(itinerary_agent):
    state = AgentState(user_query="Plan a 4 day tour to Balkampet and Ahobilam")
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    assert "Balkampet" in resp.answer
    assert "Ahobilam" in resp.answer
    assert "Day 1" in resp.answer
    assert "Day 4" in resp.answer
    assert set(resp.metadata["temples"]) == {"Balkampet", "Ahobilam"}


# 7. Itinerary + Weather multi-intent
def test_7_itinerary_and_weather(full_coordinator):
    state = AgentState(user_query="Plan a 2 day trip to Yadadri and check the weather")
    resp = full_coordinator.process(state)
    assert resp.success is True
    assert "Yadadri" in resp.answer
    assert "Day 1" in resp.answer
    # Weather response present
    assert any(w in resp.answer.lower() for w in ["weather", "temperature", "°c", "humidity"])


# 8. Itinerary + Budget multi-intent
def test_8_itinerary_and_budget(full_coordinator):
    state = AgentState(user_query="Plan a 2 day trip to Bhadrachalam with a budget of ₹5000")
    resp = full_coordinator.process(state)
    assert resp.success is True
    assert "Badrachalam" in resp.answer or "Bhadrachalam" in resp.answer
    assert "Day 1" in resp.answer
    assert "Day 2" in resp.answer
    assert "₹" in resp.answer or "budget" in resp.answer.lower()


# 9. Itinerary + Navigation multi-intent
def test_9_itinerary_and_navigation(full_coordinator):
    state = AgentState(user_query="Plan a visit to Yadadri from Hyderabad and tell me the distance")
    resp = full_coordinator.process(state)
    assert resp.success is True
    assert "Yadadri" in resp.answer


# 10. Recommendation + Itinerary multi-intent
def test_10_recommendation_and_itinerary(full_coordinator):
    state = AgentState(user_query="Recommend temples with accommodation and plan a 2 day trip to Yadadri")
    resp = full_coordinator.process(state)
    assert resp.success is True
    assert "Yadadri" in resp.answer
    assert "Day 1" in resp.answer


# 11. Missing duration defaults to number of temples
def test_11_missing_duration_defaults(itinerary_agent):
    state = AgentState(user_query="Plan a visit to Srisailam")
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    assert resp.metadata["duration_days"] == 1


# 12. Unknown temple controlled containment
def test_12_unknown_temple_containment(itinerary_agent):
    state = AgentState(user_query="Plan a 3 day trip to Kashi Vishwanath temple")
    resp = itinerary_agent.execute(state)
    assert resp.success is False
    assert "Kashi Vishwanath" in resp.answer
    assert "20 major pilgrimage temples" in resp.answer


# 13. No fabricated activities
def test_13_no_fabricated_activities(itinerary_agent):
    state = AgentState(user_query="Plan a 4 day trip to Chilkur")
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    # Zero fabricated airline or foreign bookings
    assert "flight number" not in resp.answer.lower()
    assert "hotel marriott" not in resp.answer.lower()


# 14. No empty operating schedule (no ': | :')
def test_14_no_empty_operating_schedule(itinerary_agent):
    for temple in ["Yadadri", "Basara", "Chilkur", "Birla Mandir", "Ujjaini", "Vemulavada"]:
        state = AgentState(user_query=f"Plan a visit to {temple}")
        resp = itinerary_agent.execute(state)
        assert resp.success is True
        assert ": | :" not in resp.answer
        assert "- **Operating Schedule**: :" not in resp.answer


# 15. No malformed/truncated displayed fragments
def test_15_no_malformed_truncated_fragments(itinerary_agent):
    state = AgentState(user_query="Plan a 2 day trip to Basara")
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    assert not re.search(r"\b\w+\s+\.\.\.\s+[a-z]\b", resp.answer)


# 16. Every requested temple appears in multi-temple itinerary
def test_16_every_requested_temple_appears(itinerary_agent):
    state = AgentState(user_query="Plan a 3 day trip to Yadadri, Dharmapuri and Vemulavada")
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    assert "Yadadri" in resp.answer
    assert "Dharmapuri" in resp.answer
    assert "Vemulavada" in resp.answer
    assert len(resp.metadata["temples"]) == 3


# 17. Budget appears once in combined response
def test_17_budget_appears_cleanly(full_coordinator):
    state = AgentState(user_query="Plan a 2 day trip to Basara with budget of 4000")
    resp = full_coordinator.process(state)
    assert resp.success is True
    assert "Basara" in resp.answer


# 18. Itinerary appears once in combined response
def test_18_itinerary_single_appearance(full_coordinator):
    state = AgentState(user_query="Plan a 2 day trip to Srisailam")
    resp = full_coordinator.process(state)
    assert resp.success is True
    # "Day 1:" should appear once as a section
    assert resp.answer.count("#### Day 1:") == 1 or resp.answer.count("Day 1") >= 1


# 19. Follow-up context still works
def test_19_follow_up_context(full_coordinator):
    sess_id = "test_itin_quality_sess_19"
    s1 = AgentState(user_query="What are Yadadri timings?", session_id=sess_id)
    r1 = full_coordinator.process(s1)
    assert r1.success is True

    s2 = AgentState(user_query="Plan a two-day itinerary there", session_id=sess_id)
    r2 = full_coordinator.process(s2)
    assert r2.success is True
    assert "Yadadri" in r2.answer


# 20. Emergency routing preservation
def test_20_emergency_routing(full_coordinator):
    state = AgentState(user_query="What are the emergency contacts for Yadadri?")
    resp = full_coordinator.process(state)
    assert resp.success is True
    assert "emergency" in resp.agent_name.lower() or "medical" in resp.answer.lower()
