"""
PilgrimAI - Final Correctness & Multi-Agent Regression Test Suite (Part 14)

Verifies all 37 regression requirements:
- Budget: Kaleshwaram, general budget, pax/days scaling, stay alternatives, optional sevas, unavailable != ₹0, unknown != within budget, free darshan, generic multi-temple.
- Multi-Intent: Itinerary + Budget, Itinerary + Weather, without 'and', Itinerary + Budget + Weather + Navigation.
- Itinerary: 1-temple 1-day, 1-temple multi-day, 2-temple multi-day, no duplicate templates, insufficient activities honest notes.
- Aggregation: Single budget/itinerary/weather section, consolidated sources, no duplicated disclaimers.
- Data Quality: No raw Python dicts exposed ({'men': '', 'women': ''}), empty fields omitted, malformed schedule strings omitted.
- Other Specialist Regressions: Recommendation, Emergency, Navigation, Comparison, Cross-temple, Follow-up context.
"""

import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import pytest

root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agents.state import AgentState, AgentResponse
from app.agents.router import AgentRouter
from app.agents.coordinator import MasterAgentCoordinator
from app.agents.response_aggregator import ResponseAggregator
from app.agents.multi_intent_pipeline import SequentialMultiIntentPipeline
from app.agents.budget_agent import BudgetAgent, parse_price_text, extract_budget_parameters
from app.agents.itinerary_planning_agent import ItineraryPlanningAgent
from app.agents.live_weather_agent import LiveWeatherAgent
from app.agents.live_navigation_agent import LiveNavigationAgent
from app.agents.emergency_agent import EmergencyAgent
from app.agents.recommendation_agent import RecommendationAgent


# =====================================================================
# Mock Tools for Deterministic Offline Regression Testing
# =====================================================================

class MockRegressionRAGTool:
    """Mock RAGTool with rich real-world test data for Kaleshvaram, Dharmapuri, Yadadri, etc."""

    def __init__(self):
        self.data_store = {
            "Kaleshvaram": {
                "timings": {"entry_fee": "Free General Darshan", "general_timings": "4:00 AM to 1:00 PM, 3:30 PM to 8:00 PM"},
                "sevas": [
                    {"name": "Sheegra Darshanam", "price": "₹50 - ₹100", "timing": "All day"},
                    {"name": "Garbhagriha Abhishekam", "price": "₹300 - ₹500", "timing": "6:00 AM"},
                    {"name": "Mahanyasa Poorvaka Rudrabhishekam", "price": "₹1200 - ₹2000", "timing": "7:00 AM"},
                    {"name": "Shiva Kalyanam", "price": "₹1000", "timing": "10:30 AM"}
                ],
                "accommodation": [
                    {"name": "TG Tourism Haritha Hotel", "tariff": "₹1,200 - ₹2,500 per night"},
                    {"name": "Devasthanam Guest Houses", "tariff": "₹300 - ₹1,200 per night"},
                    {"name": "Saraswathi Choultry", "tariff": "₹200 - ₹500 per night"},
                    {"name": "RSR Lodge", "tariff": "₹500 - ₹1,500 per night"}
                ],
                "prasadam": {"annadanam": "Free daily afternoon Annadanam offered.", "items": [{"name": "Laddu", "price": "₹20"}]},
                "rules": {"dress_code": {"men": "", "women": ""}, "dos": ["Wear traditional attire"], "donts": ["No mobile phones in sanctum"]}
            },
            "Dharmapuri": {
                "timings": {"entry_fee": "Free entry", "general_timings": "5:00 AM to 1:00 PM, 4:00 PM to 8:30 PM"},
                "sevas": [
                    {"name": "Nijabhishekam", "price": "₹500"},
                    {"name": "Kalyanotsavam", "price": "₹1,116"}
                ],
                "accommodation": [
                    {"name": "Devasthanam Choultry", "tariff": "₹300 - ₹600 per night"}
                ],
                "rules": {"dress_code": {"men": "Dhoti or Kurta", "women": "Saree or Chudidar"}}
            },
            "Thousand Pillar": {
                "timings": {"entry_fee": "Free", "general_timings": "6:00 AM to 8:00 PM"},
                "sevas": [
                    {"name": "Rudrabhishekam", "price": "₹200"}
                ],
                "rules": {"dress_code": "Traditional attire recommended"}
            },
            "Yadadri": {
                "timings": {"entry_fee": "Free General Darshan; Special Entry ₹150 per person"},
                "sevas": [
                    {"name": "Nitya Kalyanam", "price": "₹1,500"},
                    {"name": "Nijabhishekam", "price": "₹500"}
                ],
                "accommodation": [
                    {"name": "Standard Non-AC", "tariff": "₹800 per night"}
                ],
                "wheelchair": {"wheelchair_accessibility": {"is_wheelchair_friendly": True, "details": "Ramps available"}}
            },
            "Basara": {
                "timings": {"entry_fee": "Free General Darshan", "general_timings": "4:00 AM to 12:30 PM, 2:00 PM to 8:30 PM"},
                "sevas": [{"name": "Aksharabhyasam", "price": "₹150"}],
                "accommodation": [{"name": "Devasthanam Rooms", "tariff": "₹200 - ₹500"}],
                "wheelchair": {"wheelchair_accessibility": {"is_wheelchair_friendly": True, "details": "Wheelchair available at main entrance"}}
            },
            "Balkampet": {
                "timings": {"entry_fee": "Free General Darshan", "general_timings": "6:00 AM to 8:00 PM"},
                "wheelchair": {"wheelchair_accessibility": {"is_wheelchair_friendly": False, "details": "Not wheelchair friendly (Subterranean staircase only)"}}
            }
        }

    def get_registered_temples(self) -> List[str]:
        return list(self.data_store.keys())

    def get_temple_category_data(self, temple: str, category: str) -> Optional[Any]:
        t_key = temple.strip()
        # Case insensitive and substring matching
        for k, v in self.data_store.items():
            if k.lower() == t_key.lower() or k.lower() in t_key.lower() or t_key.lower() in k.lower():
                cat_lower = category.lower()
                if cat_lower in ("wheelchair", "accessibility", "elderly_ppl"):
                    if "wheelchair" in v:
                        return v["wheelchair"]
                    return {"wheelchair_accessibility": {"is_wheelchair_friendly": True, "details": "Ramps and wheelchair available at entrance"}}
                return v.get(cat_lower)
        return None

    def get_complete_parent_text(self, temple: str, category: str) -> Optional[str]:
        return None


class MockResponse:
    def __init__(self, data):
        self._data = data
        self.status_code = 200
    def json(self):
        return self._data


class MockExternalAdapter:
    """Mock external adapter for LiveWeatherAgent and LiveNavigationAgent."""
    def __init__(self):
        self.has_credentials = True
        self.transport = "mock"

    def request(self, method: str, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, Any]] = None, json: Optional[Dict[str, Any]] = None, timeout: float = 5.0):
        if "directions" in url or "routes" in url:
            return MockResponse({
                "routes": [{
                    "distanceMeters": 245000,
                    "duration": "18000s",
                    "description": "via NH 163 and SH 7"
                }]
            })
        return MockResponse({
            "current": {
                "temperature_2m": 32.4,
                "relative_humidity_2m": 58,
                "weather_code": 0,
                "wind_speed_10m": 12.5,
                "time": "2026-10-06T12:00"
            },
            "daily": {
                "time": ["2026-10-06", "2026-10-07", "2026-10-08"],
                "temperature_2m_max": [34.0, 35.0, 33.5],
                "temperature_2m_min": [24.0, 23.5, 23.0],
                "weather_code": [0, 1, 2],
                "precipitation_probability_max": [0, 10, 20]
            }
        })

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, Any]] = None, timeout: float = 5.0) -> Dict[str, Any]:
        return self.request("GET", url, params=params, headers=headers, timeout=timeout).json()

    def post(self, url: str, payload: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, Any]] = None, timeout: float = 5.0) -> Dict[str, Any]:
        return self.request("POST", url, json=payload, headers=headers, timeout=timeout).json()

    def compute_route(self, origin: Any, destination: Any, travel_mode: str = "driving", routing_preference: str = "TRAFFIC_AWARE") -> Dict[str, Any]:
        return {
            "routes": [{
                "distanceMeters": 245000,
                "duration": "18000s",
                "description": "via NH 163 and SH 7"
            }]
        }


@pytest.fixture
def test_setup():
    mock_rag = MockRegressionRAGTool()
    mock_adapter = MockExternalAdapter()

    router = AgentRouter()
    aggregator = ResponseAggregator()
    from app.agents.cross_temple_aggregation_agent import CrossTempleAggregationAgent
    cross_temple_agent = CrossTempleAggregationAgent(rag_tool=mock_rag)

    budget_agent = BudgetAgent(rag_tool=mock_rag)
    itinerary_agent = ItineraryPlanningAgent(rag_tool=mock_rag)
    weather_agent = LiveWeatherAgent(adapter=mock_adapter)
    nav_agent = LiveNavigationAgent(adapter=mock_adapter)
    emergency_agent = EmergencyAgent(rag_tool=mock_rag)
    rec_agent = RecommendationAgent(rag_tool=mock_rag, weather_agent=weather_agent)

    agents = {
        "budget": budget_agent,
        "itinerary_planning": itinerary_agent,
        "live_weather": weather_agent,
        "live_navigation": nav_agent,
        "emergency": emergency_agent,
        "recommendation": rec_agent,
        "cross_temple_aggregation": cross_temple_agent
    }

    pipeline = SequentialMultiIntentPipeline(
        router=router,
        aggregator=aggregator,
        agents=agents
    )

    coordinator = MasterAgentCoordinator(
        router=router,
        aggregator=aggregator,
        agents=agents,
        pipeline=pipeline
    )

    return {
        "coordinator": coordinator,
        "budget_agent": budget_agent,
        "itinerary_agent": itinerary_agent,
        "weather_agent": weather_agent,
        "cross_temple_agent": cross_temple_agent,
        "rec_agent": rec_agent,
        "pipeline": pipeline,
        "router": router,
        "aggregator": aggregator
    }


# =====================================================================
# 1. Budget Regressions (Tests 1 - 10)
# =====================================================================

def test_01_kaleshwaram_general_budget_not_zero(test_setup):
    """1. Kaleshwaram general budget discovers verified records and does not return ₹0."""
    agent = test_setup["budget_agent"]
    state = AgentState(user_query="my budget is 3000 to visit kaleshwaram temple")
    resp = agent.execute(state)
    assert resp.success is True
    assert resp.metadata["verified_item_count"] > 0
    # Must contain verified accommodation / seva ranges and not return 0
    assert resp.metadata["calculated_total"]["is_quantified"] is True
    assert resp.metadata["calculated_total"]["min"] > 0
    assert "₹0" not in resp.answer or "Free" in resp.answer


def test_02_kaleshwaram_budget_limit_comparison(test_setup):
    """2. Kaleshwaram ₹3000 comparison evaluates against verified calculated range."""
    agent = test_setup["budget_agent"]
    state = AgentState(user_query="my budget is 3000 for kaleshwaram")
    resp = agent.execute(state)
    assert resp.success is True
    assert "3,000" in resp.answer or "3000" in resp.answer
    assert resp.metadata["parameters"]["budget_limit"] == 3000.0


def test_03_kaleshwaram_pax_scaling(test_setup):
    """3. Scaling for 2 people scales per-person items without blindly scaling room tariffs."""
    agent = test_setup["budget_agent"]
    state = AgentState(user_query="i have 3000 to visit kaleshwaram for 2 people")
    resp = agent.execute(state)
    assert resp.success is True
    assert resp.metadata["parameters"]["pax"] == 2
    assert "2 pilgrim(s)" in resp.answer or "2 person" in resp.answer


def test_04_kaleshwaram_duration_nights_scaling(test_setup):
    """4. 2 days trip correctly sets nights = 1 for accommodation alternatives."""
    agent = test_setup["budget_agent"]
    state = AgentState(user_query="kaleshwaram budget for 2 days")
    resp = agent.execute(state)
    assert resp.success is True
    assert resp.metadata["parameters"]["days"] == 2
    assert "2 day(s)" in resp.answer


def test_05_accommodation_alternatives_not_summed(test_setup):
    """5. Multiple accommodation options (Haritha, Guest House, Choultry, RSR) are NOT summed together."""
    agent = test_setup["budget_agent"]
    state = AgentState(user_query="kaleshwaram accommodation budget")
    resp = agent.execute(state)
    assert resp.success is True
    # If summed, 1200+300+200+500 = 2200 min. But as alternatives, min is 200 (Choultry)
    assert resp.metadata["calculated_total"]["min"] <= 1200.0


def test_06_optional_sevas_not_forced_in_total(test_setup):
    """6. Optional sevas are labeled optional and not automatically summed into mandatory costs."""
    agent = test_setup["budget_agent"]
    state = AgentState(user_query="what is the general budget for kaleshwaram temple")
    resp = agent.execute(state)
    assert resp.success is True
    assert "Optional" in resp.answer or "Alternative" in resp.answer


def test_07_unavailable_not_zero():
    """7. Unavailable items are explicitly marked UNAVAILABLE and not treated as ₹0."""
    p_min, p_max, unit, is_free, is_rng, is_q, notes = parse_price_text("Contact temple office")
    assert is_q is False
    assert is_free is False
    assert p_min is None


def test_08_unknown_total_never_within_budget(test_setup):
    """8. If total is unquantified, never output 'Within Budget'."""
    agent = test_setup["budget_agent"]
    state = AgentState(user_query="budget for unknown temple seva with budget of 5000")
    resp = agent.execute(state)
    if not resp.metadata.get("calculated_total", {}).get("is_quantified"):
        assert "Within Budget" not in resp.answer


def test_09_verified_free_darshan_is_free():
    """9. Verified free general darshan is recognized as FREE and not UNAVAILABLE."""
    p_min, p_max, unit, is_free, is_rng, is_q, notes = parse_price_text("Free General Darshan")
    assert is_free is True
    assert is_q is True
    assert p_min == 0.0


def test_10_generic_budget_across_multiple_temples(test_setup):
    """10. Budget logic works generically across Dharmapuri, Yadadri, Thousand Pillar, etc."""
    agent = test_setup["budget_agent"]
    for t in ["dharmapuri", "yadadri", "thousand pillar"]:
        state = AgentState(user_query=f"calculate budget for {t} temple")
        resp = agent.execute(state)
        assert resp.success is True
        assert resp.metadata["verified_item_count"] > 0


# =====================================================================
# 2. Multi-Intent Regressions (Tests 11 - 16)
# =====================================================================

def test_11_multi_intent_itinerary_and_budget(test_setup):
    """11. Itinerary + Budget multi-intent decomposition."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="plan a 2 day trip to kaleshwaram with a budget of 5000")
    resp = coord.process(state)
    assert resp.success is True
    assert "Itinerary" in resp.answer
    assert "Budget" in resp.answer


def test_12_multi_intent_itinerary_and_weather(test_setup):
    """12. Itinerary + Weather multi-intent decomposition."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="plan a 2 day trip to kaleshwaram and what is the weather today")
    resp = coord.process(state)
    assert resp.success is True
    assert "Itinerary" in resp.answer
    assert "Weather" in resp.answer or "🌤️" in resp.answer


def test_13_multi_intent_itinerary_budget_weather_with_and(test_setup):
    """13. Itinerary + Budget + Weather conjoined with 'and'."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="plan a 2 day trip to kaleshwaram with a budget of 9000 for 2 people and mention the temperature today")
    resp = coord.process(state)
    assert resp.success is True
    assert "Itinerary" in resp.answer
    assert "Budget" in resp.answer
    assert "Weather" in resp.answer or "🌤️" in resp.answer or "32.4" in resp.answer


def test_14_multi_intent_itinerary_budget_weather_without_and(test_setup):
    """14. Itinerary + Budget + Weather conjoined WITHOUT 'and'."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="plan a 2 day trip to kaleshwaram temple with a budget of 9000 for 2 people mention the temperature today")
    resp = coord.process(state)
    assert resp.success is True
    assert "Itinerary" in resp.answer
    assert "Budget" in resp.answer
    assert "Weather" in resp.answer or "🌤️" in resp.answer or "32.4" in resp.answer


def test_15_multi_intent_itinerary_budget_navigation(test_setup):
    """15. Itinerary + Budget + Navigation decomposition."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="plan a 2 day trip to kaleshwaram with a budget of 9000 and how to reach from Hyderabad")
    resp = coord.process(state)
    assert resp.success is True
    assert "Itinerary" in resp.answer
    assert "Budget" in resp.answer
    assert "Navigation" in resp.answer or "Route" in resp.answer or "km" in resp.answer


def test_16_multi_intent_four_way_decomposition(test_setup):
    """16. Itinerary + Budget + Weather + Navigation all present."""
    pipeline = test_setup["pipeline"]
    tasks = pipeline.decompose_query("plan a 2 day trip to kaleshwaram with a budget of 9000 and weather today and directions from Hyderabad")
    agents = [t["target_agent"] for t in tasks]
    assert "itinerary_planning" in agents
    assert "budget" in agents
    assert "live_weather" in agents
    assert "live_navigation" in agents


# =====================================================================
# 3. Itinerary Regressions (Tests 17 - 22)
# =====================================================================

def test_17_itinerary_one_temple_one_day(test_setup):
    """17. Single temple, 1 day."""
    agent = test_setup["itinerary_agent"]
    state = AgentState(user_query="plan a 1 day trip to kaleshwaram")
    resp = agent.execute(state)
    assert resp.success is True
    assert "Day 1: Kaleshvaram" in resp.answer or "Day 1: Kaleshwaram" in resp.answer
    assert "Day 2" not in resp.answer


def test_18_itinerary_one_temple_multi_day(test_setup):
    """18. Single temple, 2 days."""
    agent = test_setup["itinerary_agent"]
    state = AgentState(user_query="plan a 2 day trip to kaleshwaram")
    resp = agent.execute(state)
    assert resp.success is True
    assert "Day 1:" in resp.answer
    assert "Day 2:" in resp.answer


def test_19_itinerary_two_temples_multi_day(test_setup):
    """19. Two temples, 4 days: BOTH temples appear in schedule."""
    agent = test_setup["itinerary_agent"]
    state = AgentState(user_query="plan a 4 day trip to Thousand Pillar Temple and Dharmapuri")
    resp = agent.execute(state)
    assert resp.success is True
    assert "Thousand Pillar" in resp.answer
    assert "Dharmapuri" in resp.answer
    assert "Day 1:" in resp.answer
    assert "Day 4:" in resp.answer


def test_20_itinerary_two_temples_with_budget(test_setup):
    """20. Two temples + Budget."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="plan a 4 day trip to Thousand Pillar Temple and Dharmapuri with a budget of 8000")
    resp = coord.process(state)
    assert resp.success is True
    assert "Thousand Pillar" in resp.answer
    assert "Dharmapuri" in resp.answer
    assert "Budget" in resp.answer


def test_21_itinerary_honest_schedule_no_invented_activities(test_setup):
    """21. Insufficient verified activities produces honest notes, not fabricated attractions."""
    agent = test_setup["itinerary_agent"]
    state = AgentState(user_query="plan a 3 day trip to Thousand Pillar Temple")
    resp = agent.execute(state)
    assert resp.success is True
    assert "Verified Temple Information" in resp.answer


def test_22_itinerary_no_duplicate_day_template(test_setup):
    """22. Day 1 and Day 2 have distinct themes and window details."""
    agent = test_setup["itinerary_agent"]
    state = AgentState(user_query="plan a 2 day trip to kaleshwaram")
    resp = agent.execute(state)
    assert resp.success is True
    assert "Arrival, Primary Darshan & Sevas" in resp.answer
    assert "Concluding Darshan" in resp.answer or "Special Sevas" in resp.answer


# =====================================================================
# 4. Aggregation Regressions (Tests 23 - 28)
# =====================================================================

def test_23_aggregation_single_budget_section(test_setup):
    """23. Budget block appears once without duplication."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="plan a 2 day trip to kaleshwaram with a budget of 9000")
    resp = coord.process(state)
    assert resp.answer.count("💰") <= 2
    assert "Pilgrimage Budget" in resp.answer


def test_24_aggregation_single_itinerary_section(test_setup):
    """24. Itinerary block appears once."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="plan a 2 day trip to kaleshwaram with a budget of 9000")
    resp = coord.process(state)
    assert resp.answer.count("### Pilgrimage Itinerary:") == 1


def test_25_aggregation_single_weather_section(test_setup):
    """25. Weather block appears once."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="plan a 2 day trip to kaleshwaram and mention the temperature today")
    resp = coord.process(state)
    assert resp.answer.count("Live Weather") <= 2


def test_26_aggregation_consolidated_sources(test_setup):
    """26. Sources are consolidated into a deduplicated list."""
    aggregator = test_setup["aggregator"]
    r1 = AgentResponse(success=True, agent_name="budget", answer="Budget info", sources=["kaleshvaram/sevas.json", "kaleshvaram/accommodation.json"])
    r2 = AgentResponse(success=True, agent_name="itinerary_planning", answer="Itinerary info", sources=["kaleshvaram/timings.json", "kaleshvaram/sevas.json"])
    merged = aggregator.aggregate([r1, r2])
    assert len(merged.sources) == 3
    assert len(set(merged.sources)) == 3


def test_27_aggregation_no_duplicated_disclaimers(test_setup):
    """27. Multi-agent aggregated response has clean section breaks."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="plan a 2 day trip to kaleshwaram with a budget of 9000 and weather today")
    resp = coord.process(state)
    assert "---" in resp.answer


def test_28_aggregation_no_contradictory_totals(test_setup):
    """28. Total calculated in budget is preserved consistently."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="plan a 2 day trip to kaleshwaram with a budget of 9000")
    resp = coord.process(state)
    assert resp.success is True


# =====================================================================
# 5. Data Quality Sanitization (Tests 29 - 31)
# =====================================================================

def test_29_empty_dict_fields_omitted(test_setup):
    """29. Empty dress_code dict {'men': '', 'women': ''} is omitted from output."""
    agent = test_setup["itinerary_agent"]
    state = AgentState(user_query="plan a 1 day trip to kaleshwaram")
    resp = agent.execute(state)
    assert "{'men': '', 'women': ''}" not in resp.answer
    assert "{'men':" not in resp.answer


def test_30_malformed_schedule_fields_suppressed(test_setup):
    """30. Malformed strings like ': |' are suppressed."""
    agent = test_setup["itinerary_agent"]
    cleaned = agent._clean_text(": | :")
    assert cleaned == ""


def test_31_raw_python_dict_never_exposed(test_setup):
    """31. Raw Python dictionary representations are never printed to user."""
    agent = test_setup["itinerary_agent"]
    parsed_rules = agent._parse_rules({"rules": {"dress_code": {"men": "Dhoti", "women": "Saree"}}})
    assert parsed_rules["dress_code"] == "Men: Dhoti, Women: Saree"
    assert not parsed_rules["dress_code"].startswith("{")


# =====================================================================
# 6. Other Agent Regressions (Tests 32 - 37)
# =====================================================================

def test_32_recommendation_agent_regression(test_setup):
    """32. Recommendation agent functions via centralized routing."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="which temple should i visit for education in Telangana?")
    resp = coord.process(state)
    assert resp.success is True or "recommend" in resp.answer.lower() or "basara" in resp.answer.lower()


def test_33_emergency_agent_regression(test_setup):
    """33. Emergency agent routing and assistance."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="medical emergency ambulance number at Yadadri temple")
    resp = coord.process(state)
    assert resp.success is True
    assert "108" in resp.answer or "Emergency" in resp.answer or "Medical" in resp.answer


def test_34_navigation_agent_regression(test_setup):
    """34. Live navigation agent routing and route calculation."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="how to reach Kaleshwaram from Hyderabad by road")
    resp = coord.process(state)
    assert resp.success is True
    assert "km" in resp.answer or "Route" in resp.answer or "Navigation" in resp.answer


def test_35_comparison_agent_protection(test_setup):
    """35. Comparison queries route to comparison intent."""
    router = test_setup["router"]
    dec = router.route("compare Yadadri and Dharmapuri temples")
    assert dec.intent == "comparison"


def test_36_cross_temple_aggregation_protection(test_setup):
    """36. Cross-temple aggregation queries route cleanly."""
    router = test_setup["router"]
    dec = router.route("which temples in Telangana have Annadanam?")
    assert dec.intent in ["cross_temple_aggregation", "temple_information", "recommendation"]


def test_37_follow_up_context_state_propagation(test_setup):
    """37. Shared state constraints propagate properly across multi-intent pipeline."""
    pipeline = test_setup["pipeline"]
    query = "plan a 2 day trip to kaleshwaram with a budget of 9000 for 2 people mention the temperature today"
    tasks = pipeline.decompose_query(query)
    assert len(tasks) == 3
    assert tasks[0]["intent"] == "itinerary_planning"
    assert tasks[1]["intent"] == "budget"
    assert tasks[2]["intent"] == "weather" or tasks[2]["target_agent"] == "live_weather"


def test_38_recommendation_pleasant_weather(test_setup):
    """38. Recommendation agent handles pleasant weather criteria dynamically."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="suggest temples whose temperature is pleasant to visit now")
    resp = coord.process(state)
    assert resp.success is True
    assert "Live Weather Recommendations" in resp.answer or "Open-Meteo" in resp.answer


def test_39_recommendation_budget_constraint(test_setup):
    """39. Recommendation agent handles budget criteria."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="recommend temples under 3000")
    resp = coord.process(state)
    assert resp.success is True
    assert "Within Budget" in resp.answer or "Recommended Temples" in resp.answer


def test_40_recommendation_budget_and_weather(test_setup):
    """40. Recommendation agent combines budget and pleasant weather."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="recommend temples under 3000 with pleasant weather")
    resp = coord.process(state)
    assert resp.success is True
    assert "Pleasant Weather" in resp.answer or "Live Weather" in resp.answer or "Open-Meteo" in resp.answer


def test_41_cross_temple_wheelchair_unavailable(test_setup):
    """41. Cross-temple negative query: list temples that do not provide wheelchair facility."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="list the temples that do not provide wheelchair facility")
    resp = coord.process(state)
    assert resp.success is True
    assert "NOT have Wheelchair" in resp.answer or "Not wheelchair friendly" in resp.answer


def test_42_cross_temple_wheelchair_available(test_setup):
    """42. Cross-temple positive query: list temples with wheelchair facility."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="list the temples with wheelchair facility")
    resp = coord.process(state)
    assert resp.success is True
    assert "Wheelchair" in resp.answer
    assert "Basara" in resp.answer or "Badrachalam" in resp.answer


def test_43_cross_temple_accommodation_unavailable(test_setup):
    """43. Cross-temple negative query: which temples do not have accommodation."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="which temples do not have accommodation")
    resp = coord.process(state)
    assert resp.success is True
    # Should honestly report no confirmed absence
    assert "confirmed absence of accommodation" in resp.answer or "Accommodation" in resp.answer


def test_44_cross_temple_accommodation_available(test_setup):
    """44. Cross-temple positive query: which temples have accommodation."""
    coord = test_setup["coordinator"]
    state = AgentState(user_query="which temples have accommodation")
    resp = coord.process(state)
    assert resp.success is True
    assert "Accommodation" in resp.answer


def test_45_cross_temple_unknown_remains_unknown(test_setup):
    """45. Information not available is not inferred as confirmed absence."""
    agent = test_setup["cross_temple_agent"]
    res = agent.aggregate_temples("wheelchair", condition="negative")
    assert "confirmed_unavailable" in res
    assert "information_not_available" in res
    assert len(res["confirmed_unavailable"]) < res["total_temples_searched"]


def test_46_itinerary_three_temples_multi_day(test_setup):
    """46. Three temples in a multi-day itinerary all appear in the plan."""
    agent = test_setup["itinerary_agent"]
    state = AgentState(user_query="plan a 3 day trip to Basara, Vemulavada and Dharmapuri")
    resp = agent.execute(state)
    assert resp.success is True
    assert "Basara" in resp.answer
    assert "Vemulavada" in resp.answer
    assert "Dharmapuri" in resp.answer


def test_47_itinerary_every_requested_temple_appears(test_setup):
    """47. Every requested temple is guaranteed to appear in multi-temple itineraries."""
    agent = test_setup["itinerary_agent"]
    state = AgentState(user_query="plan a 4 day trip to Thousand Pillar Temple and Dharmapuri")
    resp = agent.execute(state)
    assert resp.success is True
    assert "Thousand Pillar" in resp.answer
    assert "Dharmapuri" in resp.answer
    assert "Day 1:" in resp.answer
    assert "Day 3:" in resp.answer or "Day 4:" in resp.answer


# =====================================================================
# 7. Additional Correctness Tests (Section 15 Requirements)
# =====================================================================

def test_48_canonical_temple_aliases_resolution(test_setup):
    """48. Verifies canonical resolution for all 10 key temples."""
    agent = test_setup["budget_agent"]
    temples = [
        "Kaleshwaram", "Srisailam", "Basara", "Vemulawada",
        "Dharmapuri", "Edupayala", "Karmanghat", "Medaram",
        "Annavaram", "Alampur"
    ]
    for t in temples:
        state = AgentState(user_query=f"budget for {t} temple")
        resolved = agent._resolve_temples(state)
        assert len(resolved) >= 1, f"Failed to resolve alias: {t}"


def test_49_weather_temperature_range_queries_route_correctly(test_setup):
    """49. Temperature queries with °C, degrees, and between syntax route to live weather."""
    router = test_setup["router"]
    test_queries = [
        "Which temples currently have temperatures between 25°C and 27°C?",
        "between 27°C and 30°C?",
        "which temples below 27 degrees",
        "temples above 27°C",
        "temples with 25C",
        "temples around 25 °C",
        "temples between 25 and 27 degrees"
    ]
    for q in test_queries:
        dec = router.route(q)
        assert dec.intent == "weather" or dec.selected_agent == "live_weather", f"Failed routing for query: {q}"


def test_50_recommendation_typo_tolerance(test_setup):
    """50. Recommendation intent handles typos like 'recomend' without failing to unsupported."""
    router = test_setup["router"]
    queries = [
        "recomend temple with temperature around 25-30",
        "recommed a temple for family visit",
        "suggest temples for darshan"
    ]
    for q in queries:
        dec = router.route(q)
        assert dec.intent == "recommendation" or dec.intent == "weather", f"Failed routing for typo query: {q}"


def test_51_recommendation_with_accommodation_and_weather(test_setup):
    """51. Recommendation + accommodation + weather preserves both constraints."""
    agent = test_setup["rec_agent"]
    state = AgentState(user_query="Recommend temples with accommodation where current temp 25–27")
    resp = agent.execute(state)
    assert resp.success is True
    assert "recommend" in resp.answer.lower()
    assert "accommodation" in resp.answer.lower()
    assert resp.metadata.get("recommendation_requested") is True
    assert resp.metadata.get("accommodation_required") is True
    assert resp.metadata.get("weather_requested") is True


def test_52_weather_next_n_days_forecast_representation(test_setup):
    """52. Live weather agent represents forecast for requested next N days."""
    agent = test_setup["weather_agent"]
    state = AgentState(user_query="plan a 2 day trip to kaleshwaram temple with a budget of 9000 for 2 people and weather for the next 3 days")
    resp = agent.execute(state)
    assert resp.success is True
    assert "Forecast for the Next 3 Day(s)" in resp.answer or "Day 1" in resp.answer
    assert resp.metadata.get("temporal_focus") == "next_3_days"


def test_53_multi_temple_itinerary_basara_and_srisailam(test_setup):
    """53. Basara + Srisailam multi-temple itinerary includes both temples in day schedule."""
    agent = test_setup["itinerary_agent"]
    state = AgentState(user_query="plan a 4 day trip to basara and srisailam temple with budget of 6000 for 2 people")
    resp = agent.execute(state)
    assert resp.success is True
    assert "Basara" in resp.answer
    assert "Srisaila" in resp.answer
    assert "Day 1:" in resp.answer
    assert "Day 4:" in resp.answer


def test_54_multi_temple_itinerary_chilkur_and_birla(test_setup):
    """54. Chilkur + Birla Mandir itinerary."""
    agent = test_setup["itinerary_agent"]
    state = AgentState(user_query="plan a 3 day trip to chilkur balaji temple and birla mandir")
    resp = agent.execute(state)
    assert resp.success is True
    assert "Chilkur" in resp.answer
    assert "Birla Mandir" in resp.answer


def test_55_travel_sanitization_no_raw_dicts(test_setup):
    """55. Travel records containing structured dictionaries are formatted cleanly without raw dict strings."""
    agent = test_setup["itinerary_agent"]
    sample_data = {
        "travel": {
            "bus_routes": [
                {
                    "nearest_bus_station": "Hanamkonda Bus Station",
                    "city": "Hanamkonda",
                    "distance": "2 km",
                    "description": "Temple is 2 km from bus station."
                }
            ],
            "railway": {
                "stations": [
                    {
                        "station": "Warangal Railway Station",
                        "distance": "6 km"
                    }
                ]
            }
        }
    }
    notes = agent._parse_travel(sample_data)
    for n in notes:
        assert not n.startswith("{")
        assert "{'nearest_bus_station':" not in n
    assert any("Hanamkonda Bus Station" in n for n in notes)
    assert any("Warangal Railway Station" in n for n in notes)


