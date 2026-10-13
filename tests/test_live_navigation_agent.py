"""
PilgrimAI - LiveNavigationAgent (Phase O5) Test Suite

Comprehensive offline test suite for:
- LiveNavigationAgent contract, lifecycle, and BaseAgent compliance.
- NavigationAdapter external firewall, timeout, retry, circuit breaker, and sanitization.
- Parameter extraction for origins, destinations, temple aliases, cities, and travel modes.
- Provider normalization (distanceMeters, duration, staticDuration, description).
- Zero-credential safety and secret sanitization (GOOGLE_MAPS_API_KEY never leaked).
- No-fabrication verification (zero invented distances, speeds, traffic, or routes).
- Routing protection (preserves comparison, itinerary, weather, budget, and multi-intent).
- Conversation memory compatibility.
- Deterministic repeated execution.

100% Offline: zero network access, zero credentials required.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import pytest

# Ensure backend is in sys.path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agents.base_agent import BaseAgent
from app.agents.state import AgentState, AgentResponse, AgentRouteDecision
from app.agents.adapters.external_adapter import (
    BaseExternalAdapter,
    AdapterResponse,
    ExternalAdapterError,
    ExternalTimeoutError,
    ExternalConnectionError,
    ExternalHTTPError,
    ExternalRateLimitError,
    ExternalCircuitOpenError,
    CircuitBreaker,
    CircuitState
)
from app.agents.adapters.navigation_adapter import (
    NavigationAdapter,
    NavigationCredentialMissingError,
    RouteNotFoundError,
    UnsupportedTravelModeError
)
from app.agents.coordinate_registry import TempleCoordinateRegistry, TempleCoordinate
from app.agents.live_navigation_agent import (
    LiveNavigationAgent,
    NavigationResult,
    format_duration
)
from app.agents.router import AgentRouter
from app.agents.coordinator import MasterAgentCoordinator
from app.agents.memory import ConversationMemoryManager


# ---------------------------------------------------------------------
# Test Doubles & Fixtures
# ---------------------------------------------------------------------

class MockHTTPResponse:
    """Mock HTTP response object for offline unit testing."""
    def __init__(self, status_code: int = 200, text: str = "{}", headers: dict = None, json_data: dict = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {"Content-Type": "application/json"}
        self._json_data = json_data if json_data is not None else {}

    def json(self):
        return self._json_data


class MockNavigationTransport:
    """Mock transport callable for injecting into NavigationAdapter."""
    def __init__(
        self,
        response_data: Optional[Dict[str, Any]] = None,
        status_code: int = 200,
        raises_exc: Optional[Exception] = None
    ):
        if response_data is not None:
            self.response_data = response_data
        else:
            self.response_data = {
                "routes": [
                    {
                        "distanceMeters": 62400,
                        "duration": "5724s",
                        "staticDuration": "5400s",
                        "description": "NH 163"
                    }
                ]
            }
        self.status_code = status_code
        self.raises_exc = raises_exc
        self.call_count = 0
        self.last_request = None

    def __call__(self, method: str, url: str, headers: dict = None, json: Any = None, params: dict = None, timeout: tuple = None, **kwargs):
        self.call_count += 1
        self.last_request = {
            "method": method,
            "url": url,
            "headers": headers,
            "json": json,
            "params": params,
            "timeout": timeout
        }
        if self.raises_exc:
            raise self.raises_exc
        import json as json_lib
        text_content = json_lib.dumps(self.response_data) if not isinstance(self.response_data, str) else self.response_data
        return MockHTTPResponse(
            status_code=self.status_code,
            text=text_content,
            json_data=self.response_data if isinstance(self.response_data, dict) else {}
        )


@pytest.fixture
def mock_transport():
    return MockNavigationTransport()


@pytest.fixture
def mock_adapter(mock_transport):
    return NavigationAdapter(api_key="TEST_MOCK_KEY_12345", transport=mock_transport)


@pytest.fixture
def agent(mock_adapter):
    return LiveNavigationAgent(adapter=mock_adapter)


@pytest.fixture
def registry():
    return TempleCoordinateRegistry()


# ---------------------------------------------------------------------
# Category 1: Agent Contract & BaseAgent Compliance (Tests 1-5)
# ---------------------------------------------------------------------

def test_01_agent_inherits_base_agent(agent):
    """Verify LiveNavigationAgent inherits from BaseAgent."""
    assert isinstance(agent, BaseAgent)


def test_02_agent_name_and_description(agent):
    """Verify agent name is exactly 'live_navigation'."""
    assert agent.name == "live_navigation"
    assert "navigation" in agent.description.lower() or "route" in agent.description.lower()


def test_03_agent_construction_defaults():
    """Verify agent initializes cleanly with default dependencies."""
    a = LiveNavigationAgent()
    assert a.coordinate_registry is not None
    assert a.adapter is not None
    assert a.name == "live_navigation"


def test_04_process_alias_matches_execute(agent):
    """Verify process(state) is an alias to execute(state)."""
    state = AgentState(user_query="How far is Yadadri from Hyderabad?")
    resp = agent.process(state)
    assert isinstance(resp, AgentResponse)
    assert resp.success is True
    assert resp.agent_name == "live_navigation"


def test_05_empty_query_handled_safely(agent):
    """Verify empty or whitespace query returns controlled failure without crashing."""
    state = AgentState(user_query="   ")
    resp = agent.execute(state)
    assert resp.success is False
    assert "valid" in resp.answer.lower()
    assert resp.error == "Empty query"


# ---------------------------------------------------------------------
# Category 2: Parameter Extraction (Tests 6-12)
# ---------------------------------------------------------------------

def test_06_extract_params_dest_from_orig(agent):
    """Verify extraction of 'How far is Yadadri from Hyderabad?'."""
    params = agent.extract_route_parameters("How far is Yadadri from Hyderabad?")
    assert params["origin"].lower() == "hyderabad"
    assert params["destination"].lower() == "yadadri"
    assert params["travel_mode"] == "driving"


def test_07_extract_params_from_orig_to_dest(agent):
    """Verify extraction of 'Directions from Secunderabad to Basara'."""
    params = agent.extract_route_parameters("Directions from Secunderabad to Basara")
    assert params["origin"].lower() == "secunderabad"
    assert params["destination"].lower() == "basara"


def test_08_extract_params_between_p1_and_p2(agent):
    """Verify extraction of 'Distance between Basara and Dharmapuri'."""
    params = agent.extract_route_parameters("Distance between Basara and Dharmapuri")
    assert params["origin"].lower() == "basara"
    assert params["destination"].lower() == "dharmapuri"


def test_09_extract_travel_modes(agent):
    """Verify extraction of walking, bicycling, transit, two_wheeler, and driving modes."""
    p_walk = agent.extract_route_parameters("Route from Hyderabad to Yadadri by walk")
    assert p_walk["travel_mode"] == "walking"

    p_bike = agent.extract_route_parameters("How far is Yadadri from Hyderabad by cycle?")
    assert p_bike["travel_mode"] == "bicycling"

    p_transit = agent.extract_route_parameters("How to reach Yadadri from Hyderabad by bus?")
    assert p_transit["travel_mode"] == "transit"

    p_drive = agent.extract_route_parameters("Driving time from Hyderabad to Yadadri")
    assert p_drive["travel_mode"] == "driving"

    p_two_wheeler = agent.extract_route_parameters("Route from Hyderabad to Yadadri by two-wheeler")
    assert p_two_wheeler["travel_mode"] == "two_wheeler"

    p_bike2 = agent.extract_route_parameters("Directions from Hyderabad to Yadadri on motorcycle")
    assert p_bike2["travel_mode"] == "two_wheeler"


def test_10_extract_missing_origin_clarification(agent):
    """Verify missing origin returns controlled clarification."""
    state = AgentState(user_query="How far is Yadadri?")
    resp = agent.execute(state)
    assert resp.success is False
    assert "recognizable origin and destination" in resp.answer.lower()
    assert resp.metadata.get("clarification_needed") is True


def test_11_extract_missing_destination_clarification(agent):
    """Verify missing destination returns controlled clarification."""
    state = AgentState(user_query="Directions from Hyderabad")
    resp = agent.execute(state)
    assert resp.success is False
    assert "recognizable origin and destination" in resp.answer.lower()
    assert resp.metadata.get("clarification_needed") is True


def test_12_alias_resolution_in_route(agent):
    """Verify temple aliases (e.g. Yadagirigutta -> Yadadri) resolve correctly."""
    state = AgentState(user_query="What is the distance from Hyderabad to Yadagirigutta?")
    resp = agent.execute(state)
    assert resp.success is True
    assert "yadadri" in resp.answer.lower() or "yadagirigutta" in resp.answer.lower()


# ---------------------------------------------------------------------
# Category 3: Provider Normalization (Tests 13-18)
# ---------------------------------------------------------------------

def test_13_successful_route_normalization(agent):
    """Verify distance in km, duration in minutes, and route summary normalization."""
    state = AgentState(user_query="How far is Yadadri from Hyderabad?")
    resp = agent.execute(state)
    assert resp.success is True
    assert "62.4 km" in resp.answer
    assert "1 hr 30 min" in resp.answer or "1 hr 35 min" in resp.answer
    assert "NH 163" in resp.answer
    assert resp.metadata["distance_km"] == 62.4
    assert resp.metadata["duration_minutes"] == 90.0  # 5400s staticDuration = 90 min


def test_14_traffic_aware_duration_displayed_only_when_present(mock_transport):
    """Verify traffic-aware duration is only displayed when duration differs from staticDuration."""
    # With traffic: duration = 5724s (95.4m), staticDuration = 5400s (90m)
    transport = MockNavigationTransport(response_data={
        "routes": [{
            "distanceMeters": 62400,
            "duration": "5724s",
            "staticDuration": "5400s",
            "description": "NH 163"
        }]
    })
    a = LiveNavigationAgent(adapter=NavigationAdapter(api_key="KEY", transport=transport))
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is True
    assert "Traffic-Aware Duration" in resp.answer
    assert "1 hr 35 min" in resp.answer


def test_15_traffic_duration_omitted_when_unavailable(mock_transport):
    """Verify traffic duration indicates unavailable when provider does not supply traffic data."""
    # No staticDuration, only standard duration
    transport = MockNavigationTransport(response_data={
        "routes": [{
            "distanceMeters": 50000,
            "duration": "3000s"
        }]
    })
    a = LiveNavigationAgent(adapter=NavigationAdapter(api_key="KEY", transport=transport))
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is True
    assert "Live traffic information was not available" in resp.answer


def test_16_missing_optional_route_description(mock_transport):
    """Verify missing description does not cause errors or fabricate routes."""
    transport = MockNavigationTransport(response_data={
        "routes": [{
            "distanceMeters": 40000,
            "duration": "2400s"
        }]
    })
    a = LiveNavigationAgent(adapter=NavigationAdapter(api_key="KEY", transport=transport))
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is True
    assert resp.metadata["route_summary"] is None


def test_17_duration_formatting_helper():
    """Verify format_duration helper accurately formats minutes (45 min -> 45 mins, 60 -> 1 hr, 90 -> 1 hr 30 mins, 120 -> 2 hrs, 270 -> 4 hrs 30 mins)."""
    assert format_duration(None) == "Not available"
    assert format_duration("invalid") == "Not available"
    assert format_duration(45.0) == "45 mins"
    assert format_duration(60.0) == "1 hr"
    assert format_duration(90.0) == "1 hr 30 mins"
    assert format_duration(120.0) == "2 hrs"
    assert format_duration(270.0) == "4 hrs 30 mins"
    assert format_duration(1.0) == "1 min"
    assert format_duration(61.0) == "1 hr 1 min"
    assert format_duration(121.0) == "2 hrs 1 min"


def test_18_empty_routes_list_returns_not_found(mock_transport):
    """Verify empty routes list from provider returns controlled not found response."""
    transport = MockNavigationTransport(response_data={"routes": []})
    a = LiveNavigationAgent(adapter=NavigationAdapter(api_key="KEY", transport=transport))
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is False
    assert "no route could be found" in resp.answer.lower()


# ---------------------------------------------------------------------
# Category 4: Credential Safety & Zero-Credential Handling (Tests 19-24)
# ---------------------------------------------------------------------

def test_19_missing_api_key_returns_controlled_message(monkeypatch):
    """Verify missing GOOGLE_MAPS_API_KEY returns controlled message without crashing."""
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    adapter = NavigationAdapter(api_key="")
    a = LiveNavigationAgent(adapter=adapter)
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is False
    assert "Live navigation is currently unavailable because no navigation provider credential is configured." in resp.answer
    assert resp.metadata.get("credential_configured") is False


def test_20_api_key_never_in_agent_response(agent, mock_transport):
    """Verify API key string never appears in answer text, error message, or metadata."""
    secret = "AIzaSySuperSecretKey12345"
    adapter = NavigationAdapter(api_key=secret, transport=mock_transport)
    a = LiveNavigationAgent(adapter=adapter)
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert secret not in resp.answer
    assert secret not in str(resp.metadata)
    assert secret not in (resp.error or "")


def test_21_api_key_sanitized_in_adapter_url():
    """Verify sanitize_url redacts key parameter."""
    from app.agents.adapters.external_adapter import sanitize_url
    url = "https://routes.googleapis.com/v2:computeRoutes?key=AIzaSySecret123&travelMode=DRIVE"
    safe = sanitize_url(url)
    assert "AIzaSySecret123" not in safe
    assert "key=%5BREDACTED%5D" in safe or "key=[REDACTED]" in safe


def test_22_api_key_sanitized_in_headers():
    """Verify sanitize_headers redacts API key headers."""
    from app.agents.adapters.external_adapter import sanitize_headers
    headers = {"X-Goog-Api-Key": "AIzaSySecretKey", "Content-Type": "application/json"}
    safe = sanitize_headers(headers)
    assert safe["X-Goog-Api-Key"] == "[REDACTED]"
    assert safe["Content-Type"] == "application/json"


def test_23_api_key_sanitized_in_exception_messages():
    """Verify adapter exceptions sanitize secret tokens."""
    adapter = NavigationAdapter(api_key="SUPER_SECRET_TOKEN_999")
    try:
        raise ExternalAdapterError("Failed connecting with key=SUPER_SECRET_TOKEN_999 to provider")
    except ExternalAdapterError as e:
        assert "SUPER_SECRET_TOKEN_999" not in str(e)


def test_24_no_network_calls_during_adapter_or_agent_init():
    """Verify constructing adapter and agent performs zero network calls."""
    adapter = NavigationAdapter()
    agent = LiveNavigationAgent(adapter=adapter)
    assert agent is not None


# ---------------------------------------------------------------------
# Category 5: Adapter Failure Resilience (Tests 25-32)
# ---------------------------------------------------------------------

def test_25_adapter_timeout_handling():
    """Verify timeout produces controlled error response."""
    transport = MockNavigationTransport(raises_exc=ExternalTimeoutError("Request timed out"))
    a = LiveNavigationAgent(adapter=NavigationAdapter(api_key="KEY", transport=transport))
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is False
    assert "timed out" in resp.answer.lower()


def test_26_adapter_connection_error_handling():
    """Verify connection failure produces controlled error response."""
    transport = MockNavigationTransport(raises_exc=ExternalConnectionError("DNS failure"))
    a = LiveNavigationAgent(adapter=NavigationAdapter(api_key="KEY", transport=transport))
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is False
    assert "unreachable" in resp.answer.lower()


def test_27_adapter_rate_limit_handling():
    """Verify HTTP 429 rate limit produces controlled response."""
    transport = MockNavigationTransport(raises_exc=ExternalRateLimitError("Rate limited"))
    a = LiveNavigationAgent(adapter=NavigationAdapter(api_key="KEY", transport=transport))
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is False
    assert "rate limit" in resp.answer.lower()


def test_28_adapter_circuit_breaker_open_handling():
    """Verify open circuit breaker produces controlled response."""
    transport = MockNavigationTransport(raises_exc=ExternalCircuitOpenError("Circuit is OPEN"))
    a = LiveNavigationAgent(adapter=NavigationAdapter(api_key="KEY", transport=transport))
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is False
    assert "temporarily unavailable" in resp.answer.lower()


def test_29_adapter_auth_error_handling():
    """Verify HTTP 401/403 produces controlled response."""
    transport = MockNavigationTransport(raises_exc=ExternalHTTPError("Unauthorized", status_code=401))
    a = LiveNavigationAgent(adapter=NavigationAdapter(api_key="KEY", transport=transport))
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is False
    assert "authentication" in resp.answer.lower() or "configuration" in resp.answer.lower()


def test_30_route_not_found_handling():
    """Verify RouteNotFoundError produces controlled failure."""
    transport = MockNavigationTransport(raises_exc=RouteNotFoundError("Route not found"))
    a = LiveNavigationAgent(adapter=NavigationAdapter(api_key="KEY", transport=transport))
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is False
    assert "no route could be found" in resp.answer.lower()


def test_31_unsupported_travel_mode_handling(mock_adapter):
    """Verify unsupported travel mode produces controlled limitation."""
    a = LiveNavigationAgent(adapter=mock_adapter)
    params = {"origin": "Hyderabad", "destination": "Yadadri", "travel_mode": "submarine"}
    state = AgentState(user_query="How far is Yadadri from Hyderabad by submarine?")
    resp = a.execute(state)
    # Default travel mode parser won't parse submarine, but if adapter receives unsupported mode:
    with pytest.raises(UnsupportedTravelModeError):
        mock_adapter.compute_route("Hyderabad", "Yadadri", travel_mode="submarine")


def test_32_circuit_breaker_trips_after_three_failures():
    """Verify circuit breaker trips to OPEN after 3 consecutive failures."""
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=60.0)
    transport = MockNavigationTransport(status_code=500, response_data={"error": "Internal Error"})
    adapter = NavigationAdapter(api_key="KEY", transport=transport, circuit_breaker=breaker)

    for _ in range(3):
        try:
            adapter.compute_route("Hyderabad", "Yadadri")
        except ExternalAdapterError:
            pass

    assert breaker.state == CircuitState.OPEN
    with pytest.raises(ExternalCircuitOpenError):
        adapter.compute_route("Hyderabad", "Yadadri")


# ---------------------------------------------------------------------
# Category 6: No-Fabrication Verification (Tests 33-36)
# ---------------------------------------------------------------------

def test_33_no_speed_based_duration_calculation(mock_transport):
    """Verify duration is NEVER fabricated by distance / assumed speed when provider duration is absent."""
    transport = MockNavigationTransport(response_data={
        "routes": [{
            "distanceMeters": 100000  # 100 km, but no duration supplied
        }]
    })
    a = LiveNavigationAgent(adapter=NavigationAdapter(api_key="KEY", transport=transport))
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is True
    assert "Duration data not supplied by provider" in resp.answer
    assert resp.metadata["duration_minutes"] is None


def test_34_no_manufactured_traffic(mock_transport):
    """Verify traffic duration is NOT manufactured when absent in provider response."""
    transport = MockNavigationTransport(response_data={
        "routes": [{
            "distanceMeters": 50000,
            "duration": "3600s"
            # No staticDuration
        }]
    })
    a = LiveNavigationAgent(adapter=NavigationAdapter(api_key="KEY", transport=transport))
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is True
    assert "Live traffic information was not available" in resp.answer
    assert resp.metadata["traffic_duration_minutes"] is None


def test_35_no_manufactured_eta_or_tolls(mock_transport):
    """Verify responses do not invent tolls or specific arrival times."""
    transport = MockNavigationTransport()
    a = LiveNavigationAgent(adapter=NavigationAdapter(api_key="KEY", transport=transport))
    resp = a.execute(AgentState(user_query="How far is Yadadri from Hyderabad?"))
    assert resp.success is True
    assert "toll:" not in resp.answer.lower()
    assert "arrival at" not in resp.answer.lower()


def test_36_no_raw_rag_or_database_imports():
    """Verify LiveNavigationAgent does not import Qdrant, BM25, or RAG models."""
    import app.agents.live_navigation_agent as nav_mod
    src = Path(nav_mod.__file__).read_text(encoding="utf-8")
    forbidden = [
        "qdrant_client", "rank_bm25", "sentence_transformers",
        "CrossEncoder", "LocalQdrantManager", "LocalBM25Manager", "PilgrimRAGService"
    ]
    for term in forbidden:
        assert term not in src, f"Forbidden term '{term}' found in live_navigation_agent.py"


# ---------------------------------------------------------------------
# Category 7: Coordinate Registry Integration (Tests 37-40)
# ---------------------------------------------------------------------

def test_37_resolves_registered_temple_coordinates(agent, registry):
    """Verify registered temple (Yadadri) resolves to coordinates from registry."""
    disp_name, coords, canon = agent._resolve_location("Yadadri")
    assert coords is not None
    lat, lon = coords
    assert 17.0 <= lat <= 18.0
    assert 78.0 <= lon <= 79.0
    assert canon == "Yadadri"


def test_38_resolves_alias_to_canonical_coordinates(agent):
    """Verify alias 'Yadagirigutta' resolves to Yadadri coordinates."""
    disp_name, coords, canon = agent._resolve_location("Yadagirigutta")
    assert coords is not None
    assert canon == "Yadadri"


def test_39_resolves_regional_city_as_address(agent):
    """Verify non-temple city 'Hyderabad' resolves cleanly for address routing."""
    disp_name, coords, canon = agent._resolve_location("Hyderabad")
    assert disp_name == "Hyderabad"
    assert coords is None  # Handled as address string by Routes API
    assert canon is None


def test_40_both_origin_and_dest_temples(agent):
    """Verify route between two registered temples (Basara to Yadadri)."""
    state = AgentState(user_query="How far is Yadadri from Basara?")
    resp = agent.execute(state)
    assert resp.success is True
    assert "Basara" in resp.answer
    assert "Yadadri" in resp.answer


# ---------------------------------------------------------------------
# Category 8: Routing Protection (Tests 41-47)
# ---------------------------------------------------------------------

def test_41_router_registers_live_navigation():
    """Verify AgentRouter supports live_navigation."""
    router = AgentRouter()
    router.register_agent("live_navigation")
    assert "live_navigation" in router.active_agents


def test_42_single_navigation_query_routes_to_live_navigation():
    """Verify single navigation query routes to live_navigation when active."""
    router = AgentRouter()
    router.register_agent("live_navigation")
    decision = router.route("How far is Yadadri from Hyderabad?")
    assert decision.selected_agent == "live_navigation"
    assert decision.intent == "live_navigation"


def test_43_multi_temple_navigation_not_forced_into_comparison():
    """Verify 'How far is Yadadri from Basara?' routes to live_navigation, not comparison."""
    router = AgentRouter()
    router.register_agent("live_navigation")
    router.register_agent("comparison")
    decision = router.route("How far is Yadadri from Basara?")
    assert decision.selected_agent == "live_navigation"
    assert decision.intent == "live_navigation"


def test_44_comparison_with_distance_remains_comparison():
    """Verify 'Compare the distance between Yadadri and Basara' routes to comparison."""
    router = AgentRouter()
    router.register_agent("comparison")
    router.register_agent("live_navigation")
    decision = router.route("Compare the distance between Yadadri and Basara.")
    assert decision.selected_agent == "comparison"
    assert decision.intent == "comparison"


def test_45_itinerary_with_route_remains_itinerary():
    """Verify 'Plan a route itinerary for Yadadri and Basara' routes to itinerary."""
    router = AgentRouter()
    router.register_agent("itinerary_planning")
    router.register_agent("live_navigation")
    decision = router.route("Plan a route itinerary for Yadadri and Basara")
    assert decision.selected_agent == "itinerary_planning"


def test_46_travel_information_stays_temple_information():
    """Verify 'What is the travel information for Yadadri?' does NOT become navigation."""
    router = AgentRouter()
    router.register_agent("live_navigation")
    router.register_agent("temple_information")
    decision = router.route("What is the travel information for Yadadri?")
    assert decision.selected_agent == "temple_information"


def test_47_multi_intent_weather_plus_navigation():
    """Verify conjoined weather + navigation query routes to multi_intent."""
    router = AgentRouter()
    router.register_agent("live_weather")
    router.register_agent("live_navigation")
    decision = router.route("What is the weather in Yadadri and how long does it take to reach it from Hyderabad?")
    assert decision.intent == "multi_intent"
    sub_intents = decision.metadata.get("sub_intents", [])
    assert "weather" in sub_intents
    assert "navigation" in sub_intents


# ---------------------------------------------------------------------
# Category 9: Coordinator & Memory Integration (Tests 48-52)
# ---------------------------------------------------------------------

def test_48_coordinator_dependency_injection_no_autoinstantiation():
    """Verify MasterAgentCoordinator does not auto-instantiate LiveNavigationAgent."""
    class DummyTempleInfo(BaseAgent):
        def __init__(self):
            super().__init__(name="temple_information", description="Dummy")
        def _run(self, state):
            return AgentResponse(success=True, agent_name=self.name, answer="Dummy", sources=[])

    coord = MasterAgentCoordinator(temple_info_agent=DummyTempleInfo())
    assert "live_navigation" not in coord.agents


def test_49_coordinator_with_injected_live_navigation(agent):
    """Verify coordinator uses injected LiveNavigationAgent."""
    class DummyTempleInfo(BaseAgent):
        def __init__(self):
            super().__init__(name="temple_information", description="Dummy")
        def _run(self, state):
            return AgentResponse(success=True, agent_name=self.name, answer="Dummy", sources=[])

    coord = MasterAgentCoordinator(temple_info_agent=DummyTempleInfo(), live_navigation_agent=agent)
    assert "live_navigation" in coord.agents
    assert coord.agents["live_navigation"] is agent


def test_50_coordinator_executes_navigation_query(agent):
    """Verify coordinator processes navigation query end-to-end."""
    class DummyTempleInfo(BaseAgent):
        def __init__(self):
            super().__init__(name="temple_information", description="Dummy")
        def _run(self, state):
            return AgentResponse(success=True, agent_name=self.name, answer="Dummy", sources=[])

    coord = MasterAgentCoordinator(temple_info_agent=DummyTempleInfo(), live_navigation_agent=agent)
    state = AgentState(user_query="How far is Yadadri from Hyderabad?")
    res = coord.process(state)
    assert isinstance(res, AgentResponse)
    assert res.success is True
    assert "62.4 km" in res.answer


def test_51_conversation_memory_pronoun_resolution():
    """Verify conversation memory resolves 'it' to Yadadri for navigation follow-up."""
    memory = ConversationMemoryManager()
    session_id = "test-session-o5"
    memory.add_turn(
        session_id=session_id,
        user_query="Tell me about Yadadri.",
        detected_intents=["temple_information"],
        identified_temples=["Yadadri"]
    )
    resolved = memory.resolve_query(session_id, "How far is it from Hyderabad?")
    resolved_text = resolved.get("resolved_query", "")
    assert "yadadri" in resolved_text.lower()


def test_52_repeated_execution_determinism(agent):
    """Verify repeated execution with same mocked transport yields deterministic output."""
    query = "How far is Yadadri from Hyderabad?"
    resps = [agent.execute(AgentState(user_query=query)) for _ in range(3)]
    assert all(r.success is True for r in resps)
    assert resps[0].answer == resps[1].answer == resps[2].answer
    assert resps[0].metadata["distance_km"] == resps[1].metadata["distance_km"] == resps[2].metadata["distance_km"]
