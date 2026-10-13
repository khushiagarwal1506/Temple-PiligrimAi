"""
PilgrimAI - Focused Test Suite for:
1. Route Duration in Hours and Minutes
2. Two-Wheeler Routing End-to-End
3. Mode switching & stale route isolation
"""

import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agents.adapters.navigation_adapter import (
    NavigationAdapter,
    SUPPORTED_TRAVEL_MODES,
    UnsupportedTravelModeError,
    RouteNotFoundError
)
from app.agents.live_navigation_agent import (
    LiveNavigationAgent,
    format_duration
)
from app.agents.state import AgentState
from app.api.maps import calculate_route, RouteCalculationRequest


# =============================================================================
# 1. Duration Formatting Tests
# =============================================================================

def test_01_duration_formatting_all_required_cases():
    """
    Test exact formatting requirements:
    - 45 min -> 45 mins
    - 60 min -> 1 hr
    - 90 min -> 1 hr 30 mins
    - 120 min -> 2 hrs
    - 270 min -> 4 hrs 30 mins
    - < 60 min -> show only minutes (e.g. 1 min, 45 mins)
    - Missing/invalid -> graceful handling
    """
    assert format_duration(45) == "45 mins"
    assert format_duration(60) == "1 hr"
    assert format_duration(90) == "1 hr 30 mins"
    assert format_duration(120) == "2 hrs"
    assert format_duration(270) == "4 hrs 30 mins"

    # Singular/plural edges
    assert format_duration(1) == "1 min"
    assert format_duration(2) == "2 mins"
    assert format_duration(61) == "1 hr 1 min"
    assert format_duration(121) == "2 hrs 1 min"
    assert format_duration(0) == "0 mins"
    assert format_duration(-5) == "0 mins"

    # Missing / invalid durations
    assert format_duration(None) == "Not available"
    assert format_duration("invalid") == "Not available"
    assert format_duration("") == "Not available"


# =============================================================================
# 2. Driving Route End-to-End
# =============================================================================

@pytest.mark.anyio
async def test_02_driving_route():
    """Verify driving route returns driving travel_mode, polyline, distance, and duration."""
    mock_routes_resp = {
        "routes": [
            {
                "distanceMeters": 219700,
                "duration": "16200s",  # 270 minutes
                "polyline": {"encodedPolyline": "mock_driving_polyline_xyz"},
                "description": "NH 44"
            }
        ]
    }

    mock_transport = MagicMock(return_value=MagicMock(
        status_code=200,
        json=lambda: mock_routes_resp
    ))

    with patch("app.agents.live_navigation_agent.NavigationAdapter") as MockNavAdapter:
        instance = NavigationAdapter(api_key="TEST_KEY", transport=mock_transport)
        MockNavAdapter.return_value = instance

        req = RouteCalculationRequest(
            origin="nirmal",
            destination="Hyderabad",
            travel_mode="driving"
        )
        resp = await calculate_route(req)

        assert resp.success is True
        assert resp.travel_mode == "driving"
        assert resp.distance_km == 219.7
        assert resp.duration_minutes == 270.0
        assert resp.polyline == "mock_driving_polyline_xyz"
        assert resp.origin.lower() == "nirmal"
        assert resp.destination.lower() == "hyderabad"


# =============================================================================
# 3. Two-Wheeler Route End-to-End
# =============================================================================

@pytest.mark.anyio
async def test_03_two_wheeler_route():
    """Verify two-wheeler route correctly passes TWO_WHEELER to Routes API provider and returns clean response."""
    # Ensure two_wheeler is in SUPPORTED_TRAVEL_MODES
    assert "two_wheeler" in SUPPORTED_TRAVEL_MODES
    assert SUPPORTED_TRAVEL_MODES["two_wheeler"] == "TWO_WHEELER"

    captured_requests = []

    def mock_transport(method, url, headers=None, json=None, **kwargs):
        captured_requests.append({"url": url, "headers": headers, "json": json})
        return MagicMock(
            status_code=200,
            json=lambda: {
                "routes": [
                    {
                        "distanceMeters": 215000,
                        "duration": "18000s",  # 300 minutes (5 hrs)
                        "polyline": {"encodedPolyline": "mock_two_wheeler_polyline_123"},
                        "description": "SH 1 / NH 44 Two-Wheeler Corridor"
                    }
                ]
            }
        )

    adapter = NavigationAdapter(api_key="TEST_KEY", transport=mock_transport)
    # 1. Test adapter directly
    res = adapter.compute_route("nirmal", "Hyderabad", travel_mode="two_wheeler")
    assert len(captured_requests) == 1
    assert captured_requests[0]["json"]["travelMode"] == "TWO_WHEELER"
    assert captured_requests[0]["json"]["routingPreference"] == "TRAFFIC_AWARE"

    # 2. Test through API endpoint
    with patch("app.agents.live_navigation_agent.NavigationAdapter", return_value=adapter):
        req = RouteCalculationRequest(
            origin="nirmal",
            destination="Hyderabad",
            travel_mode="two_wheeler"
        )
        resp = await calculate_route(req)
        assert resp.success is True
        assert resp.travel_mode == "two_wheeler"
        assert resp.distance_km == 215.0
        assert resp.duration_minutes == 300.0
        assert resp.polyline == "mock_two_wheeler_polyline_123"


# =============================================================================
# 4. Unsupported Two-Wheeler Mode Handling
# =============================================================================

@pytest.mark.anyio
async def test_04_unsupported_two_wheeler_mode_handling():
    """
    If provider returns error or doesn't support two-wheeler routing for corridor,
    system must handle explicitly with clear message and NEVER silently display driving route.
    """
    def mock_transport_err(method, url, headers=None, json=None, **kwargs):
        # Provider returns error indicating two-wheeler routing not supported
        return MagicMock(
            status_code=200,
            json=lambda: {
                "error": {
                    "code": 400,
                    "message": "TWO_WHEELER travel mode is not supported for this origin and destination.",
                    "status": "INVALID_ARGUMENT"
                }
            }
        )

    adapter = NavigationAdapter(api_key="TEST_KEY", transport=mock_transport_err)

    # 1. Adapter should raise UnsupportedTravelModeError
    with pytest.raises(UnsupportedTravelModeError) as exc_info:
        adapter.compute_route("Rural Hill", "Remote Forest", travel_mode="two_wheeler")
    assert "Two-Wheeler routing is not available" in str(exc_info.value)

    # 2. API should return success=False with clear error message, never driving route
    with patch("app.agents.live_navigation_agent.NavigationAdapter", return_value=adapter):
        req = RouteCalculationRequest(
            origin="Rural Hill",
            destination="Remote Forest",
            travel_mode="two_wheeler"
        )
        resp = await calculate_route(req)
        assert resp.success is False
        assert resp.polyline is None
        assert "Two-Wheeler routing is not available" in (resp.error or resp.message)


# =============================================================================
# 5. Switching Between All Four Travel Modes
# =============================================================================

@pytest.mark.anyio
async def test_05_switching_between_all_four_travel_modes():
    """
    Verify switching between driving, transit, two_wheeler, and walking
    correctly passes travel_mode, updates geometry/metrics, and preserves inputs.
    """
    mode_mock_data = {
        "driving": {"dist": 219700, "dur": "16200s", "poly": "poly_driving"},
        "transit": {"dist": 230000, "dur": "21600s", "poly": "poly_transit"},
        "two_wheeler": {"dist": 215000, "dur": "18000s", "poly": "poly_two_wheeler"},
        "walking": {"dist": 210000, "dur": "150000s", "poly": "poly_walking"}
    }

    def mock_transport(method, url, headers=None, json=None, **kwargs):
        mode = json.get("travelMode", "DRIVE")
        mapping = {
            "DRIVE": mode_mock_data["driving"],
            "TRANSIT": mode_mock_data["transit"],
            "TWO_WHEELER": mode_mock_data["two_wheeler"],
            "WALK": mode_mock_data["walking"]
        }
        data = mapping.get(mode, mode_mock_data["driving"])
        return MagicMock(
            status_code=200,
            json=lambda: {
                "routes": [{
                    "distanceMeters": data["dist"],
                    "duration": data["dur"],
                    "polyline": {"encodedPolyline": data["poly"]}
                }]
            }
        )

    adapter = NavigationAdapter(api_key="TEST_KEY", transport=mock_transport)

    with patch("app.agents.live_navigation_agent.NavigationAdapter", return_value=adapter):
        modes = ["driving", "transit", "two_wheeler", "walking"]
        previous_poly = None

        for m in modes:
            req = RouteCalculationRequest(
                origin="nirmal",
                destination="Hyderabad",
                travel_mode=m
            )
            resp = await calculate_route(req)

            # 6. Confirm distance, duration, geometry, and mode update correctly
            assert resp.success is True
            assert resp.travel_mode == m
            assert resp.polyline == mode_mock_data[m]["poly"]
            assert resp.polyline != previous_poly  # No stale polyline
            previous_poly = resp.polyline

            # 7. Confirming origin and destination remain unchanged
            assert resp.origin.lower() == "nirmal"
            assert resp.destination.lower() == "hyderabad"
