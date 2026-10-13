"""
PilgrimAI - Navigation External Adapter (Phase O5)

Provides a specialized external adapter firewall for Google Maps Routes API v2.
Inherits from BaseExternalAdapter to guarantee:
- Configurable connect and read timeouts (default: 3.0s connect, 5.0s read).
- Bounded retry policy (maximum 1 retry) for transient network/server failures.
- In-process circuit breaker (trips to OPEN after 3 consecutive failures).
- Secret sanitization: GOOGLE_MAPS_API_KEY is never leaked into logs, URLs, or exceptions.
- Zero-credential safety: gracefully reports missing credentials without throwing unhandled errors.
- Deterministic offline testability via injected mock transport.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, Tuple, Union, Callable
from urllib.parse import urlparse, urlunparse

from app.agents.adapters.external_adapter import (
    BaseExternalAdapter,
    AdapterResponse,
    ExternalAdapterError,
    ExternalTimeoutError,
    ExternalConnectionError,
    ExternalHTTPError,
    ExternalRateLimitError,
    ExternalCircuitOpenError,
    sanitize_url,
    sanitize_headers,
    sanitize_text,
)

logger = logging.getLogger(__name__)

# Default Google Maps Routes API v2 endpoint
ROUTES_API_V2_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

SUPPORTED_TRAVEL_MODES = {
    "driving": "DRIVE",
    "walking": "WALK",
    "bicycling": "BICYCLE",
    "transit": "TRANSIT",
    "two_wheeler": "TWO_WHEELER",
    "two-wheeler": "TWO_WHEELER",
    "motorcycle": "TWO_WHEELER",
    "bike": "TWO_WHEELER",
    "scooter": "TWO_WHEELER"
}


class NavigationCredentialMissingError(ExternalAdapterError):
    """Raised when navigation requests are attempted without a configured API key."""
    pass


class RouteNotFoundError(ExternalAdapterError):
    """Raised when the navigation provider finds no route between origin and destination."""
    pass


class UnsupportedTravelModeError(ExternalAdapterError):
    """Raised when the requested travel mode is not supported by the provider configuration."""
    pass


class NavigationAdapter(BaseExternalAdapter):
    """
    Firewall adapter specifically interfacing with Google Maps Routes API v2.
    Translates origin and destination coordinates or addresses into Routes API v2 computeRoutes requests.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        connect_timeout: float = 3.0,
        read_timeout: float = 5.0,
        max_retries: int = 1,
        circuit_breaker: Optional[Any] = None,
        transport: Optional[Callable] = None,
        api_url: str = ROUTES_API_V2_URL
    ):
        super().__init__(
            name="navigation",
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_retries=max_retries,
            circuit_breaker=circuit_breaker,
            transport=transport
        )
        # Read server-side Routes credential from environment if not explicitly injected.
        # Prioritizes dedicated GOOGLE_ROUTES_API_KEY.
        # Note: The GOOGLE_MAPS_API_KEY fallback is provided strictly for offline mock test suites
        # and backward compatibility; in production, browser keys with HTTP-referrer restrictions
        # will be rejected by Google Routes API with HTTP 403.
        if api_key is not None:
            raw_key = api_key
        else:
            raw_key = os.environ.get("GOOGLE_ROUTES_API_KEY") or os.environ.get("GOOGLE_MAPS_API_KEY", "")
        self._api_key = raw_key.strip() if raw_key else ""
        self.api_url = api_url

    @property
    def has_credentials(self) -> bool:
        """Returns True if a non-empty API key is configured."""
        return bool(self._api_key)

    def _format_waypoint(self, location: Union[str, Tuple[float, float], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Formats a waypoint for Google Maps Routes API v2:
        - (lat, lon) -> {"location": {"latLng": {"latitude": lat, "longitude": lon}}}
        - "Address" -> {"address": "Address"}
        - Dict with latLng or address -> passes through
        """
        if isinstance(location, tuple) and len(location) == 2:
            lat, lon = location
            return {
                "location": {
                    "latLng": {
                        "latitude": float(lat),
                        "longitude": float(lon)
                    }
                }
            }
        elif isinstance(location, dict):
            if "location" in location or "address" in location:
                return location
            if "latitude" in location and "longitude" in location:
                return {
                    "location": {
                        "latLng": {
                            "latitude": float(location["latitude"]),
                            "longitude": float(location["longitude"])
                        }
                    }
                }
            return location
        else:
            # Fallback to address string
            addr = str(location).strip()
            if not re.search(r"\b(telangana|andhra|india)\b", addr, re.IGNORECASE):
                addr = f"{addr}, Telangana, India"
            return {"address": addr}

    def compute_route(
        self,
        origin: Union[str, Tuple[float, float], Dict[str, Any]],
        destination: Union[str, Tuple[float, float], Dict[str, Any]],
        travel_mode: str = "driving",
        routing_preference: str = "TRAFFIC_AWARE"
    ) -> Dict[str, Any]:
        """
        Queries Google Maps Routes API v2 computeRoutes endpoint.
        
        Args:
            origin: Origin coordinate tuple (lat, lon), address string, or waypoint dict.
            destination: Destination coordinate tuple (lat, lon), address string, or waypoint dict.
            travel_mode: Mode of travel ('driving', 'walking', 'bicycling', 'transit').
            routing_preference: 'TRAFFIC_AWARE', 'TRAFFIC_UNAWARE', or 'TRAFFIC_AWARE_OPTIMAL'.
            
        Returns:
            Raw dictionary returned by the Routes API containing routes list.
            
        Raises:
            NavigationCredentialMissingError: If no API key is configured and no mock transport is provided.
            UnsupportedTravelModeError: If travel mode is unsupported.
            RouteNotFoundError: If provider indicates no route found.
            ExternalTimeoutError: If request times out.
            ExternalConnectionError: If network connection fails.
            ExternalHTTPError: On HTTP 4xx/5xx responses.
            ExternalCircuitOpenError: If circuit breaker is open.
        """
        mode_lower = travel_mode.strip().lower()
        if mode_lower not in SUPPORTED_TRAVEL_MODES:
            raise UnsupportedTravelModeError(
                f"Travel mode '{travel_mode}' is not supported. Supported modes: {list(SUPPORTED_TRAVEL_MODES.keys())}."
            )

        # In live mode (no mock transport), require a configured API key
        if not self.has_credentials and self.transport is None:
            raise NavigationCredentialMissingError(
                "Live navigation is currently unavailable because no navigation provider credential is configured."
            )

        api_travel_mode = SUPPORTED_TRAVEL_MODES[mode_lower]

        # Routing preference is supported for DRIVE and TWO_WHEELER in Routes API v2
        payload: Dict[str, Any] = {
            "origin": self._format_waypoint(origin),
            "destination": self._format_waypoint(destination),
            "travelMode": api_travel_mode
        }
        if api_travel_mode in ("DRIVE", "TWO_WHEELER"):
            payload["routingPreference"] = routing_preference

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.description,routes.staticDuration,routes.polyline.encodedPolyline"
        }

        try:
            resp: AdapterResponse = self.request(
                method="POST",
                url=self.api_url,
                headers=headers,
                json_data=payload
            )
        except ExternalHTTPError as e:
            if e.status_code in (401, 403):
                safe_err = sanitize_text(str(e))
                logger.error(f"Navigation authentication failed: {safe_err}")
                raise ExternalHTTPError(
                    message="Navigation provider authentication failed. Please verify provider credentials.",
                    status_code=e.status_code,
                    sanitized_url=sanitize_url(self.api_url)
                ) from None
            elif e.status_code == 400 and api_travel_mode == "TWO_WHEELER":
                logger.warning(f"Two-wheeler routing unsupported or bad request: {e}")
                raise UnsupportedTravelModeError(
                    "Two-Wheeler routing is not available for this corridor or with current provider settings. Please use Driving or Transit."
                ) from None
            elif e.status_code == 404:
                raise RouteNotFoundError(
                    "No route could be found between the specified origin and destination."
                ) from None
            raise

        try:
            data = resp.json()
        except Exception as e:
            raise ExternalAdapterError(f"Malformed JSON response from navigation provider: {e}") from e

        # Check for provider-level errors in payload
        if isinstance(data, dict):
            if "error" in data:
                err_dict = data["error"]
                msg = err_dict.get("message", "Unknown provider error")
                status = err_dict.get("status", "")
                if status == "NOT_FOUND" or "ZERO_RESULTS" in msg:
                    raise RouteNotFoundError("No route could be found between the specified origin and destination.")
                if "TWO_WHEELER" in msg or "two_wheeler" in msg.lower() or "two-wheeler" in msg.lower():
                    raise UnsupportedTravelModeError(
                        "Two-Wheeler routing is not available for this corridor or with current provider settings. Please use Driving or Transit."
                    )
                raise ExternalAdapterError(f"Navigation provider returned error: {sanitize_text(msg)}")

            routes = data.get("routes", [])
            if not routes:
                if api_travel_mode == "TWO_WHEELER":
                    raise RouteNotFoundError("No Two-Wheeler route could be found between the specified origin and destination.")
                raise RouteNotFoundError("No route could be found between the specified origin and destination.")

        return data
