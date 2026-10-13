"""
PilgrimAI - Live Navigation Agent (Phase O5)

Provides real-time routing, distance, travel duration, and direction details
between origins and pilgrimage temples using Google Maps Routes API v2 through
NavigationAdapter (O1 Firewall) and TempleCoordinateRegistry (O2 Registry).

Core Principles:
- Provider: Google Maps Routes API v2 via NavigationAdapter firewall.
- Zero credential leakage: API keys are never exposed in responses, logs, or traces.
- Zero-credential safety: missing credentials return a controlled informational message.
- Zero fabrication: distances, durations, routes, traffic, tolls, and speeds are NEVER invented.
- Pure provenance: only provider-returned fields are displayed; unsupplied fields are explicitly noted.
- Decoupled: zero raw RAG, vector DB, or direct HTTP imports.
- Supports canonical temples, temple aliases, and recognized regional cities/places.
"""

import re
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent
from app.agents.state import AgentState, AgentResponse
from app.agents.adapters.external_adapter import (
    ExternalAdapterError,
    ExternalTimeoutError,
    ExternalConnectionError,
    ExternalHTTPError,
    ExternalRateLimitError,
    ExternalCircuitOpenError
)
from app.agents.adapters.navigation_adapter import (
    NavigationAdapter,
    NavigationCredentialMissingError,
    RouteNotFoundError,
    UnsupportedTravelModeError
)
from app.agents.coordinate_registry import TempleCoordinateRegistry, TempleCoordinate

logger = logging.getLogger(__name__)

# Known major regional cities/places with verified coordinates
_REGIONAL_CITY_COORDINATES: Dict[str, Tuple[float, float, str]] = {
    "hyderabad": (17.385044, 78.486671, "Hyderabad"),
    "secunderabad": (17.4399, 78.4983, "Secunderabad"),
    "warangal": (17.9689, 79.5941, "Warangal"),
    "karimnagar": (18.4386, 79.1288, "Karimnagar"),
    "nizamabad": (18.6725, 78.0941, "Nizamabad"),
    "khammam": (17.2473, 80.1514, "Khammam"),
    "nalgonda": (17.0575, 79.2684, "Nalgonda"),
    "suryapet": (17.1439, 79.6239, "Suryapet"),
    "mahabubnagar": (16.7488, 77.9856, "Mahabubnagar"),
    "siddipet": (18.1018, 78.8520, "Siddipet"),
    "ramagundam": (18.7551, 79.5133, "Ramagundam"),
    "mancherial": (18.8679, 79.4639, "Mancherial"),
    "kazipet": (17.9790, 79.5218, "Kazipet"),
    "hanamkonda": (18.0138, 79.5518, "Hanamkonda"),
    "vijayawada": (16.5062, 80.6480, "Vijayawada"),
    "guntur": (16.3067, 80.4365, "Guntur"),
    "visakhapatnam": (17.6868, 83.2185, "Visakhapatnam"),
    "tirupati": (13.6288, 79.4192, "Tirupati"),
    "kurnool": (15.8281, 78.0373, "Kurnool"),
    "bengaluru": (12.9716, 77.5946, "Bengaluru"),
    "bangalore": (12.9716, 77.5946, "Bengaluru"),
    "chennai": (13.0827, 80.2707, "Chennai"),
    "mumbai": (19.0760, 72.8777, "Mumbai"),
    "delhi": (28.6139, 77.2090, "Delhi"),
    "pune": (18.5204, 73.8567, "Pune"),
    "palvancha": (17.5988, 80.7064, "Palvancha"),
    "palwancha": (17.5988, 80.7064, "Palvancha"),
    "kothagudem": (17.5487, 80.6178, "Kothagudem"),
    "bhadrachalam": (17.6688, 80.8936, "Bhadrachalam"),
    "kaleshwaram": (18.8122, 79.9075, "Kaleshwaram"),
    "medaram": (18.2368, 80.2858, "Medaram"),
    "basara": (18.9754, 77.9542, "Basara"),
    "vemulawada": (18.4682, 78.8687, "Vemulawada"),
    "yadagirigutta": (17.5886, 78.9392, "Yadagirigutta"),
    "srisailam": (16.0745, 78.8682, "Srisailam"),
    "alampur": (15.8797, 78.1325, "Alampur"),
    "dharmapuri": (18.9486, 79.0911, "Dharmapuri")
}

_KNOWN_REGIONAL_PLACES = set(_REGIONAL_CITY_COORDINATES.keys())


class NavigationResult(BaseModel):
    """Normalized structured data model representing route computation results."""
    origin: str = Field(..., description="User-specified origin name")
    destination: str = Field(..., description="User-specified destination name")
    origin_canonical: Optional[str] = Field(default=None, description="Canonical name if a registered temple")
    destination_canonical: Optional[str] = Field(default=None, description="Canonical name if a registered temple")
    origin_coordinates: Optional[Tuple[float, float]] = Field(default=None, description="Origin latitude, longitude")
    destination_coordinates: Optional[Tuple[float, float]] = Field(default=None, description="Destination latitude, longitude")
    polyline: Optional[str] = Field(default=None, description="Encoded route polyline for map rendering")
    distance_km: Optional[float] = Field(default=None, description="Driving/travel distance in kilometers")
    duration_minutes: Optional[float] = Field(default=None, description="Estimated travel duration in minutes")
    traffic_duration_minutes: Optional[float] = Field(default=None, description="Traffic-aware duration in minutes if returned")
    route_summary: Optional[str] = Field(default=None, description="Provider route description or major highway")
    travel_mode: str = Field(default="driving", description="Travel mode (driving, walking, bicycling, transit)")
    provider: str = Field(default="Google Maps Routes API", description="Underlying navigation service provider")
    retrieved_at: Optional[str] = Field(default=None, description="ISO timestamp of route retrieval")
    cache_hit: bool = Field(default=False, description="Whether result was served from cache")
    route_status: str = Field(default="SUCCESS", description="Route status: SUCCESS, PARTIAL_DATA, NO_ROUTE, PROVIDER_ERROR, INVALID_DESTINATION")


def format_duration(minutes: Optional[Union[float, int, str]]) -> str:
    """
    Formats numeric minutes into human-readable duration string.
    Rules:
    - 45 min -> 45 mins (or 1 min if singular)
    - 60 min -> 1 hr
    - 90 min -> 1 hr 30 mins
    - 120 min -> 2 hrs
    - 270 min -> 4 hrs 30 mins
    - Under 1 hour: shows only minutes
    - Missing or invalid: 'Not available'
    """
    if minutes is None:
        return "Not available"
    try:
        total_mins = int(round(float(minutes)))
    except (ValueError, TypeError):
        return "Not available"
    if total_mins <= 0:
        return "0 mins"
    if total_mins < 60:
        return "1 min" if total_mins == 1 else f"{total_mins} mins"
    hours = total_mins // 60
    rem_mins = total_mins % 60
    hr_str = "1 hr" if hours == 1 else f"{hours} hrs"
    if rem_mins == 0:
        return hr_str
    min_str = "1 min" if rem_mins == 1 else f"{rem_mins} mins"
    return f"{hr_str} {min_str}"


class LiveNavigationAgent(BaseAgent):
    """
    Specialist agent for retrieving live navigation routes, distance, and duration.
    Interfaces exclusively through NavigationAdapter and TempleCoordinateRegistry.
    """

    def __init__(
        self,
        coordinate_registry: Optional[TempleCoordinateRegistry] = None,
        adapter: Optional[NavigationAdapter] = None,
        cache_ttl_sec: float = 60.0,
        cache_enabled: bool = False,
        memory_manager: Optional[Any] = None
    ):
        super().__init__(
            name="live_navigation",
            description="Fetches live route, distance, travel duration, and navigation details using external navigation provider."
        )
        self.coordinate_registry = coordinate_registry or TempleCoordinateRegistry()
        self.adapter = adapter or NavigationAdapter()
        self.cache_ttl_sec = float(cache_ttl_sec)
        self.cache_enabled = bool(cache_enabled)
        self.memory_manager = memory_manager
        self._cache: Dict[str, Tuple[float, NavigationResult]] = {}

    def extract_route_parameters(self, query: str) -> Dict[str, Any]:
        """
        Deterministically extracts origin, destination, and travel mode from query text.
        
        Patterns supported:
        - 'i want to travel to {dest} from {orig} give me bus routes'
        - 'travel from {orig} to {dest}'
        - 'travel to {dest} from {orig}'
        - 'how do i get from {orig} to {dest}'
        - 'route from {orig} to {dest}'
        - 'bus from {orig} to {dest}'
        - 'give me bus routes from {orig} to {dest}'
        - 'how far is {dest} from {orig}'
        - 'how far is {orig} from {dest}'
        - 'distance from {orig} to {dest}'
        - 'directions from {orig} to {dest}'
        - 'distance between {orig} and {dest}'
        - 'route between {orig} and {dest}'
        - 'how long does it take to reach {dest} from {orig}'
        - 'driving time from {orig} to {dest}'
        """
        clean_q = query.strip()
        q_lower = clean_q.lower()

        # 1. Detect travel mode
        travel_mode = "driving"
        if re.search(r"\b(two[-\s]?wheeler|two_wheeler|motorcycle|scooter|scooty|moped)\b", q_lower):
            travel_mode = "two_wheeler"
        elif re.search(r"\b(walk|walking|by\s+foot|on\s+foot)\b", q_lower):
            travel_mode = "walking"
        elif re.search(r"\b(cycle|cycling|bicycle|bicycling|by\s+bike|by\s+cycle)\b", q_lower):
            travel_mode = "bicycling"
        elif re.search(r"\b(transit|bus|buses|train|trains|public\s+transit|by\s+bus|by\s+train)\b", q_lower):
            travel_mode = "transit"

        # 2. Origin & Destination Extraction
        orig_raw: Optional[str] = None
        dest_raw: Optional[str] = None

        # Clean trailing question marks and punctuation
        stripped_q = re.sub(r"[?!.,;]+$", "", clean_q).strip()

        # Pattern A: 'between {orig} and {dest}'
        m_between = re.search(
            r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\s+(?:by\s+(?:car|walk|bike|cycle|transit|driving|foot|bus|train|road|cab|taxi|flight)|give\s+me\s+.*|please.*))?$",
            stripped_q,
            re.IGNORECASE
        )
        if m_between:
            orig_raw, dest_raw = m_between.group(1), m_between.group(2)

        # Pattern B: '{orig} and want to visit {dest}' / 'I'm at {orig} and want to visit {dest}'
        if not orig_raw or not dest_raw:
            if not re.match(r"^(?:i|we)\s+(?:want\s+to|planning\s+to|plan\s+to|would\s+like\s+to)", stripped_q, re.IGNORECASE):
                m_want = re.search(
                    r"(?:(?:i'm|i\s+am)\s+(?:in|at)\s+|from\s+)?(.+?)\s+(?:and\s+)?(?:want\s+to|planning\s+to|looking\s+to)\s+(?:visit|reach|go\s+to|travel\s+to|see)\s+(.+?)$",
                    stripped_q,
                    re.IGNORECASE
                )
                if m_want:
                    cand_orig = m_want.group(1).strip()
                    if cand_orig.lower() not in {"i", "we", "i'm", "i am", ""}:
                        orig_raw, dest_raw = cand_orig, m_want.group(2)

        # Pattern C: '{action/prefix} to {dest} from {orig} {suffix}'
        # e.g., 'i want to travel to vemulavada from hyderabad give me bus routes'
        # e.g., 'how to go to basara from hyderabad by road'
        if not orig_raw or not dest_raw:
            m_to_from = re.search(
                r"(?:(?:i\s+want\s+to\s+)?(?:travel|go|drive|head)\s+to|how\s+(?:do\s+i|can\s+i|to)\s+(?:get|reach|go|travel)\s+(?:to\s+)?|directions?\s+to|route\s+to|how\s+far\s+is|how\s+long\s+(?:does\s+it\s+take\s+(?:to\s+reach)?|to\s+reach)?|to)\s+(.+?)\s+from\s+(.+?)$",
                stripped_q,
                re.IGNORECASE
            )
            if m_to_from:
                dest_raw, orig_raw = m_to_from.group(1), m_to_from.group(2)

        # Pattern D: '{action/prefix} from {orig} to {dest} {suffix}'
        # e.g., 'give me bus routes from Hyderabad to Vemulavada', 'Directions from Secunderabad to Basara'
        if not orig_raw or not dest_raw:
            m_from_to = re.search(
                r"\bfrom\s+(.+?)\s+to\s+(.+?)$",
                stripped_q,
                re.IGNORECASE
            )
            if m_from_to:
                orig_raw, dest_raw = m_from_to.group(1), m_from_to.group(2)

        # Pattern D1: '{orig} to {dest}' directly
        # e.g., 'Hyderabad to Medaram', 'Warangal to Medaram driving', 'Hyderabad, Telangana to Medaram Sammakka Saralamma Temple'
        if not orig_raw or not dest_raw:
            m_direct_to = re.search(
                r"^(?:(?:give\s+me\s+|show\s+(?:me\s+)?)?(?:driving\s+|car\s+|bus\s+)?(?:routes?|directions?|distance)?\s*(?:for|of|from)?\s*)?([A-Za-z0-9\s,.-]+?)\s+to\s+([A-Za-z0-9\s,.-]+?)(?:\s+(?:driving|drive|by\s+road|by\s+car|by\s+bus|transit|walking|walk|by\s+foot))?$",
                stripped_q,
                re.IGNORECASE
            )
            if m_direct_to:
                cand_orig = m_direct_to.group(1).strip()
                cand_dest = m_direct_to.group(2).strip()
                if not re.search(r"^(?:how|directions?|routes?|distance|travel\s+time|what\s+is|give|show|can\s+i|how\s+do\s+i)\b", cand_orig, re.IGNORECASE):
                    orig_raw, dest_raw = cand_orig, cand_dest

        # Pattern E: '{dest} from {orig}' fallback
        if not orig_raw or not dest_raw:
            m_simple = re.search(r"(.+?)\s+from\s+(.+?)$", stripped_q, re.IGNORECASE)
            if m_simple:
                cand_dest = m_simple.group(1)
                dest_raw, orig_raw = cand_dest, m_simple.group(2)

        # Pattern F: Single location check (e.g., 'How far is Yadadri?', 'Directions to Basara', 'route map of Yadadri')
        if not orig_raw and not dest_raw:
            m_single = re.search(
                r"(?:how\s+far\s+is|directions?\s+(?:to|of|for)|route\s+(?:map\s+)?(?:of|for|to)|map\s+route\s+(?:of|for|to)|travel\s+time\s+to|driving\s+time\s+to)\s+(.+?)$",
                stripped_q,
                re.IGNORECASE
            )
            if m_single:
                dest_raw = m_single.group(1).strip()

        # Filter metric noise words mistakenly assigned as destination or origin
        METRIC_NOISE_WORDS = {
            "directions", "direction", "route", "routes", "distance",
            "travel time", "driving time", "how far", "eta", "estimated arrival"
        }
        if dest_raw and dest_raw.strip().lower() in METRIC_NOISE_WORDS:
            dest_raw = None
        if orig_raw and orig_raw.strip().lower() in METRIC_NOISE_WORDS:
            orig_raw = None

        # Clean noise words
        orig_clean = self._clean_place_name(orig_raw) if orig_raw else None
        dest_clean = self._clean_place_name(dest_raw) if dest_raw else None

        # Clean repeated origin or lingering '... to' prefixes in destination
        # E.g. 'Give me the route from Hyderabad to hyderabad to medaram'
        if dest_clean and orig_clean:
            # If destination starts with origin (case-insensitive) followed by 'to'
            pattern_orig_to = re.compile(rf"^(?:from\s+)?{re.escape(orig_clean.strip())}\s+(?:to\s+)+", re.IGNORECASE)
            dest_clean = pattern_orig_to.sub("", dest_clean).strip()

            # If destination still contains '... to <actual_dest>'
            m_nested_to = re.search(r"(?:^|.+?\s+)to\s+([A-Za-z0-9\s()]+)$", dest_clean, re.IGNORECASE)
            if m_nested_to:
                cand_dest = m_nested_to.group(1).strip()
                if cand_dest and cand_dest.lower() != orig_clean.lower():
                    dest_clean = cand_dest

        # Clean leading 'to ' or 'from ' that might remain on destination
        if dest_clean:
            dest_clean = re.sub(r"^(?:to|from|route\s+to|directions?\s+to)\s+", "", dest_clean, flags=re.IGNORECASE).strip()

        return {
            "origin": orig_clean,
            "destination": dest_clean,
            "travel_mode": travel_mode
        }

    def _clean_place_name(self, text: Optional[str]) -> Optional[str]:
        """Cleans extraneous filler words while preserving canonical identifiers."""
        if not text:
            return None
        t = text.strip()
        # Remove trailing question marks or punctuation
        t = re.sub(r"[?!.,;]+$", "", t).strip()

        # Remove leading conversational / intent filler prefixes
        t = re.sub(
            r"^(?:(?:i\s+want\s+to\s+)?(?:travel|go|drive|head)\s+(?:to|from)?|how\s+(?:do\s+i|can\s+i|to)\s+(?:get|reach|go|travel)\s+(?:to|from)?|give\s+me\s+(?:the\s+)?(?:bus\s+|driving\s+|car\s+)?(?:routes?|directions?|distance)\s+(?:to|from)?|show\s+me\s+(?:the\s+)?(?:bus\s+|driving\s+|car\s+)?(?:routes?|directions?|distance)\s+(?:to|from)?|tell\s+me\s+(?:the\s+)?(?:bus\s+|driving\s+|car\s+)?(?:routes?|directions?|distance)\s+(?:to|from)?|what\s+is\s+the\s+(?:distance|route|driving\s+time|travel\s+time)\s+(?:of|for|between|from|to)?|how\s+far\s+is|directions?\s+(?:to|from|for|of)?|routes?\s+(?:to|from|for|of)?|reach|get\s+to|travel\s+to|travel\s+from|visit|the|a|an)\s+",
            "",
            t,
            flags=re.IGNORECASE
        ).strip()

        # Remove transport mode phrases anywhere in text (e.g. 'by road', 'by car', 'by bus', 'by train', 'by bike', 'by two wheeler')
        t = re.sub(r"\bby\s+(?:car|walk|bike|cycle|transit|driving|foot|bus|train|road|cab|taxi|flight|two[-\s]?wheeler|two_wheeler|motorcycle|scooter)\b", "", t, flags=re.IGNORECASE).strip()

        # Remove trailing explanatory clauses or intent clauses
        t = re.sub(
            r"\s+(?:and|&)?\s*(?:want|planning|looking)\s+to\s+(?:visit|reach|go\s+to|travel\s+to|see)?.*$",
            "",
            t,
            flags=re.IGNORECASE
        ).strip()
        t = re.sub(
            r"\s+(?:and|&)\s+(?:explain|tell\s+me|tell|describe|give\s+me|give|provide|show\s+me|show|detail)\b.*$",
            "",
            t,
            flags=re.IGNORECASE
        ).strip()

        # Remove trailing noise words and filler suffixes
        t = re.sub(
            r"\s+(?:give\s+me\s+(?:bus\s+|driving\s+|car\s+)?(?:routes?|directions?)|show\s+(?:me\s+)?(?:bus\s+|driving\s+|car\s+)?(?:routes?|directions?)|bus\s+routes?|driving\s+routes?|routes?|directions?|bus\s+timings?|please|thanks|thank\s+you)+$",
            "",
            t,
            flags=re.IGNORECASE
        ).strip()

        # Remove any lingering leading articles
        t = re.sub(r"^(the|a|an)\s+", "", t, flags=re.IGNORECASE).strip()
        # Remove multiple spaces
        t = re.sub(r"\s+", " ", t).strip()
        return t if t else None

    def _get_temple_json(self, temple_name: str, filename: str) -> Optional[Any]:
        """Loads authoritative temple JSON data from disk for canonical temples."""
        canonical = self.coordinate_registry.resolve_alias(temple_name) or temple_name
        folder_name = canonical.lower()
        root_dir = Path(__file__).resolve().parents[3]
        file_path = root_dir / folder_name / filename
        if not file_path.exists():
            file_path = Path(__file__).resolve().parent.parent.parent.parent / folder_name / filename
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _resolve_location(
        self,
        place_name: Optional[str],
        context_temple: Optional[str] = None
    ) -> Tuple[str, Optional[Tuple[float, float]], Optional[str]]:
        """
        Resolves a place string to:
        (display_name, coordinates_tuple_or_None, canonical_name_or_None).
        
        Uses TempleCoordinateRegistry for temples and aliases.
        Resolves relative references ('nearby attraction', 'railway station', 'airport')
        against context temple's verified records.
        Uses recognized cities/places or passes address string through to provider.
        """
        if not place_name:
            return "", None, None

        norm_p = place_name.strip().lower()

        # 0. Check Relative References if context temple is available
        canon_ctx = self.coordinate_registry.resolve_alias(context_temple) if context_temple else None
        if not canon_ctx and context_temple:
            # Maybe context_temple is already canonical or place_name itself has temple reference
            for t_name in self.coordinate_registry._coordinates:
                if t_name.lower() in context_temple.lower():
                    canon_ctx = t_name
                    break

        if canon_ctx:
            # A. Relative Attraction
            if norm_p in {
                "nearby attraction", "nearby attractions", "a nearby attraction",
                "attraction", "attractions", "nearby places", "nearby place",
                "nearby tourist spot", "tourist spot"
            }:
                attr_data = self._get_temple_json(canon_ctx, "nearby_attractions.json")
                if attr_data:
                    items = attr_data if isinstance(attr_data, list) else attr_data.get("attractions", [])
                    if items and isinstance(items[0], dict) and "name" in items[0]:
                        attr_name = items[0]["name"].strip()
                        return f"{attr_name} (near {canon_ctx})", None, None

            # B. Relative Railway Station
            if norm_p in {
                "railway station", "the railway station", "nearest railway station",
                "station", "train station", "railway", "rail station"
            }:
                travel_data = self._get_temple_json(canon_ctx, "travel.json")
                if travel_data and isinstance(travel_data, dict):
                    stn = None
                    if "nearest_railway_station" in travel_data and isinstance(travel_data["nearest_railway_station"], dict):
                        stn = travel_data["nearest_railway_station"].get("name")
                    elif "railway" in travel_data and isinstance(travel_data["railway"], dict):
                        stn = travel_data["railway"].get("station")
                    if stn:
                        return stn.strip(), None, None

            # C. Relative Airport
            if norm_p in {"airport", "the airport", "nearest airport", "flight"}:
                travel_data = self._get_temple_json(canon_ctx, "travel.json")
                if travel_data and isinstance(travel_data, dict):
                    apt = None
                    if "nearest_airport" in travel_data and isinstance(travel_data["nearest_airport"], dict):
                        apt = travel_data["nearest_airport"].get("name")
                    elif "airport" in travel_data and isinstance(travel_data["airport"], dict):
                        apt = travel_data["airport"].get("name")
                    if apt:
                        return apt.strip(), None, None

        # 1. Determine if input genuinely expresses temple intent
        is_temple_query = any(
            kw in norm_p for kw in [
                "temple", "mandir", "devasthanam", "gudi", "kovil", "shrine",
                "saraswathi", "yellamma", "mukteshwara", "mallikarjuna",
                "brahmaramba", "narasimha", "balaji", "ramappa", "jogulamba"
            ]
        )
        non_temple_tokens = [
            "railway", "station", "bus stand", "bus stop", "airport",
            "hotel", "resort", "restaurant", "cafe", "bhavan", "lodge",
            "road", "street", "nagar", "colony", "cross road", "junction"
        ]
        has_non_temple_token = any(tok in norm_p for tok in non_temple_tokens)

        # 0. Check if input is a known town without temple intent (e.g. "Basara" town)
        if norm_p in ("basara", "basar") and not is_temple_query:
            lat, lon, city_name = _REGIONAL_CITY_COORDINATES["basara"]
            return city_name, (lat, lon), None

        # 1. Check Temple Coordinate Registry (handles canonical names and all aliases)
        if not has_non_temple_token:
            temple_coord = self.coordinate_registry.get_coordinate(place_name)
            if temple_coord is not None:
                return (
                    temple_coord.display_name or temple_coord.canonical_name,
                    (temple_coord.latitude, temple_coord.longitude),
                    temple_coord.canonical_name
                )

        norm_clean = re.sub(r"[^\w\s]", " ", norm_p)
        tokens = set(norm_clean.split())

        # 1.1 Token / Substring containment in canonical temples and aliases
        if not has_non_temple_token:
            from app.agents.coordinate_registry import _CANONICAL_TEMPLE_ALIASES
            for canon, aliases in _CANONICAL_TEMPLE_ALIASES.items():
                if canon.lower() in tokens or any(a.lower() in norm_p for a in aliases):
                    c = self.coordinate_registry.get_coordinate(canon)
                    if c:
                        return (
                            c.display_name or c.canonical_name,
                            (c.latitude, c.longitude),
                            c.canonical_name
                        )

        # 2. Check Known Regional Cities / Places (returns verified coordinates)
        if norm_p in _REGIONAL_CITY_COORDINATES:
            lat, lon, city_name = _REGIONAL_CITY_COORDINATES[norm_p]
            if norm_p in ("hyderabad", "secunderabad"):
                return city_name, None, None
            return city_name, (lat, lon), None

        # Regional city with regional qualifier (e.g. "Hyderabad City", "Warangal, Telangana")
        qualifiers = {"city", "town", "telangana", "india", "district", "dist", "area"}
        for k, (lat, lon, city_name) in _REGIONAL_CITY_COORDINATES.items():
            if k in tokens:
                other_words = tokens - {k} - qualifiers
                if not other_words:
                    if k in ("hyderabad", "secunderabad"):
                        return city_name, None, None
                    return city_name, (lat, lon), None

        # 3. Fallback: Title case address string for Routes API geocoding
        title_case = place_name.strip().title()
        return title_case, None, None

    def _format_markdown_response(self, result: NavigationResult) -> str:
        """Formats the NavigationResult into a structured Section Markdown response."""
        orig_disp = result.origin_canonical or result.origin
        dest_disp = result.destination_canonical or result.destination

        lines = [
            f"### Live Route Navigation: {orig_disp} → {dest_disp}",
            "",
            "#### Route Facts"
        ]

        # 1. Distance
        if result.distance_km is not None:
            lines.append(f"- **Distance**: {result.distance_km} km")
        else:
            lines.append("- **Distance**: Distance data not supplied by provider")

        # 2. Duration
        mode_label = result.travel_mode.replace("_", " ").title()
        if result.duration_minutes is not None:
            lines.append(f"- **Estimated Duration ({mode_label})**: {format_duration(result.duration_minutes)}")
        else:
            lines.append(f"- **Estimated Duration ({mode_label})**: Duration data not supplied by provider")

        # 3. Traffic Duration
        if result.traffic_duration_minutes is not None:
            lines.append(f"- **Traffic-Aware Duration**: {format_duration(result.traffic_duration_minutes)}")
        else:
            lines.append("- **Traffic-Aware Duration**: Live traffic information was not available from the configured provider.")

        # 4. Route summary / highway
        if result.route_summary:
            lines.append(f"- **Primary Route**: {result.route_summary}")

        lines.extend([
            "",
            "#### Journey Details",
            f"- **Origin**: {orig_disp}",
            f"- **Destination**: {dest_disp}",
            f"- **Travel Mode**: {mode_label}",
            f"- **Provider**: {result.provider}"
        ])

        if result.retrieved_at:
            lines.append(f"- **Retrieved At**: {result.retrieved_at}")

        lines.extend([
            "",
            "> **Note**: Navigation details are retrieved live from Google Maps Routes API. "
            "Actual travel times, road conditions, tolls, and detours may vary depending on local traffic, weather, and devasthanam parking regulations."
        ])

        return "\n".join(lines)

    def _run(self, state: AgentState) -> AgentResponse:
        """
        Executes live navigation query resolution via BaseAgent.execute lifecycle.
        """
        t0 = time.perf_counter()
        raw_query = state.user_query.strip()

        # 1. Empty input validation
        if not raw_query:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Please provide a valid navigation or route question.",
                sources=[],
                error="Empty query",
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )

        # 2. Extract Route Parameters
        params = self.extract_route_parameters(raw_query)
        ext_params = getattr(state, "extracted_params", None) or {}
        struct_constraints = getattr(state, "structured_constraints", None) or {}
        origin_str = ext_params.get("origin") or params.get("origin")
        dest_str = ext_params.get("destination") or params.get("destination")
        raw_mode = (
            ext_params.get("travel_mode") or
            struct_constraints.get("travel_mode") or
            params.get("travel_mode") or
            "driving"
        )
        travel_mode = str(raw_mode).strip().lower()
        if travel_mode in ("two-wheeler", "two wheeler", "motorcycle", "bike", "scooter"):
            travel_mode = "two_wheeler"

        # Origin Resolution Priority:
        # Priority 1: Explicitly provided in current request (origin_str)
        # Priority 2: Valid origin already associated with current trip or conversation session
        if not origin_str and self.memory_manager and state.session_id:
            origin_str = self.memory_manager.get_session_origin(state.session_id)
        if not origin_str and state.structured_constraints.get("origin"):
            origin_str = state.structured_constraints.get("origin")

        # Priority 3: Device location only if explicitly granted by user
        if not origin_str and state.structured_constraints.get("device_location"):
            origin_str = state.structured_constraints.get("device_location")

        # Priority 4: If origin or destination is missing, request manual input
        # Never silently assume Hyderabad or any other location!
        if not origin_str or not dest_str:
            target_temple = dest_str or origin_str
            if not origin_str and dest_str:
                clarification_msg = (
                    f"Where are you starting from? Enter your starting location to see the route to {dest_str}. "
                    f"Please provide a route with a recognizable origin and destination, e.g., 'route from Hyderabad to {dest_str}'."
                )
            elif not dest_str and origin_str:
                clarification_msg = (
                    f"Where are you traveling to? Enter your destination to see the route from {origin_str}. "
                    f"Please provide a route with a recognizable origin and destination, e.g., 'route from {origin_str} to Yadadri'."
                )
            else:
                clarification_msg = (
                    "Where are you starting from? Enter your starting location and destination to see the route. "
                    "Please provide a route with a recognizable origin and destination, e.g., 'route from Hyderabad to Yadadri'."
                )
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer=clarification_msg,
                sources=[],
                error="Missing origin or destination",
                metadata={
                    "clarification_needed": True,
                    "missing_origin": not bool(origin_str),
                    "missing_destination": not bool(dest_str),
                    "destination": dest_str or target_temple or "",
                    "origin": origin_str or "",
                    "travel_mode": travel_mode,
                    "extracted_params": params
                },
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )

        # Retain verified origin in session memory for multi-turn navigation
        if self.memory_manager and state.session_id and origin_str:
            self.memory_manager.set_session_origin(state.session_id, origin_str)

        # 4. Resolve locations to coordinates or verified names (with reciprocal context)
        orig_name, orig_coords, orig_canon = self._resolve_location(origin_str, dest_str)
        dest_name, dest_coords, dest_canon = self._resolve_location(dest_str, origin_str)

        # In case dest resolved to canonical and orig was relative (e.g. orig="railway station", dest="Veera Satyanarayana")
        if not orig_canon and dest_canon:
            orig_name, orig_coords, orig_canon = self._resolve_location(origin_str, dest_canon)
        if not dest_canon and orig_canon:
            dest_name, dest_coords, dest_canon = self._resolve_location(dest_str, orig_canon)

        # Overwrite with explicit structured constraints coordinates if injected
        if state.structured_constraints.get("origin_coordinates"):
            raw_oc = state.structured_constraints["origin_coordinates"]
            if isinstance(raw_oc, (list, tuple)) and len(raw_oc) >= 2:
                orig_coords = (float(raw_oc[0]), float(raw_oc[1]))
        if state.structured_constraints.get("destination_coordinates"):
            raw_dc = state.structured_constraints["destination_coordinates"]
            if isinstance(raw_dc, (list, tuple)) and len(raw_dc) >= 2:
                dest_coords = (float(raw_dc[0]), float(raw_dc[1]))

        orig_waypoint: Union[Tuple[float, float], str] = orig_coords if orig_coords else orig_name
        dest_waypoint: Union[Tuple[float, float], str] = dest_coords if dest_coords else dest_name

        # 5. Check zero-credential condition upfront
        if not self.adapter.has_credentials and self.adapter.transport is None:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Live navigation is currently unavailable because no navigation provider credential is configured.",
                sources=[],
                error="Navigation provider credential not configured",
                metadata={
                    "provider": "Google Maps Routes API",
                    "credential_configured": False,
                    "origin": origin_str,
                    "destination": dest_str
                },
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )

        # 6. Check cache if enabled
        cache_key = f"{origin_str.lower()}::{dest_str.lower()}::{travel_mode}"
        now_ts = time.time()
        if self.cache_enabled and cache_key in self._cache:
            cache_time, cached_result = self._cache[cache_key]
            if (now_ts - cache_time) <= self.cache_ttl_sec:
                result_copy = cached_result.model_copy()
                result_copy.cache_hit = True
                answer_md = self._format_markdown_response(result_copy)
                return AgentResponse(
                    success=True,
                    agent_name=self.name,
                    answer=answer_md,
                    sources=["Google Maps Routes API"],
                    metadata=result_copy.model_dump(),
                    latency_ms=(time.perf_counter() - t0) * 1000.0
                )

        # 7. Query NavigationAdapter
        try:
            route_data = self.adapter.compute_route(
                origin=orig_waypoint,
                destination=dest_waypoint,
                travel_mode=travel_mode
            )
        except NavigationCredentialMissingError:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Live navigation is currently unavailable because no navigation provider credential is configured.",
                sources=[],
                error="Navigation provider credential not configured",
                metadata={"credential_configured": False},
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )
        except UnsupportedTravelModeError as e:
            if travel_mode == "two_wheeler":
                unsupported_msg = "Two-Wheeler routing is not available for this corridor or with current provider settings. Please use Driving or Transit."
            else:
                unsupported_msg = f"The requested travel mode '{travel_mode}' is not supported by the navigation provider. Supported modes include: driving, walking, bicycling, transit, two_wheeler."
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer=unsupported_msg,
                sources=[],
                error=str(e),
                metadata={"travel_mode": travel_mode, "unsupported_mode": True},
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )
        except RouteNotFoundError as e:
            if travel_mode == "two_wheeler":
                not_found_msg = f"No Two-Wheeler route could be found between '{origin_str}' and '{dest_str}'. Please try Driving or Transit."
            else:
                not_found_msg = f"No route could be found between '{origin_str}' and '{dest_str}'. Please check the locations and try again."
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer=not_found_msg,
                sources=[],
                error=str(e),
                metadata={"origin": origin_str, "destination": dest_str, "route_not_found": True, "travel_mode": travel_mode},
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )
        except ExternalTimeoutError:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Navigation request timed out. Please check your network connection and try again.",
                sources=[],
                error="Navigation request timed out",
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )
        except ExternalConnectionError:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Navigation provider is currently unreachable. Please try again shortly.",
                sources=[],
                error="Navigation provider connection failure",
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )
        except ExternalRateLimitError:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Navigation request rate limit reached. Please wait a moment before retrying.",
                sources=[],
                error="Rate limit exceeded",
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )
        except ExternalCircuitOpenError:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Navigation service is temporarily unavailable due to repeated errors. Please retry later.",
                sources=[],
                error="Circuit breaker OPEN",
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )
        except ExternalHTTPError as e:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Navigation provider authentication or service error. Please verify provider configuration.",
                sources=[],
                error=f"HTTP {e.status_code}: {e.raw_message}",
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )
        except ExternalAdapterError as e:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Live navigation is currently unavailable due to an unexpected provider error.",
                sources=[],
                error=str(e),
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )
        except Exception as e:
            logger.error(f"Unexpected error in LiveNavigationAgent: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="An unexpected error occurred while calculating the route.",
                sources=[],
                error=str(e),
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )

        # 8. Parse Provider Route Data
        routes = route_data.get("routes", [])
        if not routes:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer=f"No route could be found between '{origin_str}' and '{dest_str}'.",
                sources=[],
                error="Empty routes array",
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )

        primary_route = routes[0]

        # Distance extraction
        distance_km: Optional[float] = None
        if "distanceMeters" in primary_route:
            try:
                distance_km = round(float(primary_route["distanceMeters"]) / 1000.0, 1)
            except (ValueError, TypeError):
                distance_km = None

        # Duration extraction
        duration_minutes: Optional[float] = None
        traffic_duration_minutes: Optional[float] = None

        # Standard duration
        duration_raw = primary_route.get("duration")
        if duration_raw:
            dur_sec = self._parse_duration_seconds(duration_raw)
            if dur_sec is not None:
                duration_minutes = round(dur_sec / 60.0, 1)

        # Static vs Traffic Duration
        static_duration_raw = primary_route.get("staticDuration")
        if static_duration_raw:
            static_sec = self._parse_duration_seconds(static_duration_raw)
            if static_sec is not None and duration_minutes is not None:
                static_mins = round(static_sec / 60.0, 1)
                # If duration differs from staticDuration, duration contains traffic!
                if abs(duration_minutes - static_mins) >= 0.5:
                    traffic_duration_minutes = duration_minutes
                    duration_minutes = static_mins

        # Route summary
        route_summary = primary_route.get("description")

        # Construct NavigationResult
        polyline_val = primary_route.get("polyline", {}).get("encodedPolyline")
        iso_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts))

        # Explicit route state calculation
        if distance_km is not None and distance_km > 0 and duration_minutes is not None and duration_minutes > 0:
            if polyline_val:
                route_status = "SUCCESS"
            else:
                route_status = "PARTIAL_DATA"
        elif distance_km is not None or duration_minutes is not None:
            route_status = "PARTIAL_DATA"
        else:
            route_status = "NO_ROUTE"

        result = NavigationResult(
            origin=orig_name or origin_str,
            destination=dest_name or dest_str,
            origin_canonical=orig_canon,
            destination_canonical=dest_canon,
            origin_coordinates=orig_coords,
            destination_coordinates=dest_coords,
            polyline=polyline_val,
            distance_km=distance_km,
            duration_minutes=duration_minutes,
            traffic_duration_minutes=traffic_duration_minutes,
            route_summary=route_summary,
            travel_mode=travel_mode,
            provider="Google Maps Routes API",
            retrieved_at=iso_time,
            cache_hit=False,
            route_status=route_status
        )

        if self.cache_enabled:
            self._cache[cache_key] = (now_ts, result)

        answer_text = self._format_markdown_response(result)

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer=answer_text,
            sources=["Google Maps Routes API"],
            metadata=result.model_dump(),
            latency_ms=(time.perf_counter() - t0) * 1000.0
        )

    def _parse_duration_seconds(self, dur_val: Any) -> Optional[float]:
        """Parses Google API duration format (e.g. '3600s' or integer seconds)."""
        if dur_val is None:
            return None
        if isinstance(dur_val, (int, float)):
            return float(dur_val)
        if isinstance(dur_val, str):
            s = dur_val.strip().rstrip("sS")
            try:
                return float(s)
            except ValueError:
                return None
        return None

    def process(self, state: AgentState) -> AgentResponse:
        """Alias for BaseAgent.execute for coordinator consistency."""
        return self.execute(state)
