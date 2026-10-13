"""
PilgrimAI - Live Weather Agent (Phase O3)

Fetches live/current and forecast weather information for canonical pilgrimage temples
using the Open-Meteo HTTP API through BaseExternalAdapter (O1 Firewall) and
TempleCoordinateRegistry (O2 Registry).

Core Principles:
- Provider: Open-Meteo (Free, CC BY 4.0 attribution, zero API keys required).
- Zero Google API key / secrets usage or leakage.
- Zero direct HTTP requests; all traffic routes through BaseExternalAdapter.
- Zero RAG / Qdrant / BM25 / embedding model imports.
- Zero fabricated weather: failures return controlled unavailable responses.
- Clear separation between factual weather observations and practical pilgrimage advice.
- Deterministic WMO weather-code normalization.
- In-process caching with explicit cache_hit and retrieved_at metadata.
- Timezone: Asia/Kolkata.
"""

import time
import re
import datetime
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

from app.agents.base_agent import BaseAgent
from app.agents.state import AgentState, AgentResponse
from app.agents.adapters.external_adapter import (
    BaseExternalAdapter,
    ExternalAdapterError,
    ExternalTimeoutError,
    ExternalConnectionError,
    ExternalHTTPError,
    ExternalRateLimitError,
    ExternalCircuitOpenError
)
from app.agents.coordinate_registry import TempleCoordinateRegistry, TempleCoordinate

logger = logging.getLogger(__name__)

# Open-Meteo Attribution
OPEN_METEO_ATTRIBUTION = "Weather data by Open-Meteo.com under CC BY 4.0"

# Deterministic WMO Weather Code Interpretations
WMO_WEATHER_CODE_DESCRIPTIONS: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}


def get_weather_condition_description(code: Any) -> str:
    """Deterministically maps numeric WMO weather codes to human-readable descriptions."""
    if code is None:
        return "Unknown weather condition"
    try:
        int_code = int(code)
        return WMO_WEATHER_CODE_DESCRIPTIONS.get(int_code, "Unknown weather condition")
    except (ValueError, TypeError):
        return "Unknown weather condition"


class LiveWeatherAgent(BaseAgent):
    """
    Specialist agent for retrieving live weather observations and pilgrimage forecasts.
    Queries Open-Meteo through BaseExternalAdapter and TempleCoordinateRegistry.
    """

    def __init__(
        self,
        coordinate_registry: Optional[TempleCoordinateRegistry] = None,
        adapter: Optional[BaseExternalAdapter] = None,
        cache_ttl_sec: float = 600.0,
        cache_enabled: bool = True
    ):
        super().__init__(
            name="live_weather",
            description="Fetches live weather observations and pilgrimage forecasts using Open-Meteo."
        )
        self.coordinate_registry = coordinate_registry or TempleCoordinateRegistry()
        self.adapter = adapter or BaseExternalAdapter(name="open_meteo")
        self.cache_ttl_sec = max(0.0, float(cache_ttl_sec))
        self.cache_enabled = cache_enabled
        # In-memory cache: key -> (cached_monotonic_time, cached_iso_time, raw_json_data)
        self._cache: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}

    def clear_cache(self) -> None:
        """Clears the in-memory weather cache."""
        self._cache.clear()

    def _resolve_all_temples(self, state: AgentState) -> List[TempleCoordinate]:
        """
        Resolves all target temples and their coordinates from AgentState or query text.
        Returns deduplicated list of TempleCoordinate in order of appearance.
        """
        resolved: List[TempleCoordinate] = []
        resolved_names: Set[str] = set()

        # 1. Check routing metadata
        route_decision = state.intermediate_results.get("route_decision")
        if route_decision:
            metadata = getattr(route_decision, "metadata", {}) or {}
            detected_temples = metadata.get("detected_temples") or []
            if metadata.get("detected_temple") and metadata.get("detected_temple") not in detected_temples:
                detected_temples.insert(0, metadata.get("detected_temple"))

            for t in detected_temples:
                coord = self.coordinate_registry.get_coordinate(t)
                if coord and coord.canonical_name not in resolved_names:
                    resolved.append(coord)
                    resolved_names.add(coord.canonical_name)

        # 2. Inspect query directly using coordinate registry aliases
        query = state.user_query or ""
        q_norm = query.lower()

        all_coords = self.coordinate_registry.get_all_coordinates()
        for canonical_name, coord in all_coords.items():
            if re.search(r"\b" + re.escape(canonical_name.lower()) + r"\b", q_norm):
                if coord.canonical_name not in resolved_names:
                    resolved.append(coord)
                    resolved_names.add(coord.canonical_name)

        from app.agents.coordinate_registry import _CANONICAL_TEMPLE_ALIASES
        for canonical_id, aliases in _CANONICAL_TEMPLE_ALIASES.items():
            for alias in sorted(aliases, key=len, reverse=True):
                if re.search(r"\b" + re.escape(alias.lower()) + r"\b", q_norm):
                    coord = self.coordinate_registry.get_coordinate(canonical_id)
                    if coord and coord.canonical_name not in resolved_names:
                        resolved.append(coord)
                        resolved_names.add(coord.canonical_name)
                    break

        return resolved

    def _extract_temperature_filter(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Deterministically extracts temperature threshold or range filter criteria from query.
        Handles °C, ° C, degrees, deg, hyphenated ranges, and word forms.
        """
        q_norm = query.lower()

        # Between / Range: e.g. "between 25°C and 27°C", "between 27°C and 30°C?", "between 25 and 30", "25 to 30", "25-30", "25–27"
        range_match = re.search(
            r"(?:between\s+)?(\d{1,2})(?:\s*°?\s*c|\s*degrees?)?\s*(?:[-–—]|to|and)\s*(\d{1,2})\s*(?:°?\s*c|degrees?|deg)?",
            q_norm
        )
        if range_match and ("between" in q_norm or any(w in q_norm for w in ["temperature", "temperatures", "weather", "deg", "degree", "°c", "°", "celsius"])):
            min_temp = float(range_match.group(1))
            max_temp = float(range_match.group(2))
            return {
                "type": "between",
                "min_temp": min(min_temp, max_temp),
                "max_temp": max(min_temp, max_temp),
                "description": f"between {min(min_temp, max_temp):g}°C and {max(min_temp, max_temp):g}°C"
            }

        # Above / Greater than: e.g. "above 27°c", "above 30 degrees", "more than 28", "higher than 27"
        above_match = re.search(
            r"(?:above|over|higher\s+than|more\s+than|exceeding|greater\s+than)\s*(\d{1,2})\s*(?:°?\s*c|degrees?|deg)?\b",
            q_norm
        )
        if above_match:
            thresh = float(above_match.group(1))
            return {
                "type": "above",
                "threshold": thresh,
                "description": f"above {thresh:g}°C"
            }

        # Below / Under / Less than: e.g. "below 25°c", "under 28 degrees", "less than 26", "below 27 degrees"
        below_match = re.search(
            r"(?:below|under|less\s+than|lower\s+than)\s*(\d{1,2})\s*(?:°?\s*c|degrees?|deg)?\b",
            q_norm
        )
        if below_match:
            thresh = float(below_match.group(1))
            return {
                "type": "below",
                "threshold": thresh,
                "description": f"below {thresh:g}°C"
            }

        # Around / Approx / About: e.g. "around 26°c", "approx 28 degrees", "about 25"
        around_match = re.search(
            r"(?:around|approx|about)\s*(\d{1,2})\s*(?:°?\s*c|degrees?|deg)?\b",
            q_norm
        )
        if around_match:
            target = float(around_match.group(1))
            return {
                "type": "around",
                "target": target,
                "min_temp": target - 2.5,
                "max_temp": target + 2.5,
                "description": f"around {target:g}°C ({target-2.5:g}°C – {target+2.5:g}°C)"
            }

        return None

    def _resolve_temple(self, state: AgentState) -> Optional[TempleCoordinate]:
        """
        Resolves the primary target temple and its coordinates from AgentState or query text.
        Returns a verified TempleCoordinate or None if unresolvable.
        """
        coords = self._resolve_all_temples(state)
        return coords[0] if coords else None

    def _fetch_open_meteo(
        self,
        latitude: float,
        longitude: float,
        cache_key: str
    ) -> Tuple[Dict[str, Any], bool, str]:
        """
        Retrieves forecast data from cache or directly through BaseExternalAdapter.
        Returns (raw_json_dict, cache_hit_bool, retrieved_at_iso).
        """
        now_mono = time.monotonic()

        # Check cache
        if self.cache_enabled and cache_key in self._cache:
            cached_mono, cached_iso, cached_data = self._cache[cache_key]
            if (now_mono - cached_mono) <= self.cache_ttl_sec:
                return cached_data, True, cached_iso

        # Dispatch through external adapter
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m,wind_direction_10m",
            "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,sunrise,sunset",
            "timezone": "Asia/Kolkata",
            "forecast_days": 7
        }

        resp = self.adapter.request("GET", url, params=params)
        raw_data = resp.json()

        retrieved_at_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if self.cache_enabled:
            self._cache[cache_key] = (now_mono, retrieved_at_iso, raw_data)

        return raw_data, False, retrieved_at_iso

    def _generate_pilgrimage_advice(
        self,
        current_temp: Optional[float],
        condition_code: Optional[int],
        precip_sum_today: Optional[float],
        precip_prob_today: Optional[int],
        max_temp_today: Optional[float],
        min_temp_today: Optional[float]
    ) -> List[str]:
        """
        Deterministically produces conservative, practical pilgrimage guidance
        derived strictly from observed weather metrics.
        """
        advice = []

        # Rain / Precipitation
        has_rain_forecast = (
            (precip_sum_today is not None and precip_sum_today > 0.5) or
            (precip_prob_today is not None and precip_prob_today >= 40) or
            (condition_code is not None and condition_code in [51, 53, 55, 61, 63, 65, 80, 81, 82])
        )
        if has_rain_forecast:
            advice.append("Precipitation or rain showers are forecast; carrying an umbrella or light raincoat is recommended.")

        # Heat / Sun protection
        if (max_temp_today is not None and max_temp_today >= 34.0) or (current_temp is not None and current_temp >= 33.0):
            advice.append("Elevated daytime temperatures are expected; stay hydrated and consider planning darshan for early morning or late afternoon.")

        # Cold / Morning chill
        if (min_temp_today is not None and min_temp_today <= 17.0) or (current_temp is not None and current_temp <= 16.0):
            advice.append("Cool temperatures expected during morning/evening hours; light layering or a shawl is recommended for early darshan.")

        # Fog / Visibility
        if condition_code is not None and condition_code in [45, 48]:
            advice.append("Foggy conditions may reduce morning travel visibility; please allow extra transit time if traveling by road.")

        # Thunderstorm safety
        if condition_code is not None and condition_code in [95, 96, 99]:
            advice.append("Thunderstorms are possible; seek indoor temple shelter during lightning and monitor local travel advisories.")

        # Standard general pilgrim advice
        advice.append("Follow standard temple dress codes and verify queue timings prior to arrival.")

        return advice

    def _build_targeted_answers(
        self,
        query: str,
        coord: TempleCoordinate,
        current_temp: Optional[float],
        condition_desc: str,
        weather_code: Optional[int],
        current_precip: float,
        current_rain: float,
        current_wind: Optional[float],
        precip_sum_today: Optional[float],
        precip_prob_today: Optional[int],
        max_temp_today: Optional[float]
    ) -> List[str]:
        q_lower = query.lower()
        targeted_lines = []

        # 1. Umbrella query
        if re.search(r"\b(umbrella|carry\s+(?:an\s+)?umbrella|raincoat)\b", q_lower):
            has_rain = (
                (precip_sum_today is not None and precip_sum_today > 0.2) or
                (current_precip > 0.0) or
                (precip_prob_today is not None and precip_prob_today >= 30) or
                (weather_code is not None and weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99])
            )
            prob_str = f"{precip_prob_today}%" if precip_prob_today is not None else "low"
            sum_str = f"{precip_sum_today or current_precip} mm"
            if has_rain:
                targeted_lines.append(
                    f"- **Should you carry an umbrella?**: **Yes, carrying an umbrella or light raincoat is recommended.** "
                    f"The forecast indicates rain or precipitation ({sum_str} expected, {prob_str} probability, condition: {condition_desc})."
                )
            else:
                heat_note = f" However, with highs of {max_temp_today or current_temp}°C, a compact umbrella or parasol is advisable for sun protection during open-air darshan queues." if (max_temp_today or current_temp or 0) >= 32.0 else ""
                targeted_lines.append(
                    f"- **Should you carry an umbrella?**: **Not strictly required for rain.** "
                    f"Current forecast shows {condition_desc} with minimal precipitation probability ({prob_str}, {sum_str}).{heat_note}"
                )

        # 2. Rain affecting travel / Road transit query
        if re.search(r"\b(rain\s+(?:affect|impact|delay|disrupt|cause)|travel\s+(?:impact|affect|safe|delayed|delays?)|drive|road\s+conditions?|highway\s+conditions?)\b", q_lower):
            prob_str = f"{precip_prob_today}%" if precip_prob_today is not None else "low"
            sum_str = f"{precip_sum_today or current_precip} mm"
            has_heavy_rain = (
                (precip_sum_today is not None and precip_sum_today > 5.0) or
                (weather_code is not None and weather_code in [63, 65, 81, 82, 95, 96, 99])
            )
            if has_heavy_rain:
                targeted_lines.append(
                    f"- **Will rain affect your travel?**: **Yes, travel may be moderately to significantly impacted.** "
                    f"Noticeable rainfall ({sum_str}, {condition_desc}) is expected, which can reduce highway visibility and create slippery driving conditions or local waterlogging. Allow extra travel time."
                )
            elif (precip_sum_today is not None and precip_sum_today > 0.5) or (current_precip > 0.0):
                targeted_lines.append(
                    f"- **Will rain affect your travel?**: **Minor impact possible.** Light showers ({sum_str}, {prob_str} prob) may cause wet road surfaces, but major delays are unlikely under current readings. Monitor local transit advisories."
                )
            else:
                targeted_lines.append(
                    f"- **Will rain affect your travel?**: **No significant rain impact is expected.** "
                    f"Current conditions indicate {condition_desc} ({sum_str} rain), making road conditions generally favorable. Note that Ghat road checkpoints and traffic congestion should still be verified locally."
                )

        # 3. Severe weather alert query
        if re.search(r"\b(severe\s+weather|alerts?|warnings?|cyclone|flood|storm\s+warning|emergency\s+weather)\b", q_lower):
            targeted_lines.append(
                f"- **Severe Weather Alert Status**: "
                f"**No severe weather alerts can be certified by this automated meteorological feed.**\n"
                f"  *Distinction between Forecast & Official Warnings*: Open-Meteo delivers numerical weather simulations and observations ({condition_desc}, wind: {current_wind} km/h, precipitation: {current_precip} mm). However, it does **not** broadcast official government disaster management warnings (such as IMD Red/Orange alerts, flood bulletins, or state emergency declarations). "
                f"Even when standard weather metrics appear normal, official severe weather warnings must be confirmed directly via India Meteorological Department (IMD) or local district administration bulletins."
            )

        return targeted_lines

    def _run_cross_temple_threshold(
        self,
        temp_filter: Dict[str, Any],
        t0: float
    ) -> AgentResponse:
        """
        Executes cross-temple live temperature filtering across all registered devasthanams.
        """
        all_temples = self.coordinate_registry.get_all_coordinates()
        results = []
        for name, coord in all_temples.items():
            if coord.latitude is not None and coord.longitude is not None:
                cache_key = f"{coord.canonical_name.lower()}:7"
                try:
                    raw_data, is_hit, iso_time = self._fetch_open_meteo(coord.latitude, coord.longitude, cache_key)
                    current = raw_data.get("current") or {}
                    temp = current.get("temperature_2m")
                    wcode = current.get("weather_code")
                    humidity = current.get("relative_humidity_2m")
                    wind = current.get("wind_speed_10m")
                    if temp is not None:
                        results.append({
                            "coord": coord,
                            "temp": float(temp),
                            "condition": get_weather_condition_description(wcode),
                            "humidity": humidity,
                            "wind": wind,
                            "retrieved_at": iso_time
                        })
                except Exception as e:
                    logger.warning(f"Error fetching live weather for {coord.canonical_name}: {e}")

        # Filter by criteria
        f_type = temp_filter["type"]
        matched = []
        if f_type == "above":
            matched = [r for r in results if r["temp"] > temp_filter["threshold"]]
            matched.sort(key=lambda x: x["temp"], reverse=True)
        elif f_type == "below":
            matched = [r for r in results if r["temp"] < temp_filter["threshold"]]
            matched.sort(key=lambda x: x["temp"])
        elif f_type in ("between", "around"):
            matched = [r for r in results if temp_filter["min_temp"] <= r["temp"] <= temp_filter["max_temp"]]
            matched.sort(key=lambda x: x["temp"])

        lines = [
            f"## 🌤️ Live Weather Filter — Temples with Temperature {temp_filter['description']}",
            "",
            f"Based on **live observations from Open-Meteo**, the following registered pilgrimage temples currently match your criteria:\n"
        ]
        if matched:
            for i, m in enumerate(matched, 1):
                c = m["coord"]
                d_name = c.display_name or c.canonical_name
                lines.append(f"{i}. **{d_name}** ({c.canonical_name})")
                lines.append(f"   - **Current Temperature**: **{m['temp']}°C**")
                lines.append(f"   - **Condition**: {m['condition']} | **Humidity**: {m['humidity']}% | **Wind Speed**: {m['wind']} km/h")
        else:
            lines.append(f"None of the 20 registered pilgrimage temples currently have temperatures {temp_filter['description']}.")
            if results:
                available_summary = ", ".join([f"{r['coord'].canonical_name} ({r['temp']}°C)" for r in results[:6]])
                lines.append(f"\nCurrent live observations at monitored temples include: {available_summary}.")

        lines.extend([
            "",
            "### 💡 Practical Pilgrimage Advice",
            "- Check live queue and darshan timings prior to travel.",
            "- Weather conditions fluctuate throughout the day; prepare appropriate attire.",
            "",
            f"*{OPEN_METEO_ATTRIBUTION}*"
        ])

        answer_text = "\n".join(lines)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        metadata = {
            "provider": "Open-Meteo",
            "query_type": "cross_temple_threshold",
            "filter": temp_filter,
            "matched_count": len(matched),
            "matched_temples": [m["coord"].canonical_name for m in matched],
            "total_evaluated": len(results),
            "attribution": OPEN_METEO_ATTRIBUTION
        }

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer=answer_text,
            sources=[OPEN_METEO_ATTRIBUTION],
            latency_ms=latency_ms,
            metadata=metadata
        )

    def _run_multi_temple(
        self,
        coords: List[TempleCoordinate],
        t0: float
    ) -> AgentResponse:
        """
        Executes live weather retrieval for multiple explicitly requested temples.
        """
        results = []
        for coord in coords:
            if coord.latitude is not None and coord.longitude is not None:
                cache_key = f"{coord.canonical_name.lower()}:7"
                try:
                    raw_data, is_hit, iso_time = self._fetch_open_meteo(coord.latitude, coord.longitude, cache_key)
                    current = raw_data.get("current") or {}
                    daily = raw_data.get("daily") or {}
                    temp = current.get("temperature_2m")
                    humidity = current.get("relative_humidity_2m")
                    wind = current.get("wind_speed_10m")
                    wcode = current.get("weather_code")
                    cond_desc = get_weather_condition_description(wcode)

                    max_t = (daily.get("temperature_2m_max") or [None])[0]
                    min_t = (daily.get("temperature_2m_min") or [None])[0]

                    results.append({
                        "coord": coord,
                        "temp": temp,
                        "condition": cond_desc,
                        "humidity": humidity,
                        "wind": wind,
                        "min_t": min_t,
                        "max_t": max_t,
                        "retrieved_at": iso_time
                    })
                except Exception as e:
                    logger.warning(f"Error fetching live weather for {coord.canonical_name}: {e}")

        if not results:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Live weather data is currently unavailable for the requested temples.",
                sources=[],
                error="Failed to retrieve weather for multiple temples.",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={"provider": "Open-Meteo", "query_type": "multi_temple_weather"}
            )

        temple_names = [r["coord"].display_name or r["coord"].canonical_name for r in results]
        lines = [
            f"## 🌤️ Live Weather Report — {', '.join(temple_names)}",
            "",
            "### 🌡️ Current Live Observations",
            ""
        ]
        for i, r in enumerate(results, 1):
            c = r["coord"]
            d_name = c.display_name or c.canonical_name
            lines.append(f"{i}. **{d_name}** ({c.canonical_name})")
            lines.append(f"   - **Current Temperature**: **{r['temp']}°C**")
            lines.append(f"   - **Condition**: {r['condition']}")
            if r["min_t"] is not None and r["max_t"] is not None:
                lines.append(f"   - **Today's Range**: {r['min_t']}°C to {r['max_t']}°C")
            lines.append(f"   - **Humidity**: {r['humidity']}% | **Wind Speed**: {r['wind']} km/h")
            lines.append("")

        lines.extend([
            "### 💡 Practical Pilgrimage Advice",
            "- Stay hydrated during afternoon darshan hours and check queue conditions.",
            "- Layer with light shawls for morning/evening visits if temperatures drop.",
            "",
            f"*{OPEN_METEO_ATTRIBUTION}*"
        ])

        answer_text = "\n".join(lines)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        metadata = {
            "provider": "Open-Meteo",
            "query_type": "multi_temple_weather",
            "temples": [r["coord"].canonical_name for r in results],
            "observations": {
                r["coord"].canonical_name: {
                    "temperature": r["temp"],
                    "condition": r["condition"],
                    "humidity": r["humidity"],
                    "wind": r["wind"]
                }
                for r in results
            },
            "attribution": OPEN_METEO_ATTRIBUTION
        }

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer=answer_text,
            sources=[OPEN_METEO_ATTRIBUTION],
            latency_ms=latency_ms,
            metadata=metadata
        )

    def _run(self, state: AgentState) -> AgentResponse:
        """
        Main execution method conforming to BaseAgent lifecycle.
        """
        t0 = time.perf_counter()
        query = (state.user_query or "").strip()

        # 1. Validate empty / whitespace input
        if not query:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Please provide a valid question regarding temple weather.",
                sources=[],
                error="Empty or whitespace query provided.",
                latency_ms=0.0,
                metadata={"provider": "Open-Meteo", "error_type": "EmptyQuery"}
            )

        # 2. Check for Cross-Temple Temperature Filter Query
        temp_filter = self._extract_temperature_filter(query)
        all_resolved = self._resolve_all_temples(state)

        if temp_filter is not None and (len(all_resolved) == 0 or "which temples" in query.lower() or "temples" in query.lower()):
            return self._run_cross_temple_threshold(temp_filter, t0)

        # 3. Check for Multi-Temple Weather Request
        if len(all_resolved) > 1:
            return self._run_multi_temple(all_resolved, t0)

        # 4. Single-Temple Weather Request
        coord = all_resolved[0] if all_resolved else None
        if not coord:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Live weather information is unavailable: could not resolve a recognized pilgrimage temple from the query.",
                sources=[],
                error="Unresolvable or unregistered temple location.",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={"provider": "Open-Meteo", "supported": False, "location": "unknown"}
            )

        # 5. Coordinate Sanity Validation
        if coord.latitude is None or coord.longitude is None:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer=f"Live weather information is unavailable: coordinates for '{coord.canonical_name}' are missing.",
                sources=[],
                error="Missing temple coordinates.",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={"provider": "Open-Meteo", "location": coord.canonical_name, "error_type": "MissingCoordinates"}
            )

        # 4. Fetch Weather Data through BaseExternalAdapter
        cache_key = f"{coord.canonical_name.lower()}:7"
        try:
            raw_data, is_cache_hit, retrieved_at_iso = self._fetch_open_meteo(
                latitude=coord.latitude,
                longitude=coord.longitude,
                cache_key=cache_key
            )
        except ExternalAdapterError as e:
            logger.warning(f"External weather adapter error for {coord.canonical_name}: {e}")
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer=f"Live weather data for {coord.canonical_name} is currently unavailable. Please try again shortly.",
                sources=[],
                error=str(e),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={
                    "provider": "Open-Meteo",
                    "location": coord.canonical_name,
                    "error_type": type(e).__name__
                }
            )
        except Exception as e:
            logger.error(f"Unexpected error retrieving weather for {coord.canonical_name}: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer=f"Live weather data for {coord.canonical_name} is currently unavailable. Please try again shortly.",
                sources=[],
                error=f"Unexpected error: {str(e)}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={
                    "provider": "Open-Meteo",
                    "location": coord.canonical_name,
                    "error_type": type(e).__name__
                }
            )

        # 5. Parse and Normalize Open-Meteo Response
        current = raw_data.get("current")
        daily = raw_data.get("daily")
        hourly = raw_data.get("hourly")

        if not current or not isinstance(current, dict):
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer=f"Live weather data for {coord.canonical_name} is currently unavailable due to unexpected provider response format.",
                sources=[],
                error="Malformed Open-Meteo response: missing 'current' block.",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={"provider": "Open-Meteo", "location": coord.canonical_name}
            )

        current_temp = current.get("temperature_2m")
        current_humidity = current.get("relative_humidity_2m")
        current_precip = current.get("precipitation", 0.0)
        current_rain = current.get("rain", 0.0)
        current_wind = current.get("wind_speed_10m")
        current_wind_dir = current.get("wind_direction_10m")
        weather_code = current.get("weather_code")
        condition_desc = get_weather_condition_description(weather_code)
        forecast_timestamp = current.get("time", "")

        # Daily details (today = index 0)
        max_temp_today = None
        min_temp_today = None
        precip_sum_today = None
        precip_prob_today = None
        sunrise_today = None
        sunset_today = None

        if daily and isinstance(daily, dict):
            max_temps = daily.get("temperature_2m_max") or []
            min_temps = daily.get("temperature_2m_min") or []
            precip_sums = daily.get("precipitation_sum") or []
            precip_probs = daily.get("precipitation_probability_max") or []
            sunrises = daily.get("sunrise") or []
            sunsets = daily.get("sunset") or []

            if max_temps:
                max_temp_today = max_temps[0]
            if min_temps:
                min_temp_today = min_temps[0]
            if precip_sums:
                precip_sum_today = precip_sums[0]
            if precip_probs:
                precip_prob_today = precip_probs[0]
            if sunrises:
                sunrise_today = sunrises[0]
            if sunsets:
                sunset_today = sunsets[0]

        # 6. Pilgrimage Advice Generation
        advice_items = self._generate_pilgrimage_advice(
            current_temp=current_temp,
            condition_code=weather_code,
            precip_sum_today=precip_sum_today,
            precip_prob_today=precip_prob_today,
            max_temp_today=max_temp_today,
            min_temp_today=min_temp_today
        )

        # 7. Check for temporal sub-intent (e.g. tomorrow, weekend, next N days)
        q_lower = query.lower()
        temporal_focus = "current"
        tomorrow_summary = ""
        weekend_summary = ""
        next_days_summary = ""

        # Check for next N days request (e.g., "next 3 days", "next 2 days", "3 days forecast")
        n_days_match = re.search(r"\b(?:next|for|upcoming)\s+(\d+)\s+days?\b", q_lower)
        if not n_days_match:
            n_days_match = re.search(r"\b(\d+)\s*[- ]*days?\s+(?:forecast|weather)\b", q_lower)

        if n_days_match and daily and isinstance(daily, dict):
            req_days = int(n_days_match.group(1))
            total_avail = len(daily.get("time", []))
            days_to_show = min(req_days, total_avail)

            if days_to_show > 0:
                temporal_focus = f"next_{req_days}_days"
                day_lines = [f"\n### 📅 Forecast for the Next {req_days} Day(s)"]
                for idx in range(days_to_show):
                    d_time = daily["time"][idx] if idx < len(daily.get("time", [])) else f"Day {idx+1}"
                    w_code = daily["weather_code"][idx] if idx < len(daily.get("weather_code", [])) else None
                    w_desc = get_weather_condition_description(w_code)
                    w_max = daily["temperature_2m_max"][idx] if idx < len(daily.get("temperature_2m_max", [])) else "N/A"
                    w_min = daily["temperature_2m_min"][idx] if idx < len(daily.get("temperature_2m_min", [])) else "N/A"
                    w_rain = daily["precipitation_sum"][idx] if idx < len(daily.get("precipitation_sum", [])) else 0.0
                    w_prob = daily["precipitation_probability_max"][idx] if idx < len(daily.get("precipitation_probability_max", [])) else 0

                    try:
                        dt_obj = datetime.datetime.fromisoformat(d_time)
                        date_label = dt_obj.strftime('%a, %b %d')
                    except Exception:
                        date_label = d_time

                    day_label = f"**Day {idx+1} ({date_label})**" if idx > 0 else f"**Today ({date_label})**"
                    day_lines.append(f"- {day_label}: {w_desc} | {w_min}°C – {w_max}°C | Rain: {w_rain} mm ({w_prob}% prob)")

                if req_days > total_avail:
                    day_lines.append(f"*(Note: Open-Meteo provides up to {total_avail} days of verified forecasts; days beyond day {total_avail} are currently unavailable).*")

                next_days_summary = "\n".join(day_lines) + "\n"

        elif "tomorrow" in q_lower and daily and len(daily.get("time", [])) > 1:
            temporal_focus = "tomorrow"
            d_time = daily["time"][1]
            d_code = daily["weather_code"][1] if len(daily.get("weather_code", [])) > 1 else None
            d_desc = get_weather_condition_description(d_code)
            d_max = daily["temperature_2m_max"][1] if len(daily.get("temperature_2m_max", [])) > 1 else "N/A"
            d_min = daily["temperature_2m_min"][1] if len(daily.get("temperature_2m_min", [])) > 1 else "N/A"
            d_rain = daily["precipitation_sum"][1] if len(daily.get("precipitation_sum", [])) > 1 else 0.0
            d_prob = daily["precipitation_probability_max"][1] if len(daily.get("precipitation_probability_max", [])) > 1 else 0
            tomorrow_summary = (
                f"\n### 📅 Tomorrow's Forecast ({d_time})\n"
                f"- **Condition**: {d_desc}\n"
                f"- **Temperature Range**: {d_min}°C to {d_max}°C\n"
                f"- **Precipitation**: {d_rain} mm (Probability: {d_prob}%)\n"
            )

        elif "weekend" in q_lower and daily and len(daily.get("time", [])) >= 3:
            temporal_focus = "weekend"
            weekend_lines = ["\n### 🗓️ Upcoming Weekend Forecast"]
            for idx, d_time in enumerate(daily.get("time", [])):
                try:
                    dt = datetime.datetime.fromisoformat(d_time)
                    if dt.weekday() in [5, 6]:  # Saturday, Sunday
                        w_code = daily["weather_code"][idx] if idx < len(daily.get("weather_code", [])) else None
                        w_desc = get_weather_condition_description(w_code)
                        w_max = daily["temperature_2m_max"][idx] if idx < len(daily.get("temperature_2m_max", [])) else "N/A"
                        w_min = daily["temperature_2m_min"][idx] if idx < len(daily.get("temperature_2m_min", [])) else "N/A"
                        w_rain = daily["precipitation_sum"][idx] if idx < len(daily.get("precipitation_sum", [])) else 0.0
                        weekend_lines.append(f"- **{dt.strftime('%A, %b %d')}**: {w_desc}, {w_min}°C - {w_max}°C, Rain: {w_rain} mm")
                except Exception:
                    pass
            if len(weekend_lines) > 1:
                weekend_summary = "\n".join(weekend_lines) + "\n"

        # 8. Synthesize Answer
        targeted_answers = self._build_targeted_answers(
            query=query,
            coord=coord,
            current_temp=current_temp,
            condition_desc=condition_desc,
            weather_code=weather_code,
            current_precip=current_precip,
            current_rain=current_rain,
            current_wind=current_wind,
            precip_sum_today=precip_sum_today,
            precip_prob_today=precip_prob_today,
            max_temp_today=max_temp_today
        )

        lines = [
            f"## 🌤️ Weather Report for {coord.canonical_name}",
            "",
        ]
        if targeted_answers:
            lines.extend([
                "### ❓ Direct Answer to Your Query",
                *targeted_answers,
                ""
            ])

        lines.extend([
            "### 🌡️ Current Weather Observations",
            f"- **Observation Timestamp**: {retrieved_at_iso} (Asia/Kolkata)",
            f"- **Condition**: {condition_desc}",
            f"- **Current Temperature**: {current_temp}°C",
            f"- **Relative Humidity**: {current_humidity}%",
            f"- **Precipitation**: {current_precip} mm (Rain: {current_rain} mm)",
            f"- **Wind Speed**: {current_wind} km/h" + (f" ({current_wind_dir}°)" if current_wind_dir is not None else ""),
        ])

        if max_temp_today is not None and min_temp_today is not None:
            lines.append(f"- **Today's Range**: {min_temp_today}°C to {max_temp_today}°C")
        if precip_prob_today is not None:
            lines.append(f"- **Precipitation Probability**: {precip_prob_today}% (Expected sum: {precip_sum_today} mm)")
        if sunrise_today and sunset_today:
            lines.append(f"- **Sunrise**: {sunrise_today} | **Sunset**: {sunset_today}")

        if next_days_summary:
            lines.append(next_days_summary)
        elif tomorrow_summary:
            lines.append(tomorrow_summary)
        elif weekend_summary:
            lines.append(weekend_summary)

        lines.extend([
            "",
            "### 💡 Practical Pilgrimage Advice",
            *[f"- {adv}" for adv in advice_items],
            "",
            f"*{OPEN_METEO_ATTRIBUTION}*"
        ])

        answer_text = "\n".join(lines)

        # 9. Structure Response Metadata
        metadata = {
            "provider": "Open-Meteo",
            "location": coord.canonical_name,
            "display_name": coord.display_name or coord.canonical_name,
            "latitude": coord.latitude,
            "longitude": coord.longitude,
            "timezone": "Asia/Kolkata",
            "forecast_timestamp": forecast_timestamp,
            "retrieved_at": retrieved_at_iso,
            "units": {
                "temperature": "celsius",
                "precipitation": "mm",
                "wind_speed": "km/h"
            },
            "attribution": OPEN_METEO_ATTRIBUTION,
            "cache_hit": is_cache_hit,
            "verification_status": coord.verification_status,
            "temporal_focus": temporal_focus,
            "current_weather": {
                "temperature": current_temp,
                "relative_humidity": current_humidity,
                "precipitation": current_precip,
                "rain": current_rain,
                "wind_speed": current_wind,
                "weather_code": weather_code,
                "condition": condition_desc
            }
        }

        # Add hourly preview if present
        if hourly and isinstance(hourly, dict):
            metadata["hourly_preview"] = {
                "temperature_first_3h": (hourly.get("temperature_2m") or [])[:3],
                "precipitation_prob_first_3h": (hourly.get("precipitation_probability") or [])[:3]
            }

        # Add daily preview if present
        if daily and isinstance(daily, dict):
            metadata["daily_preview"] = {
                "max_temperatures": (daily.get("temperature_2m_max") or [])[:3],
                "min_temperatures": (daily.get("temperature_2m_min") or [])[:3],
                "precipitation_sums": (daily.get("precipitation_sum") or [])[:3]
            }

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer=answer_text,
            sources=[OPEN_METEO_ATTRIBUTION],
            latency_ms=latency_ms,
            metadata=metadata
        )
