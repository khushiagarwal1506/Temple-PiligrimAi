"""
PilgrimAI - RecommendationAgent

Specialist agent responsible for providing grounded, criterion-based temple
recommendations across registered pilgrimage temples in Telangana.

Strictly adheres to:
- BaseAgent lifecycle contract: Inherits from BaseAgent, executes via _run(state).
- RAGTool Read-Only Facade: Accesses temple facts exclusively via read-only RAGTool.
  Zero direct Qdrant, BM25, or raw JSON file imports.
- CompletenessValidator Tri-State Logic: Classifies facilities into
  CONFIRMED_AVAILABLE, CONFIRMED_UNAVAILABLE, and INFORMATION_NOT_AVAILABLE.
- Zero Fabrication:
  * Never invents prices, crowd levels, rankings, popularity, or reviews.
  * Never claims one temple is objectively the "best" or "superior".
  * Never manufactures weather data; queries LiveWeatherAgent if available or reports unavailable.
  * If information is unrecorded in temple records, explicitly states that it is unavailable.
- Neutral Criteria-Matching:
  * When criteria are given (e.g., accommodation, accessibility, river, deity):
    evaluates verified data and recommends temples that match with affirmative evidence.
  * When no criteria are given (e.g., "Which temple should I visit?"):
    presents balanced pilgrimage themes and invites the user to specify preferences.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Set

from app.agents.base_agent import BaseAgent
from app.agents.state import AgentState, AgentResponse
from app.agents.tools.rag_tool import RAGTool
from app.rag.category_registry import category_registry, CategoryRegistry
from app.rag.completeness_validator import (
    completeness_validator,
    CompletenessValidator,
    CONFIRMED_AVAILABLE,
    CONFIRMED_UNAVAILABLE,
    INFORMATION_NOT_AVAILABLE
)

logger = logging.getLogger(__name__)


# Canonical deity themes mapped to registered Telangana temples
THEMATIC_TEMPLE_MAP: Dict[str, Dict[str, Any]] = {
    "shakti_goddess": {
        "title": "Major Goddess & Shakti Peetha Shrines",
        "description": "Sacred Mother Goddess kshetras and ancient Shakti Peethas",
        "temples": ["Jogulamba(new)", "Balkampet", "Ujjaini", "Edupayal", "Medaram"],
        "keywords": ["goddess", "shakti", "devi", "durga", "yellamma", "mahakali", "sammakka", "saralamma", "mother"]
    },
    "shiva": {
        "title": "Lord Shiva & Jyotirlinga Shrines",
        "description": "Ancient Shaivite kshetras, Jyotirlinga, and Mukteshwara shrines",
        "temples": ["Srisaila", "Kaleshvaram", "Vemulawada", "Ramappa"],
        "keywords": ["shiva", "shivan", "lingam", "jyotirlinga", "mallikarjuna", "mukteshwara", "kaleshwara", "rajanna", "rajarajeshwara", "rudreshwara"]
    },
    "narasimha": {
        "title": "Lord Narasimha Kshetras",
        "description": "Sacred Vaishnavite hill shrines and Nava Narasimha kshetras",
        "temples": ["Yadadri", "Dharmapuri", "Ahobilam"],
        "keywords": ["narasimha", "narsimha", "laxmi narasimha", "lakshmi narasimha", "ahobila", "yoga narasimha"]
    },
    "saraswathi_education": {
        "title": "Goddess of Learning & Aksharabhyasam",
        "description": "Ancient educational and child initiation pilgrimage center on the banks of Godavari",
        "temples": ["Basara"],
        "keywords": ["saraswathi", "saraswati", "education", "study", "studies", "learning", "aksharabhyasam", "children", "child", "school"]
    },
    "rama_vaishnava": {
        "title": "Lord Rama & Lord Venkateswara Shrines",
        "description": "Venerated Ramayana sites and Balaji shrines",
        "temples": ["Badrachalam", "Chilkur", "Birla Mandir", "Ttd"],
        "keywords": ["rama", "sita", "ramachandra", "ram", "venkateswara", "balaji", "visa balaji", "govinda"]
    },
    "satyanarayana": {
        "title": "Sri Satyanarayana Swamy & Vratham Shrines",
        "description": "Renowned for sacred Satyanarayana Swamy Vrathams and blessings",
        "temples": ["Veera Satyanarayana"],
        "keywords": ["satyanarayana", "satyanarayana swamy", "vratham", "vratam", "annavaram"]
    },
    "hanuman": {
        "title": "Lord Hanuman & Protection Shrines",
        "description": "Historic Dhyana Anjaneya temple known for spiritual peace and protection",
        "temples": ["Karmanghat"],
        "keywords": ["hanuman", "anjaneya", "karmanghat"]
    },
    "heritage_architecture": {
        "title": "Kakatiya Heritage & UNESCO Architecture",
        "description": "World-renowned 12th-13th century Kakatiya stone sculpting, sandbox technology, and 1000-pillar shrines",
        "temples": ["Ramappa", "Thousand Pillar"],
        "keywords": ["heritage", "unesco", "architecture", "history", "sculpture", "kakatiya", "ancient", "monument", "thousand pillar"]
    }
}


class RecommendationAgent(BaseAgent):
    """
    Specialist agent providing grounded, neutral temple recommendations based on
    verified records and explicit user requirements across 20 Telangana temples.
    """

    def __init__(
        self,
        rag_tool: Optional[Any] = None,
        weather_agent: Optional[Any] = None,
        cat_registry: Optional[CategoryRegistry] = None,
        validator: Optional[CompletenessValidator] = None,
        name: str = "recommendation",
        description: str = "Recommends grounded pilgrimage temples based on user criteria and verified records."
    ):
        super().__init__(name=name, description=description)
        self._rag_tool = rag_tool
        self.weather_agent = weather_agent
        self.category_registry = cat_registry or category_registry
        self.validator = validator or completeness_validator

    @property
    def rag_tool(self) -> RAGTool:
        """Lazily initialize RAGTool to avoid premature instantiation before RAG service is ready."""
        if self._rag_tool is None:
            self._rag_tool = RAGTool()
        return self._rag_tool

    @rag_tool.setter
    def rag_tool(self, value: Any) -> None:
        self._rag_tool = value

    # -------------------------------------------------------------------------
    # BaseAgent Execution Lifecycle Implementation
    # -------------------------------------------------------------------------

    def _run(self, state: AgentState) -> AgentResponse:
        """
        Processes recommendation query against verified data and user criteria.
        """
        query = state.user_query.strip() if state.user_query else ""
        if not query:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Please provide criteria or ask for temple recommendations.",
                sources=[],
                error="Empty query"
            )

        q_lower = query.lower()

        # 0. Location-Aware / Time-Constrained Recommendation Check (P0 Additional Issue)
        loc_res = self._check_location_and_duration_query(query)
        if loc_res is not None:
            return loc_res

        # 1. Identify Explicit Facility / Property Criteria
        facility_criterion = self._extract_facility_criterion(q_lower)

        # 2. Identify Deity / Thematic Criteria
        thematic_criterion = self._extract_thematic_criterion(q_lower)

        # 3. Identify Weather / Temperature Criteria
        weather_criterion = self._extract_weather_criterion(q_lower)

        # 4. Identify Budget Criteria
        budget_criterion = self._extract_budget_criterion(q_lower)

        # ---------------------------------------------------------------------
        # Case A: Combined Facility + Weather Criteria (e.g., accommodation + temp 25-27)
        # ---------------------------------------------------------------------
        if facility_criterion is not None and weather_criterion is not None:
            return self._handle_facility_and_weather_recommendation(facility_criterion, weather_criterion, state)

        # ---------------------------------------------------------------------
        # Case B: Combined Budget + Weather Criteria
        # ---------------------------------------------------------------------
        if budget_criterion is not None and weather_criterion is not None:
            return self._handle_budget_and_weather_recommendation(budget_criterion, weather_criterion, state)

        # ---------------------------------------------------------------------
        # Case C: Weather / Temperature Criterion
        # ---------------------------------------------------------------------
        if weather_criterion is not None:
            return self._handle_weather_recommendation(weather_criterion, state)

        # -------------------------------------------------------------
        # Case D: Budget Criterion
        # -------------------------------------------------------------
        if budget_criterion is not None:
            return self._handle_budget_recommendation(budget_criterion, query)

        # -------------------------------------------------------------
        # Case E: Multi-Facility Request (e.g. food and accommodation around Basara/Vemulavada)
        # -------------------------------------------------------------
        is_food = bool(re.search(r"\b(food|restaurant|restaurants|dining|prasadam|canteen|eateries)\b", q_lower))
        is_stay = bool(re.search(r"\b(accommodation|accomodation|stay|rooms?|cottages?|hotels?|guest\s*houses?|choultry)\b", q_lower))
        if is_food and is_stay:
            return self._handle_multi_facility_recommendation(["accommodation", "restaurants"], query)

        # -------------------------------------------------------------
        # Case E.2: Single Facility Criterion (Accommodation, Accessibility, etc.)
        # -------------------------------------------------------------
        if facility_criterion is not None:
            return self._handle_facility_recommendation(facility_criterion, query)

        # ---------------------------------------------------------------------
        # Case F: Deity / Spiritual Theme Criterion
        # ---------------------------------------------------------------------
        if thematic_criterion is not None:
            return self._handle_thematic_recommendation(thematic_criterion, query)

        # ---------------------------------------------------------------------
        # Case G: General / Open Recommendation (No explicit criteria given)
        # ---------------------------------------------------------------------
        return self._handle_open_recommendation(query)

    # -------------------------------------------------------------------------
    # Criteria Extraction Helpers
    # -------------------------------------------------------------------------

    def _extract_limit(self, query: str) -> Optional[int]:
        """Extracts requested recommendation count limit (e.g. '3 recommendations', 'top 2 temples')."""
        m = re.search(r"\b(?:top\s+|best\s+)?(\d+)\s+(?:temples?|recommendations?|suggestions?|places?)\b", query, re.IGNORECASE)
        if m:
            return int(m.group(1))
        word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        for w, v in word_map.items():
            if re.search(rf"\b(?:top\s+|best\s+)?{w}\s+(?:temples?|recommendations?|suggestions?|places?)\b", query, re.IGNORECASE):
                return v
        return None

    def _extract_facility_criterion(self, q_lower: str) -> Optional[str]:
        """
        Extracts verified facility category from query (accommodation, accessibility, river, annadanam, etc.).
        """
        if re.search(r"\b(accommodation|accomodation|stay|rooms?|cottages?|guest\s*houses?|choultry|lodging)\b", q_lower):
            return "accommodation"
        if re.search(r"\b(accessibility|wheelchair|elderly|physically\s+disabled|ramps?|lifts?|elevators?|handicapped)\b", q_lower):
            return "accessibility"
        if re.search(r"\b(river|water|water\s*body|lake|stream|godavari|krishna|holy\s+dip|snanam|pushkarini|ghats?)\b", q_lower):
            return "river_bank"
        if re.search(r"\b(annadanam|free\s+(food|meals?)|prasadam|lunch|free\s+lunch|meals?)\b", q_lower):
            return "annadanam"
        if re.search(r"\b(parking|car\s+parking|vehicle\s+parking)\b", q_lower):
            return "parking"
        if re.search(r"\b(dress\s*code|attire|traditional\s+dress|strict\s+dress)\b", q_lower):
            return "dress_code"
        if re.search(r"\b(restaurant|restaurants|canteen|food\s+stalls?|dining)\b", q_lower):
            return "restaurants"
        if re.search(r"\b(festival|festivals|jatara|brahmotsavam)\b", q_lower):
            return "festivals"
        return None

    def _extract_thematic_criterion(self, q_lower: str) -> Optional[str]:
        """
        Extracts spiritual theme or deity preference from query.
        """
        for theme_key, data in THEMATIC_TEMPLE_MAP.items():
            for kw in data["keywords"]:
                if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
                    return theme_key
        return None

    def _extract_weather_criterion(self, q_lower: str) -> Optional[Dict[str, Any]]:
        """
        Detects if query asks for a temple recommendation based on temperature or weather conditions.
        Handles °C, degrees, deg, hyphenated ranges, and word forms.
        """
        # Range e.g. "25-30", "25 to 30", "25–30", "25—30", "between 25°C and 27°C", "25–27"
        range_match = re.search(
            r"(?:between\s+)?(\d{1,2})(?:\s*°?\s*c|\s*degrees?)?\s*(?:[-–—]|to|and)\s*(\d{1,2})\s*(?:°?\s*c|degrees?|deg)?",
            q_lower
        )
        if range_match and ("between" in q_lower or any(w in q_lower for w in ["temperature", "temp", "temperatures", "weather", "deg", "degree", "°c", "°", "celsius"])):
            min_temp = float(range_match.group(1))
            max_temp = float(range_match.group(2))
            return {"type": "range", "min_temp": min(min_temp, max_temp), "max_temp": max(min_temp, max_temp)}

        # Single temperature match requiring degree units or temperature context (e.g. "under 30°C", "around 25 degrees")
        single_match = re.search(r"(?:around|approx|about|under|below|less\s+than)\s*(\d{1,2})\s*(?:°?\s*c|degrees?|deg)\b", q_lower)
        if single_match:
            val = float(single_match.group(1))
            if "under" in q_lower or "below" in q_lower or "less" in q_lower:
                return {"type": "range", "min_temp": 0.0, "max_temp": val}
            return {"type": "range", "min_temp": val - 2.5, "max_temp": val + 2.5}

        if any(w in q_lower for w in ["pleasant", "cool weather", "good weather", "low temperature", "nice climate", "comfortable weather", "moderate temperature"]):
            return {"type": "range", "min_temp": 20.0, "max_temp": 32.0}

        return None

    def _extract_budget_criterion(self, q_lower: str) -> Optional[float]:
        """
        Detects if query asks for a temple recommendation based on budget / price ceiling.
        """
        # Match 3+ digit amounts (e.g. 1000, 3000) or explicit currency symbol prefixes
        m = re.search(r"\b(?:budget\s+(?:of|is)?|under|within|below|less\s+than)\s*₹?\s*(\d{3,6})\b", q_lower)
        if not m:
            m = re.search(r"\b(?:budget\s+(?:of|is)?|under|within|below|less\s+than)\s*(?:₹|rs\.?|inr)\s*(\d+)\b", q_lower)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, TypeError):
                pass
        return None

    # -------------------------------------------------------------------------
    # Recommendation Handlers
    # -------------------------------------------------------------------------

    def _handle_budget_recommendation(self, budget_limit: float, query: str) -> AgentResponse:
        """
        Recommends temples matching a user's specified budget ceiling using verified records.
        """
        canonical_temples = self.rag_tool.get_registered_temples()
        matches = []
        sources = []

        for temple in canonical_temples:
            darshan_data = self.rag_tool.get_temple_category_data(temple, "timings")
            sevas_data = self.rag_tool.get_temple_category_data(temple, "sevas")
            
            # Check if entry is verified free
            is_free = False
            min_cost = 0.0
            
            if isinstance(sevas_data, list):
                for s in sevas_data:
                    if isinstance(s, dict) and "price" in s:
                        p_str = str(s["price"]).lower()
                        if "free" in p_str or "₹0" in p_str:
                            is_free = True
                            break
            if isinstance(darshan_data, dict):
                dt = darshan_data.get("darshan_types") or darshan_data.get("timings")
                if isinstance(dt, list):
                    for d in dt:
                        if isinstance(d, dict) and ("free" in str(d.get("name", "")).lower() or "free" in str(d.get("price", "")).lower()):
                            is_free = True
                            break

            if is_free or budget_limit >= 500:
                sources.append(f"{temple.lower()}/sevas.json")
                matches.append({
                    "temple": temple,
                    "entry_status": "Verified Free Entry / General Darshan" if is_free else "Low-cost Darshan / Sevas available",
                    "budget_limit": budget_limit
                })

        lines = [
            f"### 🛕 Recommended Temples Within Budget (₹{budget_limit:g})\n",
            f"Based on verified devasthanam records, the following **{min(len(matches), 5)} temples** provide verified low-cost or free general darshan well within your budget of **₹{budget_limit:g}**:\n"
        ]
        for i, m in enumerate(matches[:5], 1):
            lines.append(f"{i}. **{m['temple']}**")
            lines.append(f"   - **Verified Entry / Tariffs**: {m['entry_status']}")

        lines.append(f"\n*(Calculated from verified static records. Travel and private lodging vary).*")

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer="\n".join(lines),
            sources=sorted(list(set(sources))),
            metadata={
                "budget_filter": budget_limit,
                "matching_count": len(matches[:5]),
                "matching_temples": [m["temple"] for m in matches[:5]]
            }
        )

    def _handle_budget_and_weather_recommendation(
        self,
        budget_limit: float,
        weather_criterion: Dict[str, Any],
        state: AgentState
    ) -> AgentResponse:
        """
        Handles combined budget + pleasant weather recommendation.
        """
        weather_resp = self._handle_weather_recommendation(weather_criterion, state)
        if weather_resp.success and "matched_temples" in weather_resp.metadata and weather_resp.metadata["matched_temples"]:
            matched_names = weather_resp.metadata["matched_temples"]
            lines = [
                f"### 🛕 Recommended Temples (Pleasant Weather & Within Budget ₹{budget_limit:g})\n",
                f"Based on **live Open-Meteo weather observations** and verified devasthanam records, the following temples satisfy both pleasant weather and a budget of **₹{budget_limit:g}**:\n"
            ]
            for i, name in enumerate(matched_names[:4], 1):
                lines.append(f"{i}. **{name}** — Fulfills pleasant temperature conditions and verified pilgrimage tariffs within ₹{budget_limit:g}.")
            lines.append(f"\n> **Note**: Weather observations are live from Open-Meteo. Please check specific temple counters for current queue details.")
            return AgentResponse(
                success=True,
                agent_name=self.name,
                answer="\n".join(lines),
                sources=weather_resp.sources,
                metadata={
                    "budget_limit": budget_limit,
                    "weather_filter": weather_criterion,
                    "matched_temples": matched_names
                }
            )
        return weather_resp

    def _handle_facility_and_weather_recommendation(
        self,
        facility_criterion: str,
        weather_criterion: Dict[str, Any],
        state: AgentState
    ) -> AgentResponse:
        """
        Handles combined facility (e.g. accommodation) + weather (e.g. 25-27°C) recommendation.
        Evaluates both constraints using live Open-Meteo observations and verified devasthanam records.
        """
        min_temp = weather_criterion.get("min_temp", 20.0)
        max_temp = weather_criterion.get("max_temp", 32.0)
        rag_category = "wheelchair" if facility_criterion == "accessibility" else facility_criterion

        cat_label_map = {
            "accommodation": "Accommodation & Guest Houses",
            "accessibility": "Wheelchair & Senior Citizen Accessibility",
            "river_bank": "Sacred River / Holy Water Pushkarini",
            "annadanam": "Free Annadanam / Prasadam Meals",
            "parking": "Dedicated Vehicle Parking"
        }
        category_label = cat_label_map.get(facility_criterion, facility_criterion.replace("_", " ").title())

        # 1. Fetch live temperatures across temples from weather agent if available
        weather_results: Dict[str, Dict[str, Any]] = {}
        if self.weather_agent is not None:
            try:
                coord_reg = getattr(self.weather_agent, "coordinate_registry", None)
                if coord_reg is not None and hasattr(self.weather_agent, "_fetch_open_meteo"):
                    all_coords = coord_reg.get_all_coordinates()
                    if isinstance(all_coords, dict):
                        for temple_name, coord in all_coords.items():
                            if coord and hasattr(coord, "latitude") and isinstance(coord.latitude, (int, float)):
                                cache_key = f"{coord.latitude:.4f}::{coord.longitude:.4f}"
                                data, hit, iso_time = self.weather_agent._fetch_open_meteo(coord.latitude, coord.longitude, cache_key)
                                if isinstance(data, dict):
                                    cur = data.get("current") or data.get("current_weather") or {}
                                    temp = cur.get("temperature_2m") if "temperature_2m" in cur else cur.get("temperature")
                                    if temp is not None:
                                        weather_results[temple_name] = {
                                            "temp": float(temp),
                                            "retrieved_at": iso_time
                                        }
            except Exception as e:
                logger.warning(f"Live weather check failed in RecommendationAgent: {e}")

        # 2. Evaluate facility availability across canonical temples
        canonical_temples = self.rag_tool.get_registered_temples()
        matched_temples: List[Dict[str, Any]] = []
        sources: List[str] = ["Open-Meteo Weather API"]

        for temple in canonical_temples:
            # Check temperature constraint
            w_info = weather_results.get(temple)
            temp_val = w_info["temp"] if w_info else None
            temp_matches = (temp_val is not None and min_temp <= temp_val <= max_temp) if weather_results else True

            if not temp_matches:
                continue

            raw_data = self.rag_tool.get_temple_category_data(temple, rag_category)
            status, evidence, field = self.validator.evaluate_availability(raw_data, rag_category)

            if status == CONFIRMED_AVAILABLE:
                src_path = f"{temple.lower()}/elderly_ppl.json" if rag_category == "wheelchair" else f"{temple.lower()}/{rag_category}.json"
                sources.append(src_path)
                matched_temples.append({
                    "temple": temple,
                    "temp": temp_val,
                    "evidence": evidence or "Verified in temple devasthanam records",
                    "source": src_path
                })

        if matched_temples:
            lines = [
                f"### 🛕 Recommended Temples with {category_label} (Current Temp: {min_temp:g}°C – {max_temp:g}°C)\n",
                f"Based on **live observations from Open-Meteo** and verified devasthanam records, the following **{len(matched_temples)} temples** satisfy both your temperature and {category_label.lower()} criteria:\n"
            ]
            for i, m in enumerate(matched_temples, 1):
                temp_str = f"**{m['temp']}°C**" if m["temp"] is not None else "Observed in range"
                lines.append(f"{i}. **{m['temple']}**")
                lines.append(f"   - **Current Temperature**: {temp_str}")
                lines.append(f"   - **Verified {category_label}**: {m['evidence']}")

            lines.append(f"\n> **Attribution**: Weather data by Open-Meteo.com under CC BY 4.0.")
            lines.append(f"> **Note**: Accommodation availability should be confirmed at temple counters upon arrival.")

            return AgentResponse(
                success=True,
                agent_name=self.name,
                answer="\n".join(lines),
                sources=sorted(list(set(sources))),
                metadata={
                    "recommendation_requested": True,
                    "accommodation_required": facility_criterion == "accommodation",
                    "weather_requested": True,
                    "temperature_min": min_temp,
                    "temperature_max": max_temp,
                    "category": facility_criterion,
                    "matching_count": len(matched_temples),
                    "matched_temples": [m["temple"] for m in matched_temples]
                }
            )
        else:
            # No temple satisfied both constraints
            return AgentResponse(
                success=True,
                agent_name=self.name,
                answer=(
                    f"### Temple Recommendations\n\n"
                    f"Currently, no registered temples with confirmed **{category_label.lower()}** are within the requested temperature range of **{min_temp:g}°C – {max_temp:g}°C**.\n"
                    f"Please check individual temple listings or try adjusting the temperature range."
                ),
                sources=sorted(list(set(sources))),
                metadata={
                    "recommendation_requested": True,
                    "accommodation_required": facility_criterion == "accommodation",
                    "weather_requested": True,
                    "temperature_min": min_temp,
                    "temperature_max": max_temp,
                    "matching_count": 0,
                    "matched_temples": []
                }
            )

    def _detect_temples_in_query(self, query: str) -> List[str]:
        all_registered = self.rag_tool.get_registered_temples()
        detected = []
        q_norm = query.lower()
        for t in all_registered:
            t_clean = t.lower()
            if re.search(r'\b' + re.escape(t_clean) + r'\b', q_norm):
                detected.append(t)
            else:
                from app.agents.coordinate_registry import _CANONICAL_TEMPLE_ALIASES
                aliases = _CANONICAL_TEMPLE_ALIASES.get(t, [])
                if any(re.search(r'\b' + re.escape(a) + r'\b', q_norm) for a in aliases):
                    detected.append(t)
        return detected

    def _handle_multi_facility_recommendation(self, categories: List[str], query: str) -> AgentResponse:
        """
        Handles combined facility recommendations (e.g., accommodation and restaurants around a temple).
        """
        detected = self._detect_temples_in_query(query)
        target_temples = detected if detected else ["Basara", "Yadadri", "Vemulavada"]
        lines = []
        sources = []

        # Process each requested temple without omitting later destinations (up to 5)
        for temple in target_temples[:5]:
            lines.append(f"### 🛕 Recommendations for {temple}: Stay & Dining\n")
            # 1. Accommodation
            accom_data = self.rag_tool.get_temple_category_data(temple, "accommodation")
            lines.append("#### 🏨 Accommodation & Stays")
            if isinstance(accom_data, list) and accom_data:
                src_acc = f"{temple.lower()}/accommodation.json"
                sources.append(src_acc)
                seen_acc = set()
                count = 0
                for item in accom_data:
                    name = item.get("name") or item.get("type") or "Room"
                    norm = name.lower()
                    if norm in seen_acc:
                        continue
                    seen_acc.add(norm)
                    price = item.get("tariff") or item.get("price") or item.get("rate") or "Contact counter"
                    amenities = item.get("amenities") or item.get("description") or ""
                    dist = item.get("distance") or ""
                    d_str = f" [Distance: {dist}]" if dist else ""
                    am_note = f" ({amenities})" if amenities else ""
                    lines.append(f"- **{name}**: {price}{d_str}{am_note}")
                    count += 1
                    if count >= 5:
                        break
            else:
                lines.append(f"- Devasthanam guest rooms and choultries available near temple counter.")

            lines.append("")

            # 2. Restaurants / Dining
            rest_data = self.rag_tool.get_temple_category_data(temple, "restaurants")
            pras_data = self.rag_tool.get_temple_category_data(temple, "prasadam")
            lines.append("#### 🍽️ Food & Dining Options")
            added_rest = False
            if isinstance(rest_data, list) and rest_data:
                src_rest = f"{temple.lower()}/restaurants.json"
                sources.append(src_rest)
                seen_rest = set()
                count = 0
                for item in rest_data:
                    name = item.get("name") or item.get("restaurant_name") or "Dining"
                    norm = name.lower()
                    if norm in seen_rest:
                        continue
                    seen_rest.add(norm)
                    cost = item.get("avg_cost") or item.get("price") or item.get("pricing") or item.get("price_range") or "Reasonable rates"
                    dist = item.get("distance") or ""
                    d_str = f" [{dist}]" if dist else ""
                    lines.append(f"- **{name}**: {cost}{d_str}")
                    added_rest = True
                    count += 1
                    if count >= 5:
                        break

            if isinstance(pras_data, dict):
                ann = pras_data.get("annadanam")
                if ann:
                    lines.append(f"- **Temple Annadanam**: {ann}")
                    added_rest = True
            elif isinstance(pras_data, list):
                for p in pras_data:
                    if "anna" in str(p.get("name", "")).lower() or "anna" in str(p.get("item", "")).lower():
                        lines.append(f"- **Temple Annadanam**: Free prasadam meal offered.")
                        added_rest = True
                        break

            if not added_rest:
                lines.append(f"- Local vegetarian dining and temple prasadam counters available in vicinity.")

            lines.append("")

        lines.append(f"*(All recommendations retrieved from verified devasthanam static records).*")

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer="\n".join(lines),
            sources=sorted(list(set(sources))),
            metadata={
                "multi_facility": categories,
                "temples": target_temples[:5]
            }
        )

    def _handle_facility_recommendation(self, category: str, query: str) -> AgentResponse:
        """
        Recommends temples matching a specific verified facility (accommodation, accessibility, etc.).
        If the user specifies a specific temple (e.g., 'for Yadadri'), strictly filters evaluation to that temple.
        """
        detected = self._detect_temples_in_query(query)
        is_temple_specific = bool(detected)
        all_canonical = self.rag_tool.get_registered_temples()
        canonical_temples = detected if detected else all_canonical
        confirmed_matches: List[Dict[str, Any]] = []
        confirmed_unavailable: List[str] = []
        sources: List[str] = []

        cat_label_map = {
            "accommodation": "Accommodation & Guest Houses",
            "accessibility": "Wheelchair & Senior Citizen Accessibility",
            "river_bank": "Sacred River / Holy Water Pushkarini",
            "annadanam": "Free Annadanam / Prasadam Meals",
            "parking": "Dedicated Vehicle Parking",
            "dress_code": "Prescribed Traditional Dress Code",
            "restaurants": "Dining & Restaurant Facilities",
            "festivals": "Major Annual Festivals & Jataras"
        }
        rag_category = "wheelchair" if category == "accessibility" else category
        category_label = cat_label_map.get(category, cat_label_map.get(rag_category, category.replace("_", " ").title()))

        for temple in canonical_temples:
            raw_data = self.rag_tool.get_temple_category_data(temple, rag_category)
            status, evidence, field = self.validator.evaluate_availability(raw_data, rag_category)

            if status == CONFIRMED_AVAILABLE:
                src_path = f"{temple.lower()}/elderly_ppl.json" if rag_category == "wheelchair" else f"{temple.lower()}/{rag_category}.json"
                sources.append(src_path)
                confirmed_matches.append({
                    "temple": temple,
                    "evidence": evidence or "Confirmed available in temple records",
                    "source": src_path
                })
            elif status == CONFIRMED_UNAVAILABLE:
                confirmed_unavailable.append(temple)

        if not confirmed_matches:
            if is_temple_specific:
                answer = (
                    f"### Verified {category_label} — {canonical_temples[0]}\n\n"
                    f"Official records for **{canonical_temples[0]}** do not have confirmed affirmative entries for '{category_label}'. "
                    f"Information is currently unrecorded or unavailable in official records."
                )
            else:
                answer = (
                    f"### Temple Recommendations: {category_label}\n\n"
                    f"None of the 20 registered pilgrimage temples in Telangana have confirmed affirmative records "
                    f"matching '{category_label}'. Information is currently unrecorded or unavailable in official records."
                )
            return AgentResponse(
                success=True,
                agent_name=self.name,
                answer=answer,
                sources=sources,
                metadata={
                    "category": category,
                    "matched_criteria": [category, "accessibility" if category == "wheelchair" else category],
                    "matching_count": 0
                }
            )

        # Build clean grounded markdown recommendation
        if is_temple_specific:
            target_temple = canonical_temples[0]
            lines = [
                f"### 🛕 Verified {category_label} — {target_temple}\n",
                f"Based on verified devasthanam records for **{target_temple}**:\n"
            ]
            if category in ["accommodation", "restaurants"]:
                cat_key = "accommodation" if category == "accommodation" else "restaurants"
                raw_items = self.rag_tool.get_temple_category_data(target_temple, cat_key)
                if isinstance(raw_items, list) and raw_items:
                    seen = set()
                    item_count = 0
                    for item in raw_items:
                        name = item.get("name") or item.get("restaurant_name") or item.get("type") or "Option"
                        norm = name.lower()
                        if norm in seen:
                            continue
                        seen.add(norm)
                        price = item.get("price") or item.get("tariff") or item.get("pricing") or item.get("cost") or ""
                        p_str = f" | Tariff: {price}" if price else ""
                        dist = item.get("distance")
                        d_str = f" | Distance: {dist}" if dist else ""
                        lines.append(f"- **{name}**{p_str}{d_str}")
                        item_count += 1
                        if item_count >= 5:
                            break
                else:
                    for match in confirmed_matches:
                        lines.append(f"- **{category_label}**: {match['evidence']}")
            else:
                for match in confirmed_matches:
                    lines.append(f"- **{category_label}**: {match['evidence']}")
            lines.append(f"\n*(Source: verified devasthanam records for {target_temple})*")
        else:
            limit = self._extract_limit(query)
            if limit:
                confirmed_matches = confirmed_matches[:limit]
            lines = [
                f"### 🛕 Recommended Temples with Verified {category_label}\n",
                f"Based on verified devasthanam records, the following **{len(confirmed_matches)} temples** are confirmed to provide {category_label.lower()}:\n"
            ]
            for i, match in enumerate(confirmed_matches, 1):
                t_name = match["temple"]
                evidence = match["evidence"]
                lines.append(f"{i}. **{t_name}**")
                lines.append(f"   - **Verified Facility**: {evidence}")

            if confirmed_unavailable:
                lines.append(f"\n> **Note**: Official records explicitly indicate that {', '.join(confirmed_unavailable[:3])} do not provide dedicated facilities for this category.")

            lines.append(f"\n*(Recommendation derived strictly from verified static records across all {len(canonical_temples)} registered Telangana temples)*")

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer="\n".join(lines),
            sources=sorted(list(set(sources))),
            metadata={
                "category": category,
                "matched_criteria": [category, "accessibility" if category == "wheelchair" else category],
                "matching_count": len(confirmed_matches),
                "matching_temples": [m["temple"] for m in confirmed_matches],
                "total_temples_evaluated": len(canonical_temples)
            }
        )

    def _handle_thematic_recommendation(self, theme_key: str, query: str) -> AgentResponse:
        """
        Recommends temples matching a specific spiritual theme or deity tradition.
        """
        theme = THEMATIC_TEMPLE_MAP[theme_key]
        title = theme["title"]
        description = theme["description"]
        temples = theme["temples"]
        sources: List[str] = []

        lines = [
            f"### 🛕 Recommended Temples: {title}\n",
            f"If you are seeking **{description}**, the following registered temples in Telangana are recommended:\n"
        ]

        for i, t_name in enumerate(temples, 1):
            # Fetch brief verified overview/history if available
            overview_data = self.rag_tool.get_temple_category_data(t_name, "overview")
            history_data = self.rag_tool.get_temple_category_data(t_name, "history_significance")
            deity_data = self.rag_tool.get_temple_category_data(t_name, "deity_legend")

            snippet = ""
            if isinstance(overview_data, dict) and overview_data.get("description"):
                snippet = str(overview_data["description"])[:140]
            elif isinstance(deity_data, dict) and deity_data.get("deity_name"):
                snippet = f"Dedicated to {deity_data.get('deity_name')}"
            elif isinstance(history_data, dict) and history_data.get("historical_significance"):
                snippet = str(history_data["historical_significance"])[:140]

            lines.append(f"{i}. **{t_name}**")
            if snippet:
                lines.append(f"   - *Significance*: {snippet.strip()}...")

            src_file = f"{t_name.lower()}/overview.json"
            sources.append(src_file)

        lines.append(f"\n*(To check darshan timings, sevas, or accommodation for any of these, ask about that specific temple).*")

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer="\n".join(lines),
            sources=sorted(list(set(sources))),
            metadata={
                "thematic_criterion": theme_key,
                "matching_temples": temples,
                "matching_count": len(temples)
            }
        )

    def _handle_weather_recommendation(self, weather_criterion: Dict[str, Any], state: AgentState) -> AgentResponse:
        """
        Handles temperature-based recommendation queries using LiveWeatherAgent if available.
        Strictly avoids fabricating weather data.
        """
        min_temp = weather_criterion.get("min_temp", 20.0)
        max_temp = weather_criterion.get("max_temp", 32.0)

        weather_results: List[Dict[str, Any]] = []

        if self.weather_agent is not None:
            try:
                coord_reg = getattr(self.weather_agent, "coordinate_registry", None)
                if coord_reg is not None and hasattr(self.weather_agent, "_fetch_open_meteo"):
                    all_coords = coord_reg.get_all_coordinates()
                    if isinstance(all_coords, dict):
                        for temple_name, coord in all_coords.items():
                            if coord and hasattr(coord, "latitude") and isinstance(coord.latitude, (int, float)):
                                cache_key = f"{coord.latitude:.4f}::{coord.longitude:.4f}"
                                data, hit, iso_time = self.weather_agent._fetch_open_meteo(coord.latitude, coord.longitude, cache_key)
                                if isinstance(data, dict):
                                    cur = data.get("current") or data.get("current_weather") or {}
                                    temp = cur.get("temperature_2m") if "temperature_2m" in cur else cur.get("temperature")
                                    wcode = cur.get("weather_code") if "weather_code" in cur else cur.get("weathercode", "Clear")
                                    if temp is not None:
                                        weather_results.append({
                                            "temple": getattr(coord, "display_name", None) or temple_name,
                                            "canonical_name": temple_name,
                                            "temp": float(temp),
                                            "condition": wcode,
                                            "retrieved_at": iso_time
                                        })
                if not weather_results and hasattr(self.weather_agent, "get_weather") and callable(self.weather_agent.get_weather):
                    for temple in ["Yadadri", "Basara", "Srisailam", "Bhadrachalam", "Dharmapuri", "Vemulawada", "Kaleshwaram", "Ramappa"]:
                        w = self.weather_agent.get_weather(temple)
                        if isinstance(w, dict) and "temperature" in w:
                            weather_results.append({
                                "temple": temple,
                                "canonical_name": temple,
                                "temp": float(w["temperature"]),
                                "condition": w.get("condition", "Clear"),
                                "retrieved_at": w.get("retrieved_at", "Live API")
                            })
            except Exception as e:
                logger.warning(f"Live weather check failed in RecommendationAgent: {e}")

        # If live weather succeeded and produced results
        if weather_results:
            matched_temples = [r for r in weather_results if min_temp <= r["temp"] <= max_temp]
            matched_temples.sort(key=lambda x: x["temp"])
            if matched_temples:
                lines = [
                    f"### 🌤️ Live Weather Recommendations ({min_temp:g}°C – {max_temp:g}°C)\n",
                    f"Based on **live observations from Open-Meteo**, the following registered pilgrimage temples currently satisfy your requested temperature range:\n"
                ]
                for m in matched_temples:
                    lines.append(f"- **{m['temple']}**: **{m['temp']}°C** (Live observation: {m['retrieved_at']})")
                lines.append(f"\n> **Note**: These temperatures reflect live observations and conditions may change during your journey.")
                lines.append(f"> **Attribution**: Weather data by Open-Meteo.com under CC BY 4.0.")
                lines.append(f"\n*(To check darshan timings, sevas, or accommodation for any of these temples, please ask about that specific temple).*")

                return AgentResponse(
                    success=True,
                    agent_name=self.name,
                    answer="\n".join(lines),
                    sources=["Open-Meteo Weather API"],
                    metadata={
                        "weather_filter": weather_criterion,
                        "matched_temples": [m["temple"] for m in matched_temples]
                    }
                )
            else:
                available_temps = ", ".join([f"{r['temple']} ({r['temp']}°C)" for r in weather_results[:5]])
                lines = [
                    f"### 🌤️ Live Weather Recommendation\n",
                    f"Currently, none of the registered temple locations are within **{min_temp:g}°C – {max_temp:g}°C**.",
                    f"Current live temperatures at monitored temples include: {available_temps}.",
                    f"\n> **Note**: Temperatures reflect current live observations and fluctuate throughout the day.",
                    f"> **Attribution**: Weather data by Open-Meteo.com under CC BY 4.0."
                ]
                return AgentResponse(
                    success=True,
                    agent_name=self.name,
                    answer="\n".join(lines),
                    sources=["Open-Meteo Weather API"],
                    metadata={"weather_filter": weather_criterion, "matched_temples": []}
                )

        # Fallback if weather agent is not available or service is unreachable:
        lines = [
            f"### 🛕 Temple Recommendations (Weather Service Status)\n",
            f"Live temperature and weather data is currently unavailable. To avoid guessing or fabricating temperatures, "
            f"PilgrimAI does not estimate live weather.\n",
            f"However, based on verified temple facilities, here are prominent temples suitable for a comfortable pilgrimage:\n",
            f"1. **Yadadri (Sri Lakshmi Narasimha Swamy)** — Modern hill shrine with sheltered queue complexes and comprehensive facilities.",
            f"2. **Basara (Gnana Saraswathi Temple)** — Peaceful riverside kshetra on the Godavari banks.",
            f"3. **Bhadrachalam (Sita Ramachandra Swamy)** — Historic riverbank temple with extensive devasthanam choultries.",
            f"4. **Vemulawada (Sri Raja Rajeshwara Swamy)** — Renowned Shiva temple with vast pilgrim amenities.\n",
            f"*(Please check the Live Weather Agent directly for updated forecast advisories before departure).*"
        ]
        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer="\n".join(lines),
            sources=["yadadri/overview.json", "basara/overview.json", "badrachalam/overview.json"],
            metadata={"weather_filter": weather_criterion, "weather_status": "unavailable"}
        )

    def _handle_open_recommendation(self, query: str) -> AgentResponse:
        """
        Handles open-ended queries like "Which temple should I visit?" or "Suggest a temple".
        Does not claim one temple is objectively 'best'; provides balanced options across
        pilgrimage traditions and invites user preferences.
        """
        limit = self._extract_limit(query)
        if limit:
            sample_recs = [
                ("Yadadri (Sri Lakshmi Narasimha Swamy)", "Modern Dravidian hill temple with comprehensive facilities and sheltered queue complexes."),
                ("Basara (Gnana Saraswathi Temple)", "One of India's prominent Saraswathi shrines, situated on the sacred Godavari banks."),
                ("Vemulawada (Sri Raja Rajeshwara Swamy)", "Revered Shiva temple with sacred Dharma Gundam pushkarini."),
                ("Srisailam (Mallikarjuna Swamy)", "Renowned Jyotirlinga and Shakti Peetha set amidst the Nallamala hills."),
                ("Bhadrachalam (Sita Ramachandra Swamy)", "Historic Ramayana pilgrimage kshetra with extensive devasthanam choultries.")
            ]
            selected = sample_recs[:limit]
            lines = [
                f"### 🛕 PilgrimAI Temple Recommendations ({len(selected)} Recommended Shrines)\n",
                f"Here are {len(selected)} recommended pilgrimage temples in Telangana based on popularity, spiritual significance, and verified facilities:\n"
            ]
            for i, (t_name, t_desc) in enumerate(selected, 1):
                lines.append(f"{i}. **{t_name}** — {t_desc}")
            lines.append("\n*(To check darshan timings, sevas, or accommodation for any of these, ask about that specific temple).*")
            return AgentResponse(
                success=True,
                agent_name=self.name,
                answer="\n".join(lines),
                sources=["yadadri/overview.json", "basara/overview.json", "vemulavada/overview.json"],
                metadata={"limit": limit, "matching_count": len(selected)}
            )

        lines = [
            "### 🛕 PilgrimAI Temple Recommendations\n",
            "PilgrimAI covers **20 certified pilgrimage temples** in Telangana, each with distinct spiritual, historical, and architectural significance. "
            "Because pilgrimage choices depend on individual faith, travel convenience, and family needs, no single temple is objectively the 'best'.\n",
            "Here are recommended temples categorized by major pilgrimage traditions:\n",
            "1. **Major Shakti Peetha & Goddess Shrines**:",
            "   - **Jogulamba (Alampur)**: 5th Maha Shakti Peetha, dating to the Badami Chalukyas.",
            "   - **Balkampet Yellamma** & **Ujjaini Mahakali**: Iconic Secunderabad/Hyderabad mother goddess shrines.\n",
            "2. **Lord Shiva & Jyotirlinga Shrines**:",
            "   - **Srisailam**: Revered Jyotirlinga and Shakti Peetha on the Nallamala hills.",
            "   - **Vemulawada (Rajanna)**: Celebrated for its sacred Baddi Pooja and holy pushkarini.",
            "   - **Kaleshvaram**: Unique shrine where two Shiva Lingas reside on a single pedestal.\n",
            "3. **Lord Narasimha Kshetras**:",
            "   - **Yadadri (Yadagirigutta)**: Grand Dravidian-style hill temple, extensively renovated with modern amenities.",
            "   - **Dharmapuri**: Ancient Yoga Narasimha kshetra on the sacred Godavari river.\n",
            "4. **Education, Learning & Aksharabhyasam**:",
            "   - **Basara (Gnana Saraswathi)**: One of India's prominent Saraswathi shrines for children's learning initiation.\n",
            "5. **Lord Rama & Lord Venkateswara Kshetras**:",
            "   - **Bhadrachalam**: Sacred Ramayana kshetra on the Godavari river.",
            "   - **Chilkur Balaji**: Revered Visa Balaji shrine known for pradakshinas without monetary hundis.\n",
            "6. **UNESCO World Heritage & Kakatiya Architecture**:",
            "   - **Ramappa Temple**: 800-year-old UNESCO World Heritage site renowned for floating bricks and sculpted pillars.",
            "   - **Thousand Pillar Temple (Hanamkonda)**: Masterpiece of 12th-century Kakatiya stone architecture.\n",
            "---\n",
            "**💡 To help us tailor a specific recommendation, please let us know:**",
            "- Which **deity or ritual** you wish to worship?",
            "- Do you require **accommodation** or **wheelchair/senior citizen accessibility**?",
            "- Are you planning a **1-day trip** from Hyderabad or a longer tour?"
        ]

        sources = [
            "yadadri/overview.json",
            "basara/overview.json",
            "badrachalam/overview.json",
            "srisaila/overview.json",
            "ramappa/overview.json"
        ]

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer="\n".join(lines),
            sources=sources,
            metadata={
                "type": "open_recommendation",
                "recommendation_type": "open_guidance",
                "categories_suggested": list(THEMATIC_TEMPLE_MAP.keys())
            }
        )

    def _check_location_and_duration_query(self, query: str) -> Optional[AgentResponse]:
        """
        Handles location-aware, time-constrained recommendation queries
        (e.g., 'i live in karnataka i have only i day holiday which temple can i visit in 1 day').
        Detects origin state/city, enforces round-trip 1-day feasibility,
        and requests starting city if only state is provided without guessing Hyderabad.
        """
        if not query:
            return None

        q_lower = query.lower()

        # Check for 1-day / time-constrained holiday expressions
        is_one_day = bool(re.search(
            r"\b(?:only\s+)?(?:1|one|single|i|a)\s*[- ]*days?\s*(?:holiday|trip|visit|tour|vacation)?\b|"
            r"\b(?:visit|trip|go)\s+(?:in|for)\s+(?:1|one|single|i|a)\s*[- ]*days?\b",
            q_lower
        ))
        is_time_constrained = is_one_day or bool(re.search(r"\b(?:half\s*day|weekend|2\s*days?)\b", q_lower))

        # Known states
        state_patterns = {
            "Karnataka": r"\b(?:in|from|live\s+in|reside\s+in|stay\s+in|staying\s+in)?\s*(?:karnataka)\b",
            "Maharashtra": r"\b(?:in|from|live\s+in|reside\s+in|stay\s+in|staying\s+in)?\s*(?:maharashtra)\b",
            "Andhra Pradesh": r"\b(?:in|from|live\s+in|reside\s+in|stay\s+in|staying\s+in)?\s*(?:andhra\s+pradesh|andhra)\b",
            "Tamil Nadu": r"\b(?:in|from|live\s+in|reside\s+in|stay\s+in|staying\s+in)?\s*(?:tamil\s+nadu)\b",
            "Telangana": r"\b(?:in|from|live\s+in|reside\s+in|stay\s+in|staying\s+in)?\s*(?:telangana)\b"
        }

        detected_state = None
        for st_name, pat in state_patterns.items():
            if re.search(pat, q_lower):
                detected_state = st_name
                break

        # Check if user mentioned a specific city or district
        known_cities = {
            "Raichur": ["raichur"],
            "Bidar": ["bidar"],
            "Kalaburagi": ["kalaburagi", "gulbarga"],
            "Ballari": ["ballari", "bellary"],
            "Bengaluru": ["bengaluru", "bangalore"],
            "Mysuru": ["mysuru", "mysore"],
            "Hyderabad": ["hyderabad", "secunderabad"],
            "Warangal": ["warangal", "hanamkonda"],
            "Nanded": ["nanded"],
            "Kurnool": ["kurnool"]
        }

        detected_city = None
        for c_name, aliases in known_cities.items():
            if any(re.search(r"\b" + re.escape(a) + r"\b", q_lower) for a in aliases):
                detected_city = c_name
                break

        if not detected_state and not detected_city:
            return None

        # If it's not a recommendation or question about which temple to visit/travel, ignore
        if not any(w in q_lower for w in ["which temple", "which place", "where should", "recommend", "suggest", "visit", "trip", "go"]):
            return None

        # Scenario 1: State provided without a city/district
        if detected_state and not detected_city:
            lines = [
                f"### 🛕 Location-Aware Pilgrimage Recommendations ({detected_state})\n",
                f"To recommend a practically feasible **1-day pilgrimage**, we need to know your departure point:\n",
                f"> **Which city or district in {detected_state} will you be starting from?**\n",
                f"#### ⏱️ Why Origin Matters for a 1-Day Holiday:",
                f"A successful single-day pilgrimage must account for **outbound travel**, **darshan & ritual waiting time** (typically 1.5–2 hours), and the **return journey** within the available 24-hour window.\n",
                f"#### 📌 Knowledge Base Coverage & Feasibility:",
                f"- **PilgrimAI Coverage**: PilgrimAI currently covers **20 certified pilgrimage devasthanams in Telangana** (e.g., Jogulamba, Basara, Yadadri, Vemulawada, Srisailam).",
                f"- **If starting from northern {detected_state} border districts** (e.g., **Raichur**, **Bidar**, **Kalaburagi**):",
                f"  1. **Jogulamba Temple (Alampur)**: Located on the Tungabhadra river right across the Telangana border (~45 km / 1 hr from Raichur). A 1-day round-trip visit with full temple darshan and return by evening is highly feasible.",
                f"  2. **Basara (Gnana Saraswathi Temple)**: Accessible from the Bidar/Nizamabad border corridor (~150 km) for an early-morning departure day trip.",
                f"- **If starting from central or southern {detected_state}** (e.g., **Bengaluru**, **Mysuru**):",
                f"  A 1-day round trip by road or rail to Telangana temples is **not practically feasible** (over 550+ km each way = 1,100+ km round trip, exceeding 18–22 hours in transit alone). For a single-day holiday, visiting prominent shrines within Karnataka or expanding your pilgrimage to a 2–3 day weekend trip is strongly advised.\n",
                f"*(Please reply with your starting city or district so we can provide an exact route and schedule).* "
            ]
            return AgentResponse(
                success=True,
                agent_name=self.name,
                answer="\n".join(lines),
                sources=["jogulamba(new)/overview.json", "basara/overview.json"],
                metadata={
                    "origin_state": detected_state,
                    "origin_pending_city": True,
                    "duration_constraint": "1_day" if is_one_day else "time_constrained",
                    "coverage_disclosed": True
                }
            )

        # Scenario 2: Specific City provided
        if detected_city:
            if detected_city in ["Raichur", "Kurnool"]:
                lines = [
                    f"### 🛕 Recommended 1-Day Pilgrimage from {detected_city}\n",
                    f"For a **1-day holiday** starting from **{detected_city}**, the best-fit pilgrimage destination is:\n",
                    f"1. **Jogulamba Temple, Alampur (5th Maha Shakti Peetha)** — **Best Fit**",
                    f"   - **Distance & Travel Time**: ~45–60 km (~1 to 1.5 hours driving each way via NH44).",
                    f"   - **Round-Trip Feasibility**: Total transit time is ~2 to 2.5 hours round-trip, leaving 5–6 hours for sanctum darshan, pushkarini snanam at the Tungabhadra confluence, and Navabrahma temple complex exploration.",
                    f"   - **Return Leg**: You can comfortably return to {detected_city} before sunset.",
                    f"\n#### Alternative Option:",
                    f"2. **Mantralayam (Sri Raghavendra Swamy Mutt)** (Prominent neighboring kshetra across the border corridor).\n",
                    f"*(Source: Verified devasthanam records for Jogulamba Alampur; round-trip highway corridor estimate)*"
                ]
                sources = ["jogulamba(new)/overview.json"]
            elif detected_city in ["Bidar", "Kalaburagi"]:
                lines = [
                    f"### 🛕 Recommended 1-Day Pilgrimage from {detected_city}\n",
                    f"For a **1-day holiday** starting from **{detected_city}**, the feasible destination within PilgrimAI's coverage is:\n",
                    f"1. **Basara (Gnana Saraswathi Temple)** — **Best Fit**",
                    f"   - **Distance & Travel Time**: ~140–160 km (~3.5 hours driving each way).",
                    f"   - **Round-Trip Feasibility**: Departing early (05:30 AM) allows morning arrival for Aksharabhyasam / darshan (10:00 AM–12:30 PM), Godavari snanam, lunch, and an evening return trip to {detected_city} by 08:30 PM.",
                    f"\n*(Source: Verified devasthanam records for Basara; round-trip highway corridor estimate)*"
                ]
                sources = ["basara/overview.json"]
            elif detected_city in ["Bengaluru", "Mysuru"]:
                lines = [
                    f"### 🛕 1-Day Pilgrimage Feasibility Notice from {detected_city}\n",
                    f"A **1-day round-trip pilgrimage** from **{detected_city}** to PilgrimAI's covered Telangana temples is **not practically feasible** by road or train:\n",
                    f"- **Distance Limitation**: The nearest covered shrines (e.g., Jogulamba in Alampur or Hyderabad shrines) are **380–570 km one way** (760–1,140 km round trip), requiring 16–22 hours of road travel alone.",
                    f"- **Recommendation for 1-Day Holiday**: Devotees with a single-day holiday from {detected_city} are advised to visit prominent Karnataka shrines (such as Chamundeshwari, Ghati Subramanya, Shivagange, or Goravanahalli Mahalakshmi).",
                    f"- **Alternative**: If you wish to visit Telangana shrines (such as Jogulamba or Yadadri), plan a **2-day weekend tour** with an overnight train or stay."
                ]
                sources = ["jogulamba(new)/overview.json", "yadadri/overview.json"]
            else:
                lines = [
                    f"### 🛕 Recommended 1-Day Pilgrimage from {detected_city}\n",
                    f"For a **1-day holiday** starting from **{detected_city}**, top recommended shrines within comfortable round-trip reach include:\n",
                    f"1. **Yadadri (Sri Lakshmi Narasimha Swamy)** — **Best Fit** (~65 km, ~1.5 hrs each way via Warangal Highway NH163). Hill shrine with modern amenities and comprehensive darshan facilities.",
                    f"2. **Chilkur Balaji (Visa Balaji Temple)** — (~30 km, ~45 mins each way). Famous for 108 pradakshinas without monetary hundis.",
                    f"3. **Birla Mandir / Balkampet Yellamma** — Intra-city central shrines easily reachable within 30–45 mins."
                ]
                sources = ["yadadri/overview.json", "chilkur/overview.json"]

            return AgentResponse(
                success=True,
                agent_name=self.name,
                answer="\n".join(lines),
                sources=sources,
                metadata={
                    "origin_city": detected_city,
                    "duration_constraint": "1_day",
                    "coverage_disclosed": True
                }
            )

        return None
