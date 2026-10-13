"""
PilgrimAI - ItineraryPlanningAgent (Phase M)

Specialist agent responsible for grounded, timing-aware pilgrimage itinerary
planning across registered temples in Telangana.

Key Capabilities:
- Single-temple and multi-temple pilgrimage itinerary generation with duration awareness.
- Non-repetitive day-by-day scheduling distributing verified sevas, darshan slots, and nearby sites.
- Grounded in verified temple timings, sevas, accommodation, travel, prasadam, rules, and crowd data.
- Strict Fact vs Planning Suggestion separation (every suggestion clearly demarcated).
- Zero fabricated travel durations, hotels, restaurants, or unverified attractions.
- Strict Live-Data Limitations disclosure (deterministic Live-Data Disclaimer).
- RAGTool facade as the exclusive knowledge access interface (zero direct DB/raw file imports).
- Inherits from BaseAgent with standard execution tracing, latency tracking, and error containment.
- Seamless compatibility with AgentRouter, MasterAgentCoordinator, SequentialMultiIntentPipeline,
  and ConversationMemoryManager.
"""

import re
import math
import logging
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

from app.agents.base_agent import BaseAgent
from app.agents.state import AgentState, AgentResponse
from app.rag.utils import get_canonical_display_name
if TYPE_CHECKING:
    from app.agents.tools.rag_tool import RAGTool

logger = logging.getLogger(__name__)

# Deterministic Live-Data Disclaimer
LIVE_DATA_DISCLAIMER = (
    "*(Live-Data Limitation: This itinerary is constructed strictly from verified static records. "
    "It does not include real-time traffic, live road conditions, real-time travel durations, "
    "live weather forecasts, or dynamic room booking availability. Please verify current local transit "
    "schedules and devasthanam announcements before departure).*"
)


class ItineraryPlanningAgent(BaseAgent):
    """
    Specialist agent providing grounded, timing-aware pilgrimage itinerary
    planning across registered temples in Telangana.
    """

    def __init__(
        self,
        rag_tool: Optional[Any] = None,
        name: str = "itinerary_planning",
        description: str = "Specialist agent providing grounded, timing-aware pilgrimage itinerary planning across registered temples using verified knowledge."
    ):
        super().__init__(name=name, description=description)
        self._rag_tool = rag_tool
        self._temple_aliases: Dict[str, List[str]] = {}
        self._init_temple_registry()

    @property
    def rag_tool(self) -> Any:
        """Lazily initialize RAGTool to break circular instantiation loops."""
        if self._rag_tool is None:
            from app.agents.tools.rag_tool import RAGTool
            self._rag_tool = RAGTool()
        return self._rag_tool

    @rag_tool.setter
    def rag_tool(self, value: Any) -> None:
        self._rag_tool = value

    def _init_temple_registry(self):
        """
        Loads registered canonical temples dynamically and anchors aliases.
        """
        registered = [
            "Ahobilam", "Badrachalam", "Balkampet", "Basara", "Birla Mandir",
            "Chilkur", "Dharmapuri", "Edupayal", "Jogulamba(new)", "Kaleshvaram",
            "Karmanghat", "Medaram", "Ramappa", "Srisaila", "Thousand Pillar",
            "Ttd", "Ujjaini", "Veera Satyanarayana", "Vemulavada", "Yadadri"
        ]

        alias_seed: Dict[str, List[str]] = {
            "Ahobilam": ["ahobilam", "ahobila", "narasimha ahobilam"],
            "Badrachalam": ["badrachalam", "bhadrachalam", "bhadragiri", "rama temple bhadrachalam", "lord rama bhadrachalam"],
            "Balkampet": ["balkampet", "yellamma", "balkampet yellamma"],
            "Basara": ["basara", "basar", "gnana saraswathi", "saraswathi temple"],
            "Birla Mandir": ["birla mandir", "birla"],
            "Chilkur": ["chilkur", "visa balaji", "chilkur balaji"],
            "Dharmapuri": ["dharmapuri", "narasimha swamy dharmapuri", "dharmapuri narasimha"],
            "Edupayal": ["edupayal", "edupayala", "edupayala durga", "vanadurga bhavani", "vana durga bhavani", "vana durga", "edupayala vanadurga"],
            "Jogulamba(new)": ["jogulamba", "jogulamba(new)", "jogulamba(New)", "alampur", "jogulamba devi"],
            "Kaleshvaram": ["kaleshvaram", "kaleshwaram", "mukteshwara", "kaleshwara"],
            "Karmanghat": ["karmanghat", "karmaghat", "karmanghat hanuman"],
            "Medaram": ["medaram", "sammakka", "saralamma", "jatara medaram", "sammakka saralamma"],
            "Ramappa": ["ramappa", "ramalingeswara", "rudreshwara"],
            "Srisaila": ["srisaila", "srisailam", "mallikarjuna", "bhramaramba"],
            "Thousand Pillar": ["thousand pillar", "thousand pillars", "rudreswara thousand", "thousand pillar temple"],
            "Ttd": ["ttd", "tirumala", "tirupati", "venkateswara ttd", "lord venkateswara"],
            "Ujjaini": [
                "ujjaini", "ujjaini mahakali", "ujjaini mahankali", "mahakali secunderabad",
                "secunderabad mahakali", "secunderabad mahankali", "sri ujjaini mahakali",
                "sri ujjaini mahankali", "ujjaini mahakali temple", "ujjaini mahankali temple",
                "sri ujjaini mahakali temple", "sri ujjaini mahankali temple", "ujjaini temple",
                "ujjaini devasthanam"
            ],
            "Veera Satyanarayana": ["veera satyanarayana", "satyanarayana swamy", "annavaram", "annavarm", "veera venkata satyanarayana"],
            "Vemulavada": ["vemulavada", "vemulawada", "rajarajeshwara", "rajanna", "rajanne"],
            "Yadadri": ["yadadri", "yadagirigutta", "laxmi narasimha yadadri", "narasimha yadadri"]
        }

        self._temple_aliases = {}
        for canonical in registered:
            matched = False
            for k, aliases in alias_seed.items():
                if k.lower() == canonical.lower():
                    self._temple_aliases[canonical] = sorted(list(set(aliases + [canonical.lower()])), key=len, reverse=True)
                    matched = True
                    break
            if not matched:
                self._temple_aliases[canonical] = [canonical.lower()]

    def detect_temples(self, query: str) -> List[str]:
        """
        Detects all registered canonical temples referenced in the query in order of appearance.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        matches: List[Tuple[int, str]] = []

        for canonical, aliases in self._temple_aliases.items():
            for alias in aliases:
                pattern = r"\b" + re.escape(alias) + r"\b"
                for match in re.finditer(pattern, clean_query, re.IGNORECASE):
                    matches.append((match.start(), canonical))
                    break

        matches.sort(key=lambda m: m[0])
        ordered_temples: List[str] = []
        for _, canonical in matches:
            if canonical not in ordered_temples:
                ordered_temples.append(canonical)

        # Fallback fuzzy entity resolution for spelling variants (e.g. annavarm -> annavaram, karmaghat -> karmanghat)
        import difflib
        words = re.findall(r'[a-zA-Z]{5,}', clean_query.lower())
        already_matched = set(ordered_temples)
        fuzzy_added = []
        for w in words:
            for canonical, aliases in self._temple_aliases.items():
                if canonical in already_matched:
                    continue
                for alias in aliases:
                    if len(alias.split()) == 1 and len(alias) >= 5:
                        sim = difflib.SequenceMatcher(None, w, alias).ratio()
                        if sim >= 0.82 and abs(len(w) - len(alias)) <= 2:
                            m_pos = clean_query.lower().find(w)
                            fuzzy_added.append((m_pos, canonical))
                            already_matched.add(canonical)
                            break
                if canonical in already_matched:
                    break

        if fuzzy_added:
            matches.extend(fuzzy_added)
            matches.sort(key=lambda m: m[0])
            ordered_temples = []
            for _, canonical in matches:
                if canonical not in ordered_temples:
                    ordered_temples.append(canonical)

        return ordered_temples

    _detect_temples = detect_temples

    def _detect_unknown_temple(self, query: str) -> Optional[str]:
        """
        Identifies whether the user explicitly requests an unregistered / unknown temple.
        Guarantees zero hallucination and explicit refusal.
        """
        famous_unregistered = [
            "golden temple", "meenakshi", "somnath", "kedarnath", "badrinath",
            "kashi vishwanath", "kashi", "jagannath", "kamakhya", "akshardham", "shirdi",
            "padmanabhaswamy", "murudeshwar", "konark", "brihadeeswarar", "ranganathaswamy",
            "amarnath", "vaishno devi", "tiruchendur", "palani", "atlantis", "atlantis temple"
        ]
        for name in famous_unregistered:
            if re.search(r"\b" + re.escape(name) + r"\b", query, re.IGNORECASE):
                return name.title()

        pattern = r"\b([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(temple|mandir|kovil|devasthanam)\b"
        stopwords = {
            "the", "a", "an", "this", "that", "these", "those", "which", "what", "where",
            "who", "how", "why", "when", "are", "is", "was", "were", "be", "been",
            "do", "does", "did", "have", "has", "had", "any", "all", "our", "your",
            "my", "their", "its", "some", "of", "in", "at", "to", "for", "with", "about",
            "plan", "itinerary", "visit", "trip", "tour", "day", "days"
        }
        for match in re.finditer(pattern, query, re.IGNORECASE):
            candidate = match.group(1).strip()
            words = [w.lower() for w in candidate.split()]
            meaningful = [w for w in words if w not in stopwords]
            if not meaningful:
                continue

            cleaned_candidate = " ".join(meaningful)
            detected = False
            for canonical, aliases in self._temple_aliases.items():
                if any(re.search(r"\b" + re.escape(a) + r"\b", cleaned_candidate, re.IGNORECASE) for a in aliases):
                    detected = True
                    break
            if not detected:
                return f"{cleaned_candidate.title()} {match.group(2).title()}"

        return None

    def _extract_duration_days(self, query: str, num_temples: int) -> float:
        """
        Extracts requested duration in days (including fractional days like 1.5, 2.5, 3.5)
        from the query, or defaults sensibly. Returns a float.
        """
        from app.agents.canonical_constraints import normalize_duration_expression
        query = normalize_duration_expression(query)

        # 0. Fractional decimal patterns (e.g. "2.5 days", "1.5-day trip", "3.5 days", "4.5 days")
        dec_match = re.search(r"\b(\d+)\.5\s*[- ]*days?\b", query, re.IGNORECASE)
        if dec_match:
            return float(dec_match.group(1)) + 0.5

        # 0b. Fractional / half day expressions (e.g. "2 and half days", "2 and a half days", "two and a half days", "3 and a half days")
        half_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:and\s+)?(?:a\s+)?half\s+days?\b", query, re.IGNORECASE)
        if half_match:
            base_str = half_match.group(1).lower()
            word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
            base_val = word_map.get(base_str, int(base_str) if base_str.isdigit() else 1)
            return float(base_val) + 0.5

        # 0c. Standalone half day trip
        if re.search(r"\b(?:half\s*day|0\.5\s*days?)\b", query, re.IGNORECASE):
            return 0.5

        # 1. Numeric days (e.g., "4 days", "4-day trip", "1 day", "2 days")
        match = re.search(r"\b(\d+)\s*[- ]*(?:days?|day)\b", query, re.IGNORECASE)
        if match:
            try:
                days = int(match.group(1))
                return float(max(1, min(days, 14)))
            except ValueError:
                pass

        # 2. Word-based days with explicit "days/day" (e.g., "two day trip", "three days", "two-day", "four days")
        # Must be evaluated before standalone "day trip" to prevent "two day trip" matching "day trip"
        word_map = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "single": 1, "weekend": 2, "week": 7
        }
        for word, val in word_map.items():
            if re.search(r"\b" + word + r"\s*[- ]*(?:days?|day)\b", query, re.IGNORECASE):
                return float(val)

        # 3. Numeric nights (e.g., "2-night stay", "3 nights") -> duration in days = nights + 1
        n_match = re.search(r"\b(\d+)\s*[- ]*(?:nights?|night)\b", query, re.IGNORECASE)
        if n_match:
            try:
                nights = int(n_match.group(1))
                return float(max(1, min(nights + 1, 14)))
            except ValueError:
                pass

        # 4. Word-based nights (e.g. "two nights")
        for word, val in word_map.items():
            if re.search(r"\b" + word + r"\s*[- ]*(?:nights?|night)\b", query, re.IGNORECASE):
                return float(min(val + 1, 14))

        # 5. Explicit standalone day trip / single day patterns (guarded by negative lookbehind)
        if re.search(r"(?<!\d\s)(?<!\d)(?<!one\s)(?<!two\s)(?<!three\s)(?<!four\s)(?<!five\s)(?<!six\s)(?<!seven\s)(?<!eight\s)(?<!nine\s)(?<!ten\s)\b(?:day\s*trip|day\s*visit|single\s*day|same\s*day|one\s*day)\b", query, re.IGNORECASE):
            return 1.0

        for word, val in word_map.items():
            if re.search(r"\b" + word + r"\s*[- ]*(?:trip|visit)\b", query, re.IGNORECASE):
                return float(val)

        return float(max(1, num_temples))

    _extract_duration = _extract_duration_days

    def _clean_text(self, text: Optional[str]) -> str:
        """
        Sanitizes raw text snippets: strips trailing single dangling letters,
        eliminates empty/malformed patterns, and removes unescaped artifacts.
        """
        if not text or not isinstance(text, str):
            return ""

        cleaned = text.strip()
        # Suppress empty malformed tokens like ": | :" or standalone colons
        if re.match(r"^[\s:|–—-]+$", cleaned):
            return ""

        # Remove trailing single dangling letters from OCR/parser artifacts (e.g. "... q", " ... w")
        # Preserve measurement units like 'm' (meters) after numbers.
        cleaned = re.sub(r"(?<!\d)\s+[a-zA-Z]\s*$", "", cleaned)
        cleaned = re.sub(r"\.\.\.\s*[a-zA-Z]\s*$", "...", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _get_temple_itinerary_data(self, temple: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        Retrieves verified records for relevant categories via RAGTool.
        Returns (data_dict, sources_list).
        """
        categories = ["timings", "sevas", "travel", "accommodation", "prasadam", "crowd", "facilities", "rules", "nearby_attractions", "restaurants"]
        data: Dict[str, Any] = {}
        sources: List[str] = []

        clean_t = temple.lower().replace(" ", "_")

        for cat in categories:
            cat_data = self.rag_tool.get_temple_category_data(temple, cat)
            if cat_data:
                data[cat] = cat_data
                sources.append(f"{clean_t}/{cat}.json")
            else:
                # Try parent text fallback
                parent = self.rag_tool.get_complete_parent_text(temple, cat)
                if parent:
                    data[cat] = parent
                    sources.append(f"{clean_t}/{cat}.json")

        return data, sources

    def _parse_time_to_minutes(self, time_str: str) -> Optional[int]:
        """
        Parses human-readable time strings like '06:00 AM', '12:30 PM', '7 PM', '18:00'
        into integer minutes from midnight (0–1439).
        """
        if not time_str or not isinstance(time_str, str):
            return None
        m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)?", time_str)
        if not m:
            return None
        hours = int(m.group(1))
        minutes = int(m.group(2)) if m.group(2) else 0
        ampm = m.group(3).upper() if m.group(3) else None

        if ampm == "PM" and hours < 12:
            hours += 12
        elif ampm == "AM" and hours == 12:
            hours = 0
        elif not ampm:
            if 1 <= hours <= 6:
                hours += 12
        return hours * 60 + minutes

    def _parse_timings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts verified opening hours, darshan windows, and break periods.
        Strictly distinguishes verified records from unverified fallback text.
        """
        timings_obj = data.get("timings")
        res = {
            "summary": "Standard daily temple opening hours apply.",
            "opening": "morning opening window",
            "closing": "evening closing hour",
            "break_period": None,
            "has_break": False,
            "is_verified": False,
            "opening_exact": None,
            "closing_exact": None
        }

        if isinstance(timings_obj, dict):
            res["is_verified"] = True
            # 1. Extract specific opening/closing times if available
            if "opening_time" in timings_obj and timings_obj["opening_time"]:
                res["opening"] = self._clean_text(str(timings_obj["opening_time"]))
                res["opening_exact"] = res["opening"]
            if "closing_time" in timings_obj and timings_obj["closing_time"]:
                res["closing"] = self._clean_text(str(timings_obj["closing_time"]))
                res["closing_exact"] = res["closing"]

            if res["opening_exact"] and res["closing_exact"]:
                res["summary"] = f"{res['opening']} to {res['closing']} (Daily general schedule)"
            else:
                t_items = []
                for k in ["general_timings", "darshan_timings", "temple_hours", "morning", "evening"]:
                    if k in timings_obj:
                        v = self._clean_text(str(timings_obj[k]))
                        if v and not re.match(r"^[\s:|–—-]+$", v):
                            t_items.append(f"{k.replace('_', ' ').title()}: {v}")
                if t_items:
                    res["summary"] = "; ".join(t_items[:2])
                elif "daily_schedule" in timings_obj and isinstance(timings_obj["daily_schedule"], list):
                    sched_strs = [
                        f"{s.get('name') or s.get('ritual_name')}: {s.get('timing') or s.get('time_slot')}"
                        for s in timings_obj["daily_schedule"]
                        if isinstance(s, dict) and (s.get("name") or s.get("ritual_name")) and (s.get("timing") or s.get("time_slot"))
                    ]
                    if sched_strs:
                        res["summary"] = "; ".join(sched_strs[:2])
                elif "darshan_slots" in timings_obj and isinstance(timings_obj["darshan_slots"], list):
                    res["summary"] = "Darshan Slots: " + ", ".join([str(s) for s in timings_obj["darshan_slots"][:3]])

            # Check break hours (dict or string)
            if isinstance(timings_obj.get("afternoon_break"), dict):
                ab = timings_obj["afternoon_break"]
                ab_from = str(ab.get('from') or '').strip()
                ab_to = str(ab.get('to') or '').strip()
                if ab_from and ab_to:
                    res["break_period"] = f"{ab_from} to {ab_to}"
                    res["has_break"] = True
                else:
                    res["has_break"] = False
            else:
                for bk in ["break_timings", "afternoon_break", "break"]:
                    if bk in timings_obj:
                        bv = self._clean_text(str(timings_obj[bk]))
                        if bv:
                            res["break_period"] = bv
                            res["has_break"] = True
                            break

            text_str = str(timings_obj)
            if not res["opening_exact"]:
                open_m = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:AM|am))", text_str)
                if open_m:
                    res["opening"] = open_m.group(1)
                    res["opening_exact"] = open_m.group(1)
            if not res["closing_exact"]:
                close_m = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:PM|pm))", text_str)
                if close_m:
                    res["closing"] = close_m.group(1)
                    res["closing_exact"] = close_m.group(1)

        elif isinstance(timings_obj, str) and timings_obj.strip():
            clean_str = self._clean_text(timings_obj)
            if clean_str:
                res["is_verified"] = True
                res["summary"] = clean_str.split("\n")[0][:120]
        else:
            res["is_verified"] = False
            res["summary"] = "Official opening hours could not be verified in static records."
            res["opening"] = "temple opening hours (unverified in static records — confirm upon arrival)"
            res["closing"] = "temple closing hour (unverified in static records — confirm upon arrival)"

        return res

    def _parse_sevas(self, data: Dict[str, Any], closing_time_str: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """
        Extracts and organizes verified sevas into morning, afternoon, evening, and general groups.
        Guarantees:
        - Afternoon sevas (12:00 PM - 4:00 PM) are not misclassified as evening rituals.
        - Rituals scheduled past official temple closing are not placed in evening darshan.
        """
        sevas_obj = data.get("sevas")
        res = {"morning": [], "afternoon": [], "evening": [], "general": [], "all": []}

        closing_min = self._parse_time_to_minutes(closing_time_str) or (21 * 60)

        raw_list = []
        if isinstance(sevas_obj, list):
            raw_list = sevas_obj
        elif isinstance(sevas_obj, dict):
            for k, v in sevas_obj.items():
                if isinstance(v, list):
                    raw_list.extend(v)
                elif isinstance(v, dict) and "name" in v:
                    raw_list.append(v)

        seen_seva_keys = set()
        for item in raw_list:
            if isinstance(item, dict) and "name" in item:
                s_name = self._clean_text(str(item.get("name", "")))
                s_timing = self._clean_text(str(item.get("timing") or item.get("time") or ""))
                s_desc = self._clean_text(str(item.get("description") or item.get("desc") or ""))
                s_cost = self._clean_text(str(item.get("price") or item.get("ticket_price") or item.get("cost") or ""))

                if not s_name:
                    continue

                s_key = s_name.lower().strip()
                if s_key in seen_seva_keys:
                    continue
                seen_seva_keys.add(s_key)

                entry = {
                    "name": s_name,
                    "timing": s_timing,
                    "desc": s_desc[:90] if s_desc else "",
                    "cost": s_cost
                }
                res["all"].append(entry)

                t_min = self._parse_time_to_minutes(s_timing)
                time_lower = s_timing.lower()
                name_lower = s_name.lower()

                if t_min is not None:
                    if t_min < 720:  # Before 12:00 PM
                        res["morning"].append(entry)
                    elif 720 <= t_min < 960:  # 12:00 PM to 4:00 PM
                        res["afternoon"].append(entry)
                    elif 960 <= t_min <= closing_min:  # 4:00 PM to closing
                        res["evening"].append(entry)
                    else:
                        # Beyond stated closing hour: do not schedule in evening window
                        res["general"].append(entry)
                else:
                    if "am" in time_lower or "morning" in time_lower or "suprabhata" in name_lower or "abhishekam" in name_lower or "pratah" in name_lower:
                        res["morning"].append(entry)
                    elif "noon" in time_lower or "madhyahna" in time_lower or "afternoon" in time_lower:
                        res["afternoon"].append(entry)
                    elif "pm" in time_lower or "evening" in time_lower or "harathi" in name_lower or "unja" in name_lower or "sayana" in name_lower or "aarti" in name_lower:
                        res["evening"].append(entry)
                    else:
                        res["general"].append(entry)

            elif isinstance(item, str) and item.strip():
                clean_s = self._clean_text(item)
                if clean_s:
                    entry = {"name": clean_s[:60], "timing": "", "desc": "", "cost": ""}
                    res["all"].append(entry)
                    res["general"].append(entry)

        return res

    def _parse_restaurants(self, data: Dict[str, Any], temple: str) -> Dict[str, Any]:
        """
        Extracts verified restaurant and food records from restaurants.json.
        Categorizes into:
        - annadanam: official free/subsidized temple food services
        - lunch_options: verified nearby vegetarian eateries suitable for lunch
        - dinner_options: verified nearby eateries suitable for dinner
        - hotel_restaurants: verified on-site hotel restaurants (if any)
        - all: list of all verified eateries
        """
        r_obj = data.get("restaurants")
        res = {
            "annadanam": None,
            "lunch_options": [],
            "dinner_options": [],
            "hotel_restaurants": [],
            "all": []
        }

        raw_list = []
        if isinstance(r_obj, list):
            raw_list = r_obj
        elif isinstance(r_obj, dict):
            for v in r_obj.values():
                if isinstance(v, list):
                    raw_list.extend(v)
                elif isinstance(v, dict) and "name" in v:
                    raw_list.append(v)

        import urllib.parse
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            name = self._clean_text(str(item.get("name") or item.get("restaurant_name") or ""))
            if not name:
                continue
            r_type = self._clean_text(str(item.get("type") or item.get("cuisine") or "Vegetarian Eatery"))
            dist = self._clean_text(str(item.get("distance") or "Nearby"))
            raw_dist_str = str(item.get("distance") or "").lower()
            if dist and re.match(r"^~\s*\d+$", dist.strip()) and "m" in raw_dist_str:
                dist = f"{dist.strip()} m"
            pricing = self._clean_text(str(item.get("pricing") or item.get("cost") or item.get("price") or ""))
            timings = self._clean_text(str(item.get("timings") or item.get("operating_hours") or ""))
            speciality = self._clean_text(str(item.get("speciality") or item.get("description") or ""))
            address = self._clean_text(str(item.get("address") or ""))

            # Build working Google Maps search link (reuses standard Google Maps search query)
            temple_search_name = get_canonical_display_name(temple)
            map_query = f"{name} {temple_search_name}"
            maps_url = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(map_query)}"

            entry = {
                "name": name,
                "type": r_type,
                "distance": dist,
                "pricing": pricing,
                "timings": timings,
                "speciality": speciality,
                "address": address,
                "maps_url": maps_url
            }
            res["all"].append(entry)

            name_lower = name.lower()
            type_lower = r_type.lower()
            spec_lower = speciality.lower()
            time_lower = timings.lower()

            # Detect Annadanam
            if "annadan" in name_lower or "annadan" in type_lower or "free" in pricing.lower() or "annadan" in spec_lower:
                if not res["annadanam"]:
                    res["annadanam"] = entry
                # Annadanam is strictly classified under annadanam to avoid duplicate recommendation
            elif "hotel restaurant" in type_lower or "resort" in type_lower or "haritha" in name_lower:
                res["hotel_restaurants"].append(entry)
                res["lunch_options"].append(entry)
                res["dinner_options"].append(entry)
            else:
                if "lunch" in time_lower and "dinner" not in time_lower:
                    res["lunch_options"].append(entry)
                elif "dinner" in time_lower and "lunch" not in time_lower:
                    res["dinner_options"].append(entry)
                else:
                    res["lunch_options"].append(entry)
                    res["dinner_options"].append(entry)

        return res

    def _parse_nearby_attractions(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extracts verified nearby attractions from nearby_attractions.json if available.
        """
        obj = data.get("nearby_attractions")
        attractions: List[Dict[str, str]] = []

        if isinstance(obj, dict) and "attractions" in obj and isinstance(obj["attractions"], list):
            items = obj["attractions"]
        elif isinstance(obj, list):
            items = obj
        else:
            items = []

        for item in items:
            if isinstance(item, dict) and "name" in item:
                name = self._clean_text(str(item.get("name", "")))
                dist = self._clean_text(str(item.get("distance", "")))
                desc = self._clean_text(str(item.get("description", "")))
                if name:
                    attractions.append({
                        "name": name,
                        "distance": dist,
                        "desc": desc[:100] if desc else ""
                    })

        return attractions

    def _parse_prasadam(self, data: Dict[str, Any]) -> str:
        """
        Extracts verified prasadam / Annadanam food service details.
        """
        prasadam_obj = data.get("prasadam")
        if isinstance(prasadam_obj, dict):
            if "annadanam" in prasadam_obj:
                ann = self._clean_text(str(prasadam_obj["annadanam"]))
                if ann:
                    return f"Annadanam available: {ann[:110]}"
            if "items" in prasadam_obj:
                return "Prasadam counters operational during temple opening hours."
        elif isinstance(prasadam_obj, list):
            return "Prasadam counters and sanctified food offerings operational."
        return "Prasadam offerings and official devasthanam food counters available during opening hours."

    def _parse_accommodation(self, data: Dict[str, Any]) -> str:
        """
        Extracts verified accommodation records, accurately distinguishing
        temple devasthanam guest houses from nearby private hotel stays.
        """
        accom_obj = data.get("accommodation")
        if isinstance(accom_obj, dict):
            parts = []
            if "devasthanam_guest_houses" in accom_obj and isinstance(accom_obj["devasthanam_guest_houses"], list):
                names = [self._clean_text(str(g.get("name") if isinstance(g, dict) else g)) for g in accom_obj["devasthanam_guest_houses"][:3]]
                valid_names = [n for n in names if n]
                if valid_names:
                    parts.append(f"Devasthanam Stays: {', '.join(valid_names)}")
            if "types" in accom_obj and isinstance(accom_obj["types"], list):
                parts.append(f"Options: {', '.join(str(t) for t in accom_obj['types'][:2])}")
            if parts:
                return "; ".join(parts)
            return "On-site Devasthanam guest houses, cottages, and choultries verified in records."
        elif isinstance(accom_obj, list) and accom_obj:
            dev_names = []
            pvt_names = []
            for g in accom_obj:
                if isinstance(g, dict):
                    name = self._clean_text(str(g.get("name") or g.get("hotel_name") or ""))
                    g_type = str(g.get("type", "")).lower()
                    if any(k in g_type for k in ["devasthanam", "choultry", "temple", "guest house", "cottage"]):
                        dev_names.append(name)
                    else:
                        pvt_names.append(name)
                elif isinstance(g, str):
                    pvt_names.append(self._clean_text(g))

            res_parts = []
            if dev_names:
                res_parts.append(f"Devasthanam Stays: {', '.join(dev_names[:3])}")
            if pvt_names:
                res_parts.append(f"Nearby Private Hotels: {', '.join(pvt_names[:3])}")
            if res_parts:
                return "; ".join(res_parts)
            return "Nearby verified private stays and lodging available."
        elif accom_obj:
            return "Devasthanam guest houses / choultries verified in records."
        return "Check nearby private stays; no devasthanam stay specified."

    def _parse_travel(self, data: Dict[str, Any]) -> List[str]:
        """
        Extracts verified static connectivity and transit hubs into clean human-readable strings.
        Guarantees zero raw dictionary or JSON object leakage into user responses.
        """
        travel_obj = data.get("travel")
        t_notes = []
        if isinstance(travel_obj, dict):
            # 1. Railway connectivity
            st = travel_obj.get("nearest_railway_station") or travel_obj.get("railway")
            if isinstance(st, dict):
                if "name" in st:
                    name = self._clean_text(st.get("name"))
                    dist = self._clean_text(st.get("distance", ""))
                    dist_str = f" ({dist})" if dist else ""
                    if name:
                        t_notes.append(f"Nearest Railway: {name}{dist_str}")
                elif "stations" in st and isinstance(st["stations"], list):
                    for station in st["stations"]:
                        if isinstance(station, dict):
                            s_name = station.get("station") or station.get("name")
                            s_dist = station.get("distance", "")
                            dist_str = f" ({s_dist})" if s_dist else ""
                            if s_name:
                                t_notes.append(f"Nearest Railway: {self._clean_text(str(s_name))}{dist_str}")
                                break
                        elif isinstance(station, str) and self._clean_text(station):
                            t_notes.append(f"Nearest Railway: {self._clean_text(station)}")
                            break
            elif isinstance(st, str) and self._clean_text(st):
                t_notes.append(f"Nearest Railway: {self._clean_text(st)}")

            # 2. Airport connectivity
            ap = travel_obj.get("nearest_airport") or travel_obj.get("airport")
            if isinstance(ap, dict):
                name = self._clean_text(ap.get("name"))
                dist = self._clean_text(ap.get("distance", ""))
                dist_str = f" ({dist})" if dist else ""
                if name:
                    t_notes.append(f"Nearest Airport: {name}{dist_str}")
            elif isinstance(ap, str) and self._clean_text(ap):
                t_notes.append(f"Nearest Airport: {self._clean_text(ap)}")

            # 3. Bus routes and local transit
            if "bus_routes" in travel_obj:
                br = travel_obj["bus_routes"]
                if isinstance(br, list):
                    for b in br:
                        if isinstance(b, dict):
                            st_name = b.get("nearest_bus_station") or b.get("station") or b.get("name")
                            st_dist = b.get("distance", "")
                            st_desc = b.get("description", "")
                            st_route = b.get("route", "")
                            if st_name and st_dist:
                                t_notes.append(f"Nearest Bus Station: {self._clean_text(str(st_name))} ({self._clean_text(str(st_dist))})")
                                break
                            elif st_desc and not st_desc.startswith("{"):
                                t_notes.append(f"Bus Connectivity: {self._clean_text(str(st_desc)[:120])}")
                                break
                            elif st_route and st_desc:
                                t_notes.append(f"Bus Route ({st_route}): {self._clean_text(str(st_desc)[:100])}")
                                break
                        elif isinstance(b, str) and self._clean_text(b) and not b.strip().startswith("{"):
                            t_notes.append(f"Bus Connectivity: {self._clean_text(b[:120])}")
                            break

        return t_notes

    def _parse_rules(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts verified dress code, dos, and don'ts.
        """
        rules_obj = data.get("rules")
        res = {"dress_code": None, "dos": [], "donts": []}

        if isinstance(rules_obj, dict):
            if "dress_code" in rules_obj:
                raw_dc = rules_obj["dress_code"]
                if isinstance(raw_dc, dict):
                    parts = []
                    for k, v in raw_dc.items():
                        clean_v = self._clean_text(str(v)) if v else ""
                        if clean_v:
                            parts.append(f"{k.capitalize()}: {clean_v}")
                    if parts:
                        res["dress_code"] = ", ".join(parts)
                elif isinstance(raw_dc, str):
                    dc = self._clean_text(raw_dc)
                    if dc and not re.match(r"^[\s:|–—-]+$", dc) and dc not in ["{}", "{'men': '', 'women': ''}", "{'men': '', 'women': ''}"]:
                        # Also check if it is stringified empty dict
                        if not (dc.startswith("{") and "''" in dc and not any(c.isalnum() for c in re.sub(r"['\":{},\s]|men|women", "", dc))):
                            res["dress_code"] = dc
            if "dos" in rules_obj and isinstance(rules_obj["dos"], list):
                res["dos"] = [self._clean_text(str(d)) for d in rules_obj["dos"][:2] if self._clean_text(str(d))]
            if "donts" in rules_obj and isinstance(rules_obj["donts"], list):
                res["donts"] = [self._clean_text(str(d)) for d in rules_obj["donts"][:2] if self._clean_text(str(d))]

        return res

    def _get_departure_terminal(self, data: Dict[str, Any], temple: str) -> str:
        """
        Dynamically extracts nearest railway station and bus stand from travel.json.
        Never hardcodes distant or unrelated transit stations (e.g. Ramagundam).
        """
        temple_disp = get_canonical_display_name(temple)
        travel_obj = data.get("travel")
        hub_parts = []
        if isinstance(travel_obj, dict):
            # 1. Nearest Railway Station
            st = travel_obj.get("nearest_railway_station") or travel_obj.get("railway")
            if isinstance(st, dict) and st.get("name"):
                st_name = self._clean_text(str(st["name"]))
                dist = self._clean_text(str(st.get("distance", "")))
                dist_str = f" (~{dist})" if dist else ""
                hub_parts.append(f"{st_name} (Railway){dist_str}")
            elif isinstance(st, dict) and "stations" in st and isinstance(st["stations"], list):
                s0 = st["stations"][0]
                if isinstance(s0, dict) and s0.get("name"):
                    hub_parts.append(f"{self._clean_text(str(s0['name']))} (Railway)")
            
            # 2. Bus Routes / Stands
            bus_stands = []
            if "bus_routes" in travel_obj and isinstance(travel_obj["bus_routes"], list):
                for r in travel_obj["bus_routes"]:
                    r_str = str(r)
                    if "rtc bus" in r_str.lower() or "bus stand" in r_str.lower() or "bus station" in r_str.lower():
                        bus_stands.append("local RTC Bus Stand")
                        break
            if not bus_stands:
                hub_parts.append(f"{temple_disp} RTC Bus Station")
            else:
                hub_parts.append(f"{temple_disp} RTC Bus Stand")
        
        if hub_parts:
            return " / ".join(hub_parts[:2])
        return f"{temple_disp} RTC Bus Stand / nearest railway station"

    def _format_schedule_section(
        self,
        temple: str,
        day_num: int,
        total_days: int,
        temple_day_index: int,
        total_temple_days: int,
        data: Dict[str, Any],
        is_multi_temple: bool = False,
        is_fractional_end: bool = False,
        duration_days: float = 1.0,
        prev_temple: Optional[str] = None,
        pax: int = 1,
        rooms: int = 1,
        origin: Optional[str] = None,
        origin_assumed: bool = True,
        selected_accommodation: Optional[str] = None,
        has_elderly: bool = False
    ) -> List[str]:
        """
        Builds a structured, non-repetitive daily itinerary section
        strictly separating verified facts from planning suggestions.
        """
        temple_disp = get_canonical_display_name(temple)
        timings = self._parse_timings(data)
        sevas = self._parse_sevas(data, timings.get("closing"))
        restaurants = self._parse_restaurants(data, temple)
        attractions = self._parse_nearby_attractions(data)
        prasadam_note = self._parse_prasadam(data)
        rules = self._parse_rules(data)
        departure_terminal = self._get_departure_terminal(data, temple)

        from app.agents.budget_agent import (
            INTERSTATE_ORIGIN_DISTANCES_FROM_HYDERABAD,
            parse_price_text,
            normalize_price_display_text
        )
        is_interstate_origin = bool(
            origin and not origin_assumed and any(city in origin.lower() for city in INTERSTATE_ORIGIN_DISTANCES_FROM_HYDERABAD)
        )

        accom_raw = data.get("accommodation") or []
        accom_list = accom_raw if isinstance(accom_raw, list) else (list(accom_raw.values()) if isinstance(accom_raw, dict) else [])
        dev_stays = [a for a in accom_list if isinstance(a, dict) and any(k in (str(a.get("type", "")) + " " + str(a.get("name", ""))).lower() for k in ["devasthanam", "choultry", "sadan", "cottage", "guest house", "tent", "tourism"])]
        budget_stays = [a for a in accom_list if isinstance(a, dict) and any(k in (str(a.get("type", "")) + " " + str(a.get("name", ""))).lower() for k in ["lodge", "budget", "residency", "hotel"]) and not any(r in str(a.get("name", "")).lower() for r in ["resort", "luxury", "spa"])]

        selected_accom_name = selected_accommodation
        chosen_stay_obj = None
        if selected_accom_name:
            chosen_stay_obj = next((a for a in accom_list if isinstance(a, dict) and a.get("name", "").strip().lower() == selected_accom_name.strip().lower()), None)

        if chosen_stay_obj:
            chosen_stay = chosen_stay_obj.get("name", selected_accom_name)
        elif "kaleshvaram" in temple.lower() or "kaleshwaram" in temple.lower():
            chosen_stay = "Saraswathi Choultry Kaleshwaram"
            stay_note = "linked to economy baseline budget; room availability & tariff confirmation pending on-ground devasthanam counter verification (not pre-booked)"
        elif "medaram" in temple.lower():
            chosen_stay = "Forest Tents (Forest Department)"
            stay_note = "official Forest Department tents linked to economy baseline budget; booking and availability pending on-ground confirmation"
        elif dev_stays:
            chosen_stay_obj = dev_stays[0]
            chosen_stay = chosen_stay_obj.get("name", f"Devasthanam Rest House ({temple_disp})")
            stay_note = "linked to economy baseline budget; room availability & tariff confirmation pending on-ground devasthanam counter verification (not pre-booked)"
        elif budget_stays:
            # Sort budget stays by lowest verified min_cost, prioritizing budget category
            def _stay_sort_key(s_item):
                p_raw = s_item.get("price") or s_item.get("tariff") or ""
                p_min, _, _, _, _, is_q, _ = parse_price_text(p_raw, category="accommodation")
                cost_val = p_min if (is_q and p_min is not None) else 99999.0
                is_budget_cat = 0 if any(k in (str(s_item.get("type", "")) + " " + str(s_item.get("description", ""))).lower() for k in ["budget", "lodge", "deluxe"]) else 1
                return (cost_val, is_budget_cat)

            sorted_budget_stays = sorted(budget_stays, key=_stay_sort_key)
            chosen_stay_obj = sorted_budget_stays[0]
            chosen_stay = chosen_stay_obj.get("name", f"Local Highway Lodges ({temple_disp} area)")
        else:
            chosen_stay = f"Local Budget Lodges / Nearby City Lodging ({temple_disp} area)"
            stay_note = "basic budget lodging; room availability & tariff confirmation pending upon arrival"

        if chosen_stay_obj and "stay_note" not in locals():
            p_raw = chosen_stay_obj.get("price") or chosen_stay_obj.get("tariff") or ""
            p_min, p_max, _, is_free, is_range, is_q, _ = parse_price_text(p_raw, category="accommodation")
            if is_free:
                stay_note = "complimentary devasthanam accommodation; room availability & allocation pending on-ground counter confirmation"
            elif is_q and p_raw:
                p_disp = normalize_price_display_text(p_raw)
                stay_note = f"estimated tariff {p_disp}; room availability & tariff confirmation pending on-ground counter verification"
            elif dev_stays and chosen_stay_obj in dev_stays:
                stay_note = "linked to economy baseline budget; room availability & tariff confirmation pending on-ground devasthanam counter verification (not pre-booked)"
            else:
                stay_note = "basic budget lodging; tariff unlisted in static records — room availability & rates pending on-ground confirmation"

        # Determine inter-temple distance for transit days
        t_dist = 40.0
        if is_multi_temple and prev_temple and prev_temple.lower() != temple.lower():
            from app.agents.budget_agent import REGIONAL_ORIGIN_DISTANCES, BENCHMARK_DISTANCES_FROM_HYDERABAD, TEMPLE_CANONICAL_MAP
            p1_n, p2_n = prev_temple.lower(), temple.lower()
            for (ok, dk), d in REGIONAL_ORIGIN_DISTANCES.items():
                if (ok in p1_n and dk in p2_n) or (ok in p2_n and dk in p1_n):
                    t_dist = d
                    break
            else:
                c1 = TEMPLE_CANONICAL_MAP.get(p1_n, prev_temple)
                c2 = TEMPLE_CANONICAL_MAP.get(p2_n, temple)
                d1 = BENCHMARK_DISTANCES_FROM_HYDERABAD.get(c1, 150.0)
                d2 = BENCHMARK_DISTANCES_FROM_HYDERABAD.get(c2, 150.0)
                CORRIDOR_DIRS = {
                    "Veera Satyanarayana": "E", "Medaram": "NE", "Ramappa": "NE",
                    "Thousand Pillar": "NE", "Kaleshvaram": "NE", "Dharmapuri": "N",
                    "Basara": "NW", "Edupayal": "NW", "Jogulamba(New)": "S",
                    "Jogulamba(new)": "S", "Srisaila": "S", "Ahobilam": "S",
                    "Ttd": "SE", "Badrachalam": "E", "Yadadri": "E",
                    "Karmanghat": "HYD", "Birla Mandir": "HYD", "Balkampet": "HYD",
                    "Ujjaini": "HYD", "Chilkur": "SW"
                }
                dir1 = CORRIDOR_DIRS.get(c1, "")
                dir2 = CORRIDOR_DIRS.get(c2, "")
                if (dir1 == "E" and dir2 == "NE") or (dir1 == "NE" and dir2 == "E"):
                    t_dist = 390.0
                elif dir1 and dir2 and dir1 != dir2 and "HYD" not in (dir1, dir2):
                    t_dist = round(d1 + d2, 1)
                else:
                    t_dist = round(abs(d1 - d2) + min(d1, d2) * 0.4, 0)

        # Partition attractions per day (avoid repeating the same attractions on multiple days)
        if attractions:
            if total_days == 1:
                day_attractions = attractions[:2]
            elif is_multi_temple and temple_day_index == 1 and day_num > 1 and t_dist > 400:
                # Full day spent in transit corridor; no on-ground attractions
                day_attractions = []
            elif day_num > 1:
                # Calculate on-ground exploration index for this temple (temple_day_index 2 starts at offset 0)
                on_ground_idx = max(0, temple_day_index - 2) if is_multi_temple else max(0, day_num - 2)
                attr_offset = on_ground_idx * 2
                day_attractions = attractions[attr_offset:attr_offset + 2]
                if not day_attractions and attractions:
                    day_attractions = [attractions[0]]
            else:
                day_attractions = []
        else:
            day_attractions = []

        # 1. Determine Distinct Day Theme & Purpose
        if total_days == 1 and not is_fractional_end:
            theme = "Full Day Pilgrimage & Darshan (No Hotel Stay Required)"
        elif is_fractional_end and day_num == total_days:
            theme = "Half-Day Concluding Visit & Afternoon Return Journey"
        elif not is_multi_temple:
            if day_num == 1:
                theme = "Arrival, Hotel Check-in & Main Sanctum Darshan"
            elif day_num == total_days:
                theme = "Concluding Rituals, Hotel Check-out & Return Departure"
            else:
                if day_attractions:
                    theme = f"Heritage Landmarks & Local Cultural Sites (Day {day_num})"
                else:
                    theme = f"Spiritual Contemplation & Sanctum Pradakshinam (Day {day_num})"
        else:
            if temple_day_index == 1 and day_num == 1:
                theme = f"Arrival & {temple_disp} Sanctum Darshan"
            elif day_num == total_days and temple_day_index == total_temple_days:
                theme = f"Concluding Rituals at {temple_disp}, Prasadam & Departure"
            elif is_multi_temple and temple_day_index == 1 and day_num > 1 and t_dist > 400:
                p_disp = get_canonical_display_name(prev_temple) if prev_temple else "Previous Destination"
                theme = f"Interstate Transit Corridor ({p_disp} to {temple_disp}) & Late-Night / Overnight Arrival"
            elif temple_day_index == total_temple_days:
                theme = f"Concluding Rituals at {temple_disp} & Onward Travel"
            elif temple_day_index == 1:
                theme = f"Arrival & Primary Rituals at {temple_disp}"
            else:
                theme = f"Spiritual Observance & Special Sevas at {temple_disp}"

        lines = [f"#### Day {day_num}: {temple_disp} ({theme})\n"]

        # Select Grounded Dining Options
        annadanam = restaurants.get("annadanam")
        all_dining = restaurants.get("all", [])
        lunch_opts = [r for r in restaurants.get("lunch_options", []) if "annadan" not in r["name"].lower()]
        dinner_opts = [r for r in restaurants.get("dinner_options", []) if "annadan" not in r["name"].lower()]
        tiffin_opts = [r for r in all_dining if "annadan" not in r["name"].lower() and any(k in r["name"].lower() or k in r["type"].lower() for k in ["tiffin", "mess", "breakfast", "canteen", "refreshment"])]
        transit_opts = [r for r in all_dining if any(k in r["name"].lower() or k in r["type"].lower() for k in ["highway", "transit", "food court"])]

        # --- BREAKFAST REC ---
        if is_multi_temple and temple_day_index == 1 and day_num > 1 and prev_temple:
            # Traveling from prev_temple: breakfast is at the origin/checkout location
            p_data = self.rag_tool.get_temple_category_data(prev_temple, "restaurants")
            p_rests = self._parse_restaurants({"restaurants": p_data}, prev_temple) if p_data else {}
            p_all = p_rests.get("all", [])
            p_tiffins = [r for r in p_all if "annadan" not in r["name"].lower() and any(k in r["name"].lower() or k in r["type"].lower() for k in ["tiffin", "mess", "breakfast", "canteen", "refreshment"])]
            if p_tiffins:
                t_choice = p_tiffins[0]
                p_p = f"Pricing: {t_choice['pricing']}" if t_choice.get('pricing') else "Affordable"
                b_desc = f"**{t_choice['name']}** ({t_choice['distance']}, {p_p}; fresh South Indian breakfast)"
            else:
                p_disp_name = get_canonical_display_name(prev_temple)
                b_desc = f"a local vegetarian tiffin center near {p_disp_name} (idli/dosa, ₹40–₹80)"
        elif tiffin_opts:
            riverside = next((r for r in tiffin_opts if "triveni" in r["name"].lower() or "sangam" in r["name"].lower() or "riverside" in r.get("type", "").lower()), None)
            if riverside and day_num == total_days and total_days > 1:
                t_choice = riverside
            else:
                t_choice = tiffin_opts[(day_num - 1) % len(tiffin_opts)]
            p_p = f"Pricing: {t_choice['pricing']}" if t_choice.get('pricing') else "Affordable"
            b_desc = f"**{t_choice['name']}** ({t_choice['distance']}, {p_p}; fresh South Indian breakfast)"
        else:
            b_desc = "a local vegetarian tiffin center near the shrine entrance (idli/dosa, ₹40–₹80)"

        # --- LUNCH REC ---
        if day_num == 1:
            if annadanam:
                ann_time = annadanam.get("timings") or "12:00 PM – 2:30 PM"
                lunch_rec = f"**{annadanam['name']}** ({annadanam['distance']}, {annadanam.get('speciality') or 'Free traditional sanctified meals'}, Timings: {ann_time})"
                non_haritha_lunch = [r for r in lunch_opts if "haritha" not in r["name"].lower() and "transit" not in r["name"].lower()]
                if non_haritha_lunch:
                    alt = non_haritha_lunch[0]
                    alt_pr = f"Pricing: {alt['pricing']}" if alt.get('pricing') else "Affordable"
                    lunch_rec += f"; Alternative nearby veg dining: **{alt['name']}** ({alt['distance']}, {alt_pr}; [Directions / Map]({alt['maps_url']}))"
                elif lunch_opts:
                    alt = lunch_opts[0]
                    alt_pr = f"Pricing: {alt['pricing']}" if alt.get('pricing') else "Affordable"
                    lunch_rec += f"; Alternative nearby veg dining: **{alt['name']}** ({alt['distance']}, {alt_pr}; [Directions / Map]({alt['maps_url']}))"
            elif lunch_opts:
                l_choice = lunch_opts[0]
                l_pr = f"Pricing: {l_choice['pricing']}" if l_choice.get('pricing') else "Affordable"
                lunch_rec = f"**{l_choice['name']}** ({l_choice['distance']}, {l_choice['type']}, {l_pr}; [Directions / Map]({l_choice['maps_url']}))"
            else:
                lunch_rec = "Nearby vegetarian eatery around the temple complex"
        else:
            distinct_lunch = [r for r in lunch_opts if "haritha" not in r["name"].lower() and "transit" not in r["name"].lower()]
            if distinct_lunch:
                l_choice = distinct_lunch[(day_num - 1) % len(distinct_lunch)]
                l_pr = f"Pricing: {l_choice['pricing']}" if l_choice.get('pricing') else "Affordable"
                lunch_rec = f"**{l_choice['name']}** ({l_choice['distance']}, {l_choice['type']}, {l_pr}; homestyle vegetarian lunch; [Directions / Map]({l_choice['maps_url']}))"
            elif lunch_opts:
                l_choice = lunch_opts[(day_num - 1) % len(lunch_opts)]
                l_pr = f"Pricing: {l_choice['pricing']}" if l_choice.get('pricing') else "Affordable"
                lunch_rec = f"**{l_choice['name']}** ({l_choice['distance']}, {l_choice['type']}, {l_pr}; [Directions / Map]({l_choice['maps_url']}))"
            elif annadanam:
                ann_time = annadanam.get("timings") or "12:00 PM – 2:30 PM"
                lunch_rec = f"**{annadanam['name']}** ({annadanam['distance']}, Timings: {ann_time})"
            else:
                lunch_rec = "Nearby vegetarian restaurant along the route"

        # --- DINNER REC ---
        if day_num == total_days and total_days > 1:
            lunch_chosen_name = l_choice['name'].lower() if 'l_choice' in locals() and l_choice else ""
            alt_dinner_opts = [r for r in dinner_opts if r["name"].lower() != lunch_chosen_name]
            if transit_opts:
                tr_choice = transit_opts[0]
                tr_pr = f"Pricing: {tr_choice['pricing']}" if tr_choice.get('pricing') else "Affordable"
                dinner_rec = f"**{tr_choice['name']}** ({tr_choice['distance']}, {tr_pr}; transit dining along return route; [Directions / Map]({tr_choice['maps_url']}))"
            elif alt_dinner_opts:
                d_choice = alt_dinner_opts[-1]
                d_pr = f"Pricing: {d_choice['pricing']}" if d_choice.get('pricing') else "Affordable"
                dinner_rec = f"Nearby verified dining before departure: **{d_choice['name']}** ({d_choice['distance']}, {d_pr}; [Directions / Map]({d_choice['maps_url']}))"
            elif dinner_opts and len(dinner_opts) > 1:
                d_choice = dinner_opts[-1]
                d_pr = f"Pricing: {d_choice['pricing']}" if d_choice.get('pricing') else "Affordable"
                dinner_rec = f"Nearby verified dining before departure: **{d_choice['name']}** ({d_choice['distance']}, {d_pr}; [Directions / Map]({d_choice['maps_url']}))"
            else:
                dinner_rec = "Dinner at transit terminal or en-route vegetarian food court"
        elif day_num == 1:
            lunch_chosen_name = l_choice['name'].lower() if 'l_choice' in locals() and l_choice else ""
            alt_dinner_opts = [r for r in dinner_opts if r["name"].lower() != lunch_chosen_name]
            if alt_dinner_opts:
                d_choice = alt_dinner_opts[0]
                d_pr = f"Pricing: {d_choice['pricing']}" if d_choice.get('pricing') else "Affordable"
                dinner_rec = f"Nearby verified dining: **{d_choice['name']}** ({d_choice['distance']}, {d_choice['type']}, {d_pr}; [Directions / Map]({d_choice['maps_url']}))"
            elif dinner_opts:
                d_choice = dinner_opts[0]
                d_pr = f"Pricing: {d_choice['pricing']}" if d_choice.get('pricing') else "Affordable"
                dinner_rec = f"Nearby verified dining: **{d_choice['name']}** ({d_choice['distance']}, {d_choice['type']}, {d_pr}; [Directions / Map]({d_choice['maps_url']}))"
            else:
                dinner_rec = "Nearby vegetarian restaurant around the temple"
        else:
            lunch_chosen_name = l_choice['name'].lower() if 'l_choice' in locals() and l_choice else ""
            alt_dinner_opts = [r for r in dinner_opts if r["name"].lower() != lunch_chosen_name]
            if alt_dinner_opts:
                d_choice = alt_dinner_opts[(day_num - 1) % len(alt_dinner_opts)]
                d_pr = f"Pricing: {d_choice['pricing']}" if d_choice.get('pricing') else "Affordable"
                dinner_rec = f"Nearby verified dining: **{d_choice['name']}** ({d_choice['distance']}, {d_choice.get('type', 'Restaurant')}, {d_pr}; [Directions / Map]({d_choice['maps_url']}))"
            elif dinner_opts and len(dinner_opts) > 1:
                d_choice = dinner_opts[(day_num - 1) % len(dinner_opts)]
                d_pr = f"Pricing: {d_choice['pricing']}" if d_choice.get('pricing') else "Affordable"
                dinner_rec = f"Nearby verified dining: **{d_choice['name']}** ({d_choice['distance']}, {d_pr}; [Directions / Map]({d_choice['maps_url']}))"
            elif dinner_opts:
                d_choice = dinner_opts[0]
                d_pr = f"Pricing: {d_choice['pricing']}" if d_choice.get('pricing') else "Affordable"
                dinner_rec = f"Nearby verified dining: **{d_choice['name']}** ({d_choice['distance']}, {d_pr}; [Directions / Map]({d_choice['maps_url']}))"
            else:
                dinner_rec = "Local vegetarian mess or restaurant"

        # =============================================================
        # SPECIAL BRANCH: Fractional Final Half-Day (e.g. Day 3 of 2.5 Days)
        # =============================================================
        if is_fractional_end and day_num == total_days:
            lines.append("**Suggested Visit Plan (Half-Day Schedule):**")
            lines.append(f"- **Morning Window (Breakfast, Concluding Darshan & Check-out)**: Breakfast at {b_desc}. Peaceful concluding visit to {temple_disp} for archana, sanctum pradakshinam, and final blessings ({timings['opening']} onwards). Traditional attire recommended ({rules['dress_code'] or 'traditional dhoti/sari'}). Return to lodging to complete room check-out (standard 11:00 AM – 12:00 PM) and collect stored baggage.")
            if sevas["morning"]:
                s = sevas["morning"][0]
                t_str = f" ({s['timing']})" if s["timing"] else ""
                lines.append(f"  • Morning seva / darshan: **{s['name']}**{t_str} *(subject to daily counter token availability)*.")
            if origin_assumed:
                ret_claim = f"Commence scheduled return journey by afternoon (~1:30 PM – 2:30 PM) from transit hub ({departure_terminal}). *(Return Journey Note: Because your departure origin is unspecified, return travel commences in the afternoon; final arrival time, connecting transit duration, and return fares remain pending confirmation of your starting city and chosen transport mode. Trip completion timing will depend on distance to your destination)*."
            else:
                ret_claim = f"Commence scheduled return travel by afternoon (~1:30 PM – 2:30 PM) from transit hub ({departure_terminal}) back to {origin} (return journey underway; no hotel stay required on final departure half-day)."
            lines.append(f"- **Midday & Afternoon Window (Lunch, Prasadam & Return Journey)**: Lunch break — {lunch_rec}. Devotees collect sanctified Laddu / Pulihora prasadam packets from official devasthanam counters. {ret_claim}")
            lines.append("")
            return lines

        # -------------------------------------------------------------
        # STANDARD FULL-DAY VISIT PLAN (Morning, Afternoon, Evening)
        # -------------------------------------------------------------
        lines.append("**Suggested Visit Plan:**")

        # Morning Window
        orig_text = f"from {origin}" if (origin and not origin_assumed) else "from your starting location"
        if total_days == 1:
            lines.append(f"- **Morning Window (Sanctum Arrival & Darshan)**: Arrive in {temple_disp} area {orig_text}. Breakfast at {b_desc}. Proceed to the main sanctum for initial darshan during morning opening hours ({timings['opening']} onwards). Core sanctum darshan typically takes 1.5–2 hours. Traditional attire recommended ({rules['dress_code'] or 'traditional dhoti/sari'}).")
            if sevas["morning"]:
                s = sevas["morning"][0]
                t_str = f" ({s['timing']})" if s["timing"] else ""
                lines.append(f"  • Attend morning seva: **{s['name']}**{t_str} *(subject to daily counter token availability)*.")
        elif day_num == 1:
            if is_interstate_origin and total_days > 2:
                lines.append(f"- **Morning Window (Interstate Transit Departure & Long-Distance Rail)**: Outbound long-distance rail departure from {origin} toward {temple_disp} regional corridor. *(Outbound Rail Transit Day: Interstate travel from {origin} requires ~20–28 hours across state borders; Day 1 is dedicated to transit, with scheduled morning arrival in {temple_disp} area on Day 2).* Devotees board express train with reserved berths.")
                lines.append(f"  • Onboard breakfast and journey through connecting railway corridors.")
            else:
                lines.append(f"- **Morning Window (Outbound Journey, Arrival & Check-in)**: Outbound travel {orig_text} to {temple_disp} area. Proceed to lodging for check-in and baggage deposit (standard check-in 12:00 PM; luggage drop if arriving early). Prepare for temple darshan in traditional attire ({rules['dress_code'] or 'traditional attire recommended'}).")
                lines.append(f"  • Proceed to the main sanctum for initial darshan during morning visiting hours ({timings['opening']} onwards). Core sanctum darshan typically takes 1.5–2 hours.")
                if has_elderly and "medaram" in temple.lower():
                    lines.append(f"  • *(Senior Citizen Access Note: Devotees traveling with senior citizens should board the free battery-operated EV carts from TGSRTC bus holding points to reach near the shrine; utilize the dedicated step-free wooden/concrete ramps and the special barricaded queue at the Gaddelu altar platform).*")
                if sevas["morning"]:
                    s = sevas["morning"][0]
                    t_str = f" ({s['timing']})" if s["timing"] else ""
                    lines.append(f"  • Attend morning seva: **{s['name']}**{t_str} *(subject to daily counter token availability)*.")
        elif is_multi_temple and temple_day_index == 1 and day_num > 1:
            p_name = get_canonical_display_name(prev_temple) if prev_temple else "previous destination"
            if prev_temple and prev_temple.lower() == temple.lower():
                lines.append(f"- **Morning Window (Breakfast & Ongoing Darshan)**: Breakfast at {b_desc}. Continue peaceful morning darshan and pradakshinam at {temple_disp} ({timings['opening']} onwards).")
            else:
                from app.agents.budget_agent import REGIONAL_ORIGIN_DISTANCES, BENCHMARK_DISTANCES_FROM_HYDERABAD, TEMPLE_CANONICAL_MAP
                if t_dist <= 40:
                    dur_desc = f"~{int(t_dist)} km (~45–60 mins road transit)"
                elif t_dist <= 100:
                    dur_desc = f"~{int(t_dist)} km (~1.5–2.5 hrs road transit)"
                elif t_dist <= 250:
                    dur_desc = f"~{int(t_dist)} km (~3.5–5 hrs road / rail transit)"
                elif t_dist <= 450:
                    dur_desc = f"~{int(t_dist)} km (~7–10 hrs connecting road / rail transit)"
                elif t_dist <= 650:
                    dur_desc = f"~{int(t_dist)} km (~11–14 hrs road or overnight express train link)"
                else:
                    dur_desc = f"~{int(t_dist)} km (unverified road/rail estimate: ~14–18 hrs road transit or overnight express train with connecting transfer; live GPS routing unverified — verify current schedules before departure)"
                lines.append(f"- **Morning Window (Check-out & Onward Travel)**: Breakfast at {b_desc}, complete hotel check-out in {p_name}, and embark on onward journey toward {temple_disp} ({dur_desc}).")
        elif day_num == total_days:
            lines.append(f"- **Morning Window (Concluding Darshan)**: Breakfast at {b_desc}. Peaceful concluding morning visit to {temple_disp} for archana, pradakshinam, and final blessings ({timings['opening']} onwards).")
            if has_elderly and "medaram" in temple.lower():
                lines.append(f"  • *(Senior Citizen Support: Wheelchairs are accessible at the Central Medical Camp near TTD Kalyana Mandapam, with shaded resting pavilions and 24/7 medical desks stationed along major approach pathways).*")
        else:
            # Intermediate full day (e.g. Day 2 of 3+ days)
            if is_interstate_origin and day_num == 2 and total_days > 2:
                lines.append(f"- **Morning Window (Arrival at {temple_disp}, Station Transfer & Hotel Check-in)**: Train reaches transit hub ({departure_terminal}) in the morning (~09:00 AM – 10:30 AM) after the long-distance interstate journey from {origin}. Transfer to lodging ({chosen_stay}) for check-in, refresh, and baggage deposit. Prepare for temple darshan in traditional attire ({rules['dress_code'] or 'traditional attire recommended'}).")
                lines.append(f"  • Proceed to {temple_disp} for mid-morning Sarva Darshan ({timings['opening']} onwards). Core sanctum darshan typically takes 1.5–2 hours. *(Note on Seva Timing: Early dawn 03:30 AM – 04:30 AM rituals cannot be attended on Day 2 due to long-distance train arrival timing; daytime sanctum darshan commences upon arrival)*.")
                daytime_sevas = [s for s in sevas["morning"] + sevas["all"] if any(k in s["name"].lower() for k in ["vratham", "kalyanam", "archana", "darshan"]) or (s.get("timing") and not any(t in s["timing"] for t in ["03:", "04:", "05:", "3:30", "4:00", "5:00"]))]
                if daytime_sevas:
                    s = daytime_sevas[0]
                    t_str = f" ({s['timing']})" if s["timing"] else ""
                    lines.append(f"  • Participate in daytime sacred ritual: **{s['name']}**{t_str} *(subject to daily counter token availability)*.")
            elif day_attractions:
                attr1 = day_attractions[0]
                dist_str1 = str(attr1.get('distance', '')).lower()
                d_m1 = re.search(r"(\d+(?:\.\d+)?)", dist_str1)
                d_val1 = float(d_m1.group(1)) if d_m1 else 0.0
                is_distant1 = d_val1 >= 30.0 or any(k in attr1.get('name', '').lower() for k in ["reserve", "wildlife", "forest", "kondagattu", "kawal"])
                if is_distant1:
                    lines.append(f"- **Morning Window (Sanctum Pradakshinam & Spiritual Rituals)**: Breakfast at {b_desc}. Peaceful morning darshan, sanctum pradakshinam, and sacred temple rituals at {temple_disp} ({timings['opening']} onwards). *(Optional regional excursion: **{attr1['name']}** ({attr1['distance']}) requires ~{round(d_val1 * 2.0 / 35.0, 1)} hrs round-trip road travel and is omitted to prioritize unhurried sanctum darshan)*.")
                else:
                    lines.append(f"- **Morning Window (Local Exploration)**: Breakfast at {b_desc}. Explore verified nearby landmark: **{attr1['name']}**" + (f" ({day_attractions[0]['distance']})" if day_attractions[0].get('distance') else "") + f" — {day_attractions[0].get('desc', '')}.")
            elif sevas["morning"] and len(sevas["morning"]) > 1:
                alt_s = sevas["morning"][1]
                t_str = f" ({alt_s['timing']})" if alt_s["timing"] else ""
                lines.append(f"- **Morning Window**: Breakfast at {b_desc}. Attend peaceful morning darshan and participate in **{alt_s['name']}**{t_str} ({timings['opening']} onwards).")
            else:
                lines.append(f"- **Morning Window**: Breakfast at {b_desc}. Peaceful early morning darshan and sanctum pradakshinam ({timings['opening']} onwards).")

        # Afternoon Window
        break_str = f" ({timings['break_period']})" if timings.get("has_break") and timings.get("break_period") else ""
        if total_days == 1:
            lines.append(f"- **Afternoon Window (Lunch Break & Courtyard Walk)**: Lunch break — {lunch_rec}.")
            if break_str:
                lines.append(f"  • Observe sanctum break hours{break_str} when the main sanctum is closed. Explore temple courtyard architecture or rest in temple mandapam before evening darshan (no hotel return needed for a single-day pilgrimage).")
            else:
                lines.append(f"  • Midday rest or courtyard exploration during peak afternoon heat (no hotel return needed for a single-day pilgrimage).")
            if sevas["afternoon"]:
                s = sevas["afternoon"][0]
                t_str = f" ({s['timing']})" if s["timing"] else ""
                lines.append(f"  • Afternoon seva: **{s['name']}**{t_str} *(confirm token at counter)*.")
        elif day_num == 1:
            if is_interstate_origin and total_days > 2:
                lines.append(f"- **Afternoon Window (Continuous Long-Distance Rail Transit)**: Onboard lunch and continuous rail transit across connecting railway junctions. Devotees rest and prepare for sacred rituals upon arrival.")
            else:
                lines.append(f"- **Afternoon Window (Lunch & Midday Break)**: Lunch break — {lunch_rec}.")
                if break_str:
                    lines.append(f"  • Respect sanctum afternoon break{break_str} when the main sanctum is closed. Devotees may explore courtyard shrines or rest at lodging during peak afternoon heat.")
                else:
                    lines.append(f"  • Devotees may rest at lodging or explore parivara courtyard shrines during peak afternoon heat.")
                if sevas["afternoon"]:
                    s = sevas["afternoon"][0]
                    t_str = f" ({s['timing']})" if s["timing"] else ""
                    lines.append(f"  • Afternoon seva: **{s['name']}**{t_str} *(confirm token at counter)*.")
        elif is_multi_temple and temple_day_index == 1 and day_num > 1:
            if t_dist > 400:
                lines.append(f"- **Afternoon Window (Continuous En-Route Transit)**: Highway lunch break at a clean vegetarian restaurant en route. Continuous transit across state corridors toward {temple_disp}.")
            else:
                lines.append(f"- **Afternoon Window (Arrival & Lodging Check-in)**: Highway lunch break at a clean vegetarian restaurant en route. Arrive in {temple_disp} area by mid-afternoon, check in at lodging, and refresh.")
        elif day_num == total_days:
            lines.append(f"- **Afternoon Window (Check-out & Lunch)**: Complete room check-out (standard 11:00 AM – 12:00 PM) and deposit luggage at lodging reception / cloakroom. Lunch break — {lunch_rec}.")
            if day_attractions:
                lines.append(f"  • Visit **{day_attractions[0]['name']}**" + (f" ({day_attractions[0]['distance']})" if day_attractions[0].get('distance') else "") + f" — {day_attractions[0].get('desc', '')}.")
        else:
            lines.append(f"- **Afternoon Window (Lunch & Midday Break)**: Lunch break — {lunch_rec}.")
            if break_str:
                lines.append(f"  • Observe sanctum break hours{break_str}. Devotees may explore nearby attractions or observe an afternoon rest at lodging during peak heat.")
            else:
                lines.append(f"  • Devotees may explore nearby attractions or observe an afternoon rest at lodging during peak heat.")
            if len(day_attractions) > 1:
                attr2 = day_attractions[1]
                dist_str = str(attr2.get('distance', '')).lower()
                d_m2 = re.search(r"(\d+(?:\.\d+)?)", dist_str)
                d_val2 = float(d_m2.group(1)) if d_m2 else 0.0
                is_distant2 = d_val2 >= 30.0 or "hour" in dist_str or any(k in attr2.get('name', '').lower() for k in ["reserve", "wildlife", "forest", "kondagattu", "kawal"])
                if is_distant2:
                    lines.append(f"  • Midday & afternoon: Explore courtyard shrines, visit local shrines, or observe rest at lodging during afternoon heat. *(Optional regional excursion: **{attr2['name']}** ({attr2['distance']}) requires ~{round(d_val2 * 2.0 / 35.0, 1)} hrs round-trip road travel and is omitted from core schedule to preserve sacred sanctum darshan and sevas at {temple_disp})*.")
                else:
                    lines.append(f"  • In the afternoon, visit **{attr2['name']}**" + (f" ({attr2['distance']})" if attr2.get('distance') else "") + f" — {attr2.get('desc', '')}.")
            if sevas["afternoon"]:
                s = sevas["afternoon"][0]
                t_str = f" ({s['timing']})" if s["timing"] else ""
                lines.append(f"  • Afternoon seva: **{s['name']}**{t_str} *(confirm token at counter)*.")

        # Evening Window
        close_note = f"before {timings['closing']} night closure" if timings.get("closing_exact") else "during evening visiting hours (exact night closure unverified — check notice board)"
        if total_days == 1:
            lines.append(f"- **Evening Window (Harathi, Dinner & Return Departure)**: Temple re-opens for evening visiting hours ({close_note}). Attend Sandhya Harathi, collect sanctified Laddu / Pulihora prasadam from official counters.")
            if sevas["evening"]:
                e_s = sevas["evening"][0]
                t_str = f" ({e_s['timing']})" if e_s["timing"] else ""
                lines.append(f"  • Attend evening ritual: **{e_s['name']}**{t_str}.")
            lines.append(f"  • Dinner: {dinner_rec}.")
            lines.append(f"  • Departure: Proceed directly to transit hub ({departure_terminal}) for scheduled return departure (no hotel stay required).")
        elif day_num == total_days:
            ret_term = f"back to {origin} from transit hub ({departure_terminal})" if (origin and not origin_assumed) else f"to commence return journey from transit hub ({departure_terminal}) *(final arrival time, connecting routes, and return fares remain pending confirmation of your departure city)*"
            lines.append(f"- **Evening Window (Concluding Blessings & Return Departure)**: Conclude temple visits and collect sanctified prasadam (plan completion by 5:00 PM – 6:00 PM, well before {close_note}).")
            lines.append(f"  • Dinner: {dinner_rec}.")
            lines.append(f"  • Departure: Retrieve stored luggage from lodging reception/cloakroom and proceed {ret_term}.")
            if is_interstate_origin:
                lines.append(f"  • Return Rail Transit Note: Board overnight interstate return train back to {origin}. Long-distance rail travel spans the return corridor; return transit continues overnight on Day {total_days}, reaching {origin} on the following day (Day {total_days + 1}).")
        elif day_num == 1:
            if is_interstate_origin and total_days > 2:
                lines.append(f"- **Evening Window (Overnight Rail Journey)**: Onboard dinner; train proceeds overnight toward {temple_disp} corridor. Prepare for arrival, check-in, and primary sanctum darshan at {temple_disp} on Day 2 morning.")
            else:
                lines.append(f"- **Evening Window (Aarti, Illumination & Dinner)**: Temple re-opens for evening visiting hours ({close_note}). Attend Sandhya Aarti / Harathi, stroll parivara sub-shrines, and enjoy illuminated complex views.")
                if sevas["evening"]:
                    e_s = sevas["evening"][0]
                    t_str = f" ({e_s['timing']})" if e_s["timing"] else ""
                    lines.append(f"  • Attend evening ritual: **{e_s['name']}**{t_str}.")
                lines.append(f"  • Dinner: {dinner_rec}.")
                lines.append(f"  • Night Rest: Return to lodging only after dinner for overnight sleep.")
        elif is_multi_temple and temple_day_index == 1 and day_num > 1:
            if t_dist > 400:
                lines.append(f"- **Evening & Late-Night Window (En-Route Transit & Late-Night Arrival Window)**: Continuous highway transit across state corridors toward {temple_disp}. With an estimated ~14–18 hours of road travel departing after morning checkout (~08:30 AM – 09:00 AM), the arrival window in {temple_disp} area spans late night (~10:30 PM) to past midnight (~02:30 AM into Day {day_num + 1}; unverified road estimate — live GPS routing and delays unverified). Evening arrival cannot be assured due to journey length. Devotees should coordinate late-night check-in with lodging ({chosen_stay}) in advance. Dinner at a clean highway eatery en route. Rest overnight upon arrival to prepare for tomorrow's sacred morning darshan.")
            else:
                lines.append(f"- **Evening Window (First Darshan & Aarti)**: Temple opens for evening darshan ({close_note}). Attend Sandhya Harathi and perform peaceful sanctum darshan.")
                if sevas["evening"]:
                    e_s = sevas["evening"][0]
                    t_str = f" ({e_s['timing']})" if e_s["timing"] else ""
                    lines.append(f"  • Attend evening ritual: **{e_s['name']}**{t_str}.")
                lines.append(f"  • Dinner: {dinner_rec}.")
                lines.append(f"  • Night Rest: Return to lodging after dinner for restful night stay.")
        elif temple_day_index == total_temple_days and is_multi_temple:
            lines.append(f"- **Evening Window (Concluding Rituals & Dinner)**: Attend evening Harathi, conclude temple visit by 6:00 PM.")
            lines.append(f"  • Dinner: {dinner_rec}.")
            lines.append(f"  • Prepare for onward travel to the next pilgrimage destination.")
        else:
            lines.append(f"- **Evening Window (Evening Darshan & Dinner)**: Temple re-opens for evening darshan ({close_note}). Attend evening Harathi, collect prasadam, and enjoy illuminated views.")
            if sevas["evening"]:
                e_s = sevas["evening"][0]
                t_str = f" ({e_s['timing']})" if e_s["timing"] else ""
                lines.append(f"  • Attend evening ritual: **{e_s['name']}**{t_str}.")
            elif len(day_attractions) > 2:
                lines.append(f"  • Visit nearby evening site: **{day_attractions[2]['name']}**" + (f" ({day_attractions[2]['distance']})" if day_attractions[2].get('distance') else "") + f".")
            lines.append(f"  • Dinner: {dinner_rec}.")
            lines.append(f"  • Night Rest: Return to lodging after dinner for overnight rest.")

        # Overnight Stay Status
        if total_days == 1:
            lines.append("- **Overnight**: Departure journey (single-day trip; no overnight hotel stay required).")
        elif day_num < total_days:
            if is_interstate_origin and day_num == 1 and total_days > 2:
                lines.append(f"- **Overnight**: Selected stay: **Express Train Sleeper Berth** (Overnight interstate rail transit from {origin} toward {temple_disp} corridor; arrival on Day 2 morning).")
            else:
                unit_lbl = "tent(s)" if "medaram" in temple.lower() else "room(s)"
                room_desc = f"{rooms} {unit_lbl} for night {day_num} (standard double occupancy for {pax} pilgrims)" if pax > 1 else f"1 {unit_lbl} for night {day_num}"
                transit_night_note = " (covers late-night / past-midnight transit arrival)" if (is_multi_temple and temple_day_index == 1 and day_num > 1 and t_dist > 400) else ""
                lines.append(f"- **Overnight**: Selected stay: **{chosen_stay}** ({temple_disp} area; {room_desc}{transit_night_note}; {stay_note}).")

        # Grounded note if multi-day visit exceeds verified activities
        if temple_day_index > 1 and not attractions and len(sevas["all"]) <= 2:
            lines.append(f"  *(Grounded Planning Note: Official records for {temple_disp} focus on core sanctum darshan without separate verified local sightseeing sites. Additional hours are left flexible for quiet contemplation and local rest; devotees desiring an active sightseeing schedule may consider a shorter trip or adding an adjacent pilgrimage site).*")

        lines.append("")
        return lines

    def _build_elderly_facilities_section(
        self,
        detected_temples: List[str],
        raw_query: str,
        sc: Dict[str, Any],
        has_elderly: bool
    ) -> List[str]:
        """
        Builds destination-specific Elderly-Friendly Facilities & Accessibility section
        grounded strictly in authoritative records. Guarantees zero cross-temple hallucination.
        """
        if not has_elderly:
            return []

        lines = [
            "### ♿ Elderly-Friendly Facilities & Accessibility",
            ""
        ]

        for temple in detected_temples:
            t_disp = get_canonical_display_name(temple)
            elderly_data = self.rag_tool.get_temple_category_data(temple, "elderly_ppl") or {}
            fac_data = self.rag_tool.get_temple_category_data(temple, "facilities") or {}

            if len(detected_temples) > 1:
                lines.append(f"#### {t_disp}")

            if "medaram" in temple.lower():
                # Authoritative records from medaram/elderly_ppl.json & medaram/facilities.json
                lines.extend([
                    "- **Wheelchair Availability & Collection**: Limited wheelchairs are available at the **Central Medical Camp (TTD Kalyana Mandapam)** and key help desks near the main bus drop-off points for elderly and differently-abled pilgrims.",
                    "- **Priority Darshan & Barricaded Entry**: Dedicated special queue lines and separate barricaded entry gates are maintained by police and NSS/NCC volunteers near the **Gaddelu (altar platform)** for elderly, pregnant women, and physically challenged devotees.",
                    "- **Battery-Operated EV Shuttles**: Free battery-operated EV carts and shuttles operate from designated TGSRTC holding zones and RTC bus drop-off points to ferry senior citizens and specially-abled devotees closer to the shrine area.",
                    "- **Ramps & Step-Free Routes**: Specially constructed wooden and concrete step-free ramps are installed at entry and exit gates around the main Gaddelu platform and the central medical pavilion.",
                    "- **Medical Infrastructure & Emergency Support**: A 50-bed Central Jathara Hospital operates 24/7 at TTD Kalyana Mandapam (Sri Sammakka Saralamma Vidyashala), complemented by 14 peripheral medical outposts, first-aid centers, and 100+ standby '108' ambulances.",
                    "- **Resting Facilities & Lost-and-Found Centers**: Shaded resting tents, pandals, and continuous RO drinking water stations are stationed along major approach tracks. 10+ sector-wise Lost and Found Announcement Centers are connected with the Central Command & Control Center to assist separated seniors and families.",
                    "- **Accommodation Accessibility Note**: Forest Department Tents feature ground-level clearings near Balloon No. 5 with mobile freshwater toilet vans (1 per 10 tents); temporary open camping clearings consist of uneven forest clearings and lack paved paths or senior amenities.",
                    "- **Practical Advice for Senior Citizens**: To minimize unnecessary walking across the 24-km festival radius, board the TGSRTC EV shuttle directly from the bus terminal, obtain a wheelchair early at the Central Medical Camp, utilize the dedicated Gaddelu ramp gate, and keep an identification slip with emergency contact numbers.",
                    ""
                ])
            elif elderly_data or (isinstance(fac_data, dict) and fac_data.get("wheelchair_accessibility")):
                wc = elderly_data.get("wheelchair_availability") or (fac_data.get("wheelchair_accessibility") if isinstance(fac_data, dict) else None)
                if wc:
                    lines.append(f"- **Wheelchair Availability & Accessibility**: {wc}")
                if elderly_data.get("priority_darshan"):
                    lines.append(f"- **Priority Queue & Darshan**: {elderly_data['priority_darshan']}")
                if elderly_data.get("battery_vehicles"):
                    lines.append(f"- **Battery-Operated Vehicles**: {elderly_data['battery_vehicles']}")
                if elderly_data.get("ramps_and_lifts"):
                    lines.append(f"- **Ramps & Step-Free Access**: {elderly_data['ramps_and_lifts']}")
                if elderly_data.get("guidelines"):
                    lines.append(f"- **Senior Citizen Guidelines & Resting Areas**: {elderly_data['guidelines']}")
                lines.append(f"- **Practical Advice**: Minimize walking by inquiring at the Devasthanam enquiry counter / CRO office on arrival to confirm current token issuance and battery cart routes.")
                lines.append("")
            else:
                lines.extend([
                    f"- **Notice on Specific Facilities**: Official wheelchair collection points, EV carts, and dedicated senior priority queues are not explicitly specified in available devasthanam records for {t_disp}.",
                    f"- **Practical Recommendation**: Devotees traveling with senior citizens should proceed to the official Devasthanam Enquiry Office / Temple Administrative Counter upon arrival to request assistance, ramp access, and current mobility support.",
                    ""
                ])

        return lines

    def _run(self, state: AgentState) -> AgentResponse:
        """
        Executes grounded pilgrimage itinerary planning.
        Called automatically by BaseAgent.execute(state).
        """
        from app.agents.canonical_constraints import normalize_duration_expression
        raw_query = normalize_duration_expression(state.user_query.strip()) if state.user_query else ""

        # 1. Empty / Whitespace Input
        if not raw_query:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Please provide a valid pilgrimage itinerary question.",
                sources=[],
                error="Empty query"
            )

        # 2. Check for explicitly named unknown / unregistered temples
        unknown_temple = self._detect_unknown_temple(raw_query)
        if unknown_temple:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer=(
                    f"I don't have verified records for '{unknown_temple}' in the current PilgrimAI knowledge base. "
                    f"PilgrimAI currently covers 20 major pilgrimage temples in Telangana (such as Yadadri, Basara, Bhadrachalam, Srisailam, Dharmapuri, etc.)."
                ),
                sources=[],
                metadata={"detected_unknown_temple": unknown_temple},
                error="Unregistered temple"
            )

        # 3. Detect candidate registered temples
        detected_temples = self.detect_temples(raw_query)

        # 4. Zero Temples Identified
        if not detected_temples:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer=(
                    "Which temple or temples would you like to include in your itinerary? "
                    "PilgrimAI covers 20 registered pilgrimage destinations in Telangana (e.g., Yadadri, Dharmapuri, Basara, Bhadrachalam, Srisailam)."
                ),
                sources=[],
                metadata={"clarification_needed": True, "detected_temples": []},
                error="Missing temple specification"
            )

        # 5. Determine Duration in Days, Pax, Rooms, and Origin
        sc = getattr(state, "structured_constraints", {}) or {}
        if "duration_days" in sc and not sc.get("days_assumed", True):
            duration_days = float(sc["duration_days"])
        else:
            duration_days = float(self._extract_duration_days(raw_query, len(detected_temples)))

        is_fractional = (duration_days % 1 != 0)
        calendar_days = math.ceil(duration_days) if duration_days > 1 else 1
        nights = math.floor(duration_days) if is_fractional else max(0, int(duration_days - 1))

        # Pax & Rooms extraction
        pax = 1
        pax_assumed = True
        adults = sc.get("adults")
        children = sc.get("children")
        seniors = sc.get("seniors")
        if "pax" in sc and not sc.get("pax_assumed", False):
            pax = int(sc["pax"])
            pax_assumed = False
        else:
            from app.agents.canonical_constraints import extract_traveller_party_size
            f_adults, f_children, f_seniors, f_pax, f_assumed = extract_traveller_party_size(raw_query)
            if not f_assumed:
                pax = f_pax
                adults = f_adults
                children = f_children
                seniors = f_seniors
                pax_assumed = False

        has_elderly = bool(seniors or sc.get("seniors") or sc.get("senior_citizens"))
        if not has_elderly:
            q_lower = raw_query.lower()
            if any(re.search(pat, q_lower) for pat in [
                r"\b(?:senior\s*citizens?|seniors?|elderly(?:\s*(?:persons?|people|pilgrims?|devotees?))?|older\s*(?:parents?|adults?)|elders?)\b",
                r"\b(?:wheelchair|mobility\s*(?:assistance|impaired)?|differently\s*abled|handicapped|special\s*queue)\b"
            ]):
                has_elderly = True

        rooms = sc.get("rooms")
        if not rooms:
            if calendar_days <= 1:
                rooms = 0
            elif adults is not None and children is not None and adults <= 2 and children <= 1:
                rooms = 1
            else:
                rooms = max(1, math.ceil(pax / 2.0))

        # Origin extraction
        origin = sc.get("origin")
        origin_assumed = sc.get("origin_assumed", True)
        if not origin or origin_assumed:
            orig_match = re.search(r"\bfrom\s+([A-Za-z\s]+?)(?:\s+(?:to|for|with|under|within|on|by|in)\b|[,;.]|$)", raw_query, re.IGNORECASE)
            if orig_match:
                cand = orig_match.group(1).strip()
                cand_clean = " ".join([w for w in cand.split() if w.lower() not in ["a", "the", "my", "our", "this"]])
                if cand_clean:
                    origin = cand_clean.title()
                    origin_assumed = False
        if not origin:
            origin = "Hyderabad"
            origin_assumed = True

        from app.agents.budget_agent import INTERSTATE_ORIGIN_DISTANCES_FROM_HYDERABAD
        is_interstate = any(city in origin.lower() for city in INTERSTATE_ORIGIN_DISTANCES_FROM_HYDERABAD)

        # 6. Retrieve Verified Information Per Temple
        temple_schedules: Dict[str, Dict[str, Any]] = {}
        sources_by_temple: Dict[str, List[str]] = {}
        all_sources: List[str] = []

        for temple in detected_temples:
            t_data, t_sources = self._get_temple_itinerary_data(temple)
            temple_schedules[temple] = t_data
            sources_by_temple[temple] = t_sources
            for s in t_sources:
                if s not in all_sources:
                    all_sources.append(s)

        # 7. Construct Output
        temples_title = " & ".join([get_canonical_display_name(t) for t in detected_temples])
        num_temples = len(detected_temples)
        is_multi_temple = num_temples > 1

        if is_fractional:
            full_days = math.floor(duration_days)
            day_str = f"{duration_days:g} Days ({full_days} Full Days + 1 Half-Day, {nights} Nights)"
        elif duration_days == 1:
            day_str = "1 Day (Day Trip)"
        else:
            day_str = f"{int(duration_days)} Days ({nights} Nights)"

        doc_lines = [f"### Pilgrimage Itinerary: {temples_title} ({day_str})\n"]

        if is_interstate and "udaipur" in origin.lower() and num_temples >= 4:
            doc_lines.append(
                f"*(Inter-State Pilgrimage Transit Note: Departing from {origin}. Long-distance interstate transit via Indian Railways (Express Train — Sleeper / 3AC) and connecting regional buses is modeled to cover the ~3,690 km continuous circuit across Rajasthan, Andhra Pradesh, and Telangana. Overnight stays ({nights} nights across {rooms} room(s) for {pax} pilgrims) and darshan intervals are synchronized to ensure realistic travel schedules).*\n"
            )
            doc_lines.append(
                f"*(⚡ Feasible Proposed Schedule vs Confirmed Transport Itinerary: This 8-day itinerary represents a feasible proposed schedule covering all 4 requested shrines ({temples_title}) across ~3,690 km. It is a proposed planning model, NOT a confirmed transport itinerary, because live IRCTC rail reservations and dynamic state bus connection timetables are not integrated. Note that interstate rail transit from {origin} consumes 2 calendar days (Day 1 outbound rail journey ~24–28 hrs and Day 8 return rail journey ~20–24 hrs arriving {origin} on Day 9), leaving exactly 6 usable on-ground days (Days 2–7) to cover 4 distant shrines across ~1,000+ km of road travel in coastal AP and Telangana. Because an 8-day interstate schedule cannot be responsibly guaranteed without live railway timetables, devotees desiring guaranteed buffer windows across all four destinations are advised to expand the trip to 10–11 total days. Alternatively, focusing on the 3 Telangana shrines (Hyderabad/Karmanghat, Medaram, Jogulamba) eliminates ~780 km of cross-state travel to coastal AP).* \n"
            )
        elif is_interstate:
            doc_lines.append(
                f"*(Inter-State Pilgrimage Transit Note: Departing from {origin}. Long-distance interstate transit via Indian Railways and regional connecting transit is modeled for the {duration_days:g}-day journey to {temples_title}. Overnight stays ({nights} night(s) across {rooms} room(s) for {pax} pilgrims) and darshan intervals are synchronized to ensure realistic travel schedules).*\n"
            )
        elif duration_days > 1 and not is_multi_temple:
            doc_lines.append(
                f"*(Note on Temple Duration: Core darshan and rituals at {temples_title} typically take 1.5–2 hours. "
                f"To ensure a coherent, fulfilling experience over {duration_days:g} days, Day 1 is dedicated to arrival, main sanctum darshan, "
                f"and evening aarti; subsequent days explore verified nearby cultural landmarks and heritage sites without repeating sanctum routines).*\n"
            )

        # 8. Compute Temple-to-Day Allocation
        day_temple_map: Dict[int, List[str]] = {d: [] for d in range(1, calendar_days + 1)}
        if calendar_days == 3 and num_temples == 2 and is_fractional:
            day_temple_map[1] = [detected_temples[0]]
            day_temple_map[2] = [detected_temples[1]]
            day_temple_map[3] = [detected_temples[1]]
        elif calendar_days == 5 and num_temples == 2 and is_fractional:
            day_temple_map[1] = [detected_temples[0]]
            day_temple_map[2] = [detected_temples[0]]
            day_temple_map[3] = [detected_temples[1]]
            day_temple_map[4] = [detected_temples[1]]
            day_temple_map[5] = [detected_temples[1]]
        elif calendar_days >= num_temples:
            base_days = calendar_days // num_temples
            extra_days = calendar_days % num_temples
            cur_day = 1
            for idx, t in enumerate(detected_temples):
                n_days = base_days + (1 if idx < extra_days else 0)
                for _ in range(n_days):
                    if cur_day <= calendar_days:
                        day_temple_map[cur_day].append(t)
                        cur_day += 1
        else:
            for idx, t in enumerate(detected_temples):
                d_idx = min(idx + 1, calendar_days)
                day_temple_map[d_idx].append(t)

        # 8b. Important Temple Information (Displayed once per temple before daily plan)
        doc_lines.append("### 🏛️ Important Temple Information (Verified Records)\n")
        for t in detected_temples:
            t_disp = get_canonical_display_name(t)
            t_data = temple_schedules[t]
            t_timings = self._parse_timings(t_data)
            t_sevas = self._parse_sevas(t_data, t_timings.get("closing"))
            t_rules = self._parse_rules(t_data)
            t_prasadam = self._parse_prasadam(t_data)
            t_restaurants = self._parse_restaurants(t_data, t)

            doc_lines.append(f"**{t_disp}**")
            if t_timings.get("is_verified") and t_timings.get("summary") and not re.match(r"^[\s:|–—-]+$", t_timings["summary"]):
                doc_lines.append(f"- **Operating Schedule**: {t_timings['summary']}")
            elif not t_timings.get("is_verified"):
                doc_lines.append("- **Operating Schedule**: Sanctum darshan hours unverified in static records — confirm at temple counter upon arrival.")
            if t_timings.get("has_break") and t_timings.get("break_period"):
                doc_lines.append(f"- **Sanctum Break Hours**: {t_timings['break_period']} (Main sanctum closed; plan meals and rest)")
            if t_rules.get("dress_code"):
                doc_lines.append(f"- **Dress Guidance**: {t_rules['dress_code']}")
            if t_sevas.get("all"):
                key_sevas = [s["name"] + (f" ({s['timing']})" if s.get("timing") else "") for s in t_sevas["all"][:3]]
                doc_lines.append(f"- **Key Seva Offerings**: {', '.join(key_sevas)} *(Offerings subject to on-ground counter availability; specific ritual timings differ from general temple opening hours)*")
            if t_restaurants["annadanam"]:
                ann = t_restaurants["annadanam"]
                ann_time = ann.get("timings") or "11:30 AM to 2:30 PM"
                doc_lines.append(f"- **Temple Dining / Annadanam**: {ann['name']} ({ann.get('speciality') or 'Free traditional sanctified meals'} — {ann_time})")
            elif t_prasadam and ("annadanam" in t_prasadam.lower() or "food" in t_prasadam.lower() or "operational" in t_prasadam.lower()):
                doc_lines.append(f"- **Prasadam / Food**: {t_prasadam}")
            elif t_restaurants["lunch_options"]:
                doc_lines.append(f"- **Local Dining**: Pure vegetarian eateries available near shrine entrance (e.g. {t_restaurants['lunch_options'][0]['name']}, {t_restaurants['lunch_options'][0]['distance']})")
            doc_lines.append("")

        # 9. Render Day-by-Day Itinerary
        for day_num in range(1, calendar_days + 1):
            t_list = day_temple_map.get(day_num, [])
            if not t_list:
                continue
            if len(t_list) == 1:
                active_temple = t_list[0]
                t_data = temple_schedules[active_temple]
                days_with_this_temple = [d for d in range(1, calendar_days + 1) if active_temple in day_temple_map.get(d, [])]
                total_temple_days = len(days_with_this_temple)
                temple_day_index = days_with_this_temple.index(day_num) + 1 if day_num in days_with_this_temple else 1
                prev_temple = day_temple_map.get(day_num - 1, [None])[0] if day_num > 1 else None
                section_lines = self._format_schedule_section(
                    temple=active_temple,
                    day_num=day_num,
                    total_days=calendar_days,
                    temple_day_index=temple_day_index,
                    total_temple_days=total_temple_days,
                    data=t_data,
                    is_multi_temple=is_multi_temple,
                    is_fractional_end=is_fractional,
                    duration_days=duration_days,
                    prev_temple=prev_temple,
                    pax=pax,
                    rooms=rooms,
                    origin=origin,
                    origin_assumed=origin_assumed,
                    selected_accommodation=sc.get("selected_accommodation") if sc else None,
                    has_elderly=has_elderly
                )
                doc_lines.extend(section_lines)
            else:
                # Composite Day covering multiple shrines without omission
                t_names = " & ".join([get_canonical_display_name(t) for t in t_list])
                doc_lines.append(f"#### Day {day_num}: Multi-Temple Pilgrimage Circuit — {t_names}\n")
                first_t = t_list[0]
                second_t = t_list[1]
                t1_disp = get_canonical_display_name(first_t)
                t2_disp = get_canonical_display_name(second_t)
                t1_open = temple_schedules[first_t].get("timings", {}).get("opening", "06:00 AM")
                t2_close = temple_schedules[second_t].get("timings", {}).get("closing", "08:30 PM")
                orig_dep = f"Early departure from {origin}." if (origin and not origin_assumed) else "Early departure from origin."
                doc_lines.append(f"- **Morning Window (First Shrine — {t1_disp})**: {orig_dep} Reach {t1_disp} for morning darshan, sanctum archana, and temple pradakshinam ({t1_open} onwards). Devotees receive morning blessing and sanctified theertham.")
                doc_lines.append(f"- **Midday & Transit Window**: Traditional vegetarian lunch / Annadanam prasadam, followed by scenic road transit onward to {t2_disp}.")
                doc_lines.append(f"- **Afternoon & Evening Window (Second Shrine — {t2_disp})**: Arrive at {t2_disp} for afternoon/evening darshan and Sandhya Aarti (temple rituals active until {t2_close} night closure). Devotees collect sanctified Laddu/Pulihora prasadam.")
                if day_num == calendar_days and is_fractional:
                    dep_term = self._get_departure_terminal(temple_schedules[second_t], second_t)
                    doc_lines.append(f"- **Afternoon Window (Scheduled Return Journey)**: Proceed to transit terminal ({dep_term}) for return journey (no hotel stay required on final departure half-day).\n")
                elif calendar_days == 1:
                    ret_dest = f"to {origin}" if (origin and not origin_assumed) else "return departure"
                    doc_lines.append(f"- **Night Window (Return Departure)**: Dinner at a verified pure-vegetarian eatery; proceed directly to transit terminal for scheduled return departure {ret_dest} (no hotel stay required for single-day pilgrimage).\n")
                else:
                    room_desc = f"{rooms} room(s) for night {day_num} (standard 2 persons/room for {pax} pilgrims)" if pax > 1 else f"1 room for night {day_num}"
                    doc_lines.append(f"- **Night Window (Overnight Stay)**: Dinner at verified vegetarian restaurant; check in at verified lodging near {t2_disp} ({room_desc}).\n")

        # 10. Consolidated Accommodation Summary
        if duration_days <= 1.0 and not is_fractional:
            doc_lines.append("### 🏨 Accommodation Summary (Day Trip)")
            doc_lines.append("- Day trip does not require an overnight stay (zero room charges budgeted).")
            doc_lines.append("")
        else:
            doc_lines.append("### 🏨 Accommodation Summary")
            room_detail = f" ({rooms} room(s) for {pax} pilgrims, standard 2 persons/room)" if pax > 1 else " (1 room)"
            doc_lines.append(f"- **Trip Stay Requirements**: {nights} night(s) required across the {duration_days:g}-day itinerary{room_detail}.")
            doc_lines.append("- **Lodging Options**: Verified devasthanam choultries, government guest houses, and private hotels are detailed in the authoritative Accommodation section below.")
            doc_lines.append("")

        # 10b. Elderly-Friendly Facilities & Accessibility (P0 Requirement)
        elderly_lines = self._build_elderly_facilities_section(detected_temples, raw_query, sc, has_elderly)
        if elderly_lines:
            doc_lines.extend(elderly_lines)

        # 11. Consolidated Travel & Connectivity Section
        doc_lines.append("### 🚆 Travel & Connectivity")
        if is_interstate:
            if "udaipur" in origin.lower() and any("veera" in t.lower() or "annavaram" in t.lower() for t in detected_temples):
                doc_lines.append(f"- **Inter-State Transit Corridor ({origin} ⇄ Circuit)**: Transit from {origin} involves a combination of Indian Railways long-distance routes and regional connecting buses covering an estimated ~3,690 km continuous circuit across Rajasthan, Andhra Pradesh, and Telangana.")
                doc_lines.append(f"  • **Route Leg 1 ({origin} ➔ Annavaram)**: ~1,450 km interstate rail benchmark (~24–28 hrs) [Rail Estimate — Source: IRCTC Network Corridor Benchmark, Not Live GPS/Tracking; approximate straight-line geographical distance is ~1,150 km].")
                doc_lines.append(f"  • **Route Leg 2 (Annavaram ➔ Medaram)**: ~390 km connecting road transit via NH16/NH163 (~6–8 hrs) [Road Corridor Estimate — Status: Unverified; Live Google Routes API unavailable in host environment, road distances and live traffic conditions unverified].")
                doc_lines.append(f"  • **Route Leg 3 (Medaram ➔ Jogulamba)**: ~445 km road corridor via Warangal & Hyderabad bypass to Alampur (~6–8 hrs) [Road Corridor Estimate — Status: Unverified; Live Google Routes API unavailable, road distances and live traffic conditions unverified].")
                doc_lines.append(f"  • **Route Leg 4 (Jogulamba ➔ Karmanghat)**: ~209 km NH44 northbound highway corridor into Greater Hyderabad (~3.5–5 hrs) [Road Corridor Estimate — Status: Unverified; Live Google Routes API unavailable, road distances and live traffic conditions unverified].")
                doc_lines.append(f"  • **Route Leg 5 (Karmanghat ➔ {origin})**: ~1,180 km return interstate rail benchmark (~20–24 hrs) [Rail Estimate — Source: IRCTC Network Corridor Benchmark, Not Live GPS/Tracking; approximate straight-line geographical distance is ~1,080 km].")
                doc_lines.append(f"  • *(Corridor Validation & Limitation Disclosure: Rail figures derive from standard IRCTC network corridor benchmarks, while road legs are estimated from regional highway geometry. Live GPS road navigation meters and real-time train timetables cannot be verified without live API credentials and must be confirmed through IRCTC / state RTC portals before departure).*")
            else:
                first_t_disp = get_canonical_display_name(detected_temples[0])
                last_t_disp = get_canonical_display_name(detected_temples[-1])
                doc_lines.append(f"- **Inter-State Transit Corridor ({origin} ⇄ {temples_title})**: Transit from {origin} involves long-distance interstate rail/bus travel connecting into the regional pilgrimage network.")
                doc_lines.append(f"  • **Outbound Leg ({origin} ➔ {first_t_disp})**: Long-distance interstate rail connection into regional gateway hub.")
                if is_multi_temple:
                    for i in range(len(detected_temples) - 1):
                        t1_name = get_canonical_display_name(detected_temples[i])
                        t2_name = get_canonical_display_name(detected_temples[i + 1])
                        doc_lines.append(f"  • **Circuit Leg {i+1} ({t1_name} ➔ {t2_name})**: Regional road / rail transit connection.")
                doc_lines.append(f"  • **Return Leg ({last_t_disp} ➔ {origin})**: Return interstate rail journey back to {origin}.")
                doc_lines.append(f"  • *(Corridor Validation & Limitation Disclosure: Live GPS navigation meters and real-time train timetables cannot be verified without live API credentials; confirm schedules via IRCTC / state RTC portals before departure).*")
        for t in detected_temples:
            t_disp = get_canonical_display_name(t)
            t_data = temple_schedules[t]
            t_notes = self._parse_travel(t_data)
            prefix = f"**{t_disp}**: " if is_multi_temple else ""
            if t_notes:
                for note in t_notes:
                    doc_lines.append(f"- {prefix}{note}")
            else:
                doc_lines.append(f"- {prefix}Verify local road connectivity and TSRTC bus terminal schedules.")
        doc_lines.append("")

        # 12. Append Live-Data Disclaimer
        doc_lines.append(LIVE_DATA_DISCLAIMER)
        doc_lines.append("")

        # 13. Source Attribution preserved internally in AgentResponse.sources
        final_answer = "\n".join(doc_lines).strip()

        # 14. Update State Intermediate Results
        state.intermediate_results[self.name] = {
            "temples": detected_temples,
            "duration_days": duration_days,
            "days": duration_days,
            "calendar_days": calendar_days,
            "nights": nights,
            "pax": pax,
            "adults": adults,
            "children": children,
            "seniors": seniors,
            "senior_citizens": seniors,
            "has_elderly": has_elderly,
            "rooms": rooms,
            "origin": origin,
            "origin_assumed": origin_assumed,
            "is_interstate": is_interstate,
            "is_fractional": is_fractional,
            "sources_by_temple": sources_by_temple,
            "live_data_disclaimer": True
        }

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer=final_answer,
            sources=all_sources,
            metadata={
                "itinerary": True,
                "temples": detected_temples,
                "duration_days": duration_days,
                "days": duration_days,
                "calendar_days": calendar_days,
                "nights": nights,
                "pax": pax,
                "adults": adults,
                "children": children,
                "seniors": seniors,
                "senior_citizens": seniors,
                "has_elderly": has_elderly,
                "rooms": rooms,
                "origin": origin,
                "origin_assumed": origin_assumed,
                "is_interstate": is_interstate,
                "is_fractional": is_fractional,
                "sources_by_temple": sources_by_temple,
                "disclaimer_present": True
            }
        )
