"""
PilgrimAI - Canonical Trip Constraints Engine
Extracts request-scoped canonical constraints and prevents cross-request parameter leakage.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
import re
import math
import logging

from app.agents.state import AgentResponse

logger = logging.getLogger(__name__)

WORD_TO_NUM: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "single": 1, "double": 2,
    "a": 1, "an": 1
}

RESERVED_NON_TRAVELLER_WORDS: Set[str] = {
    "day", "days", "night", "nights", "hour", "hours", "hr", "hrs",
    "week", "weeks", "month", "months", "year", "years",
    "rupee", "rupees", "rs", "inr", "k", "thousand", "lakh", "lakhs", "lac", "lacs",
    "temple", "temples", "mandir", "mandirs", "devasthanam", "shrine", "shrines",
    "pillar", "pillars", "budget", "limit", "cost", "costs", "price", "prices",
    "km", "kms", "kilometer", "kilometers", "meter", "meters", "m",
    "am", "pm", "star", "stars", "room", "rooms", "bed", "beds",
    "percent", "percentage", "discount", "tier",
    "trip", "trips", "tour", "tours", "visit", "visits", "itinerary", "plan", "plans",
    "st", "nd", "rd", "th"
}

CANONICAL_ADULT_WORDS: Set[str] = {
    "adult", "adults", "adlt", "adlts", "adulta", "adultas", "adlut", "adluts"
}

CANONICAL_CHILD_WORDS: Set[str] = {
    "child", "children", "childs", "chld", "chlds", "kid", "kids", "kiddos",
    "childern", "chilren", "chldren", "kds"
}

CANONICAL_GENERAL_WORDS: Set[str] = {
    "people", "peple", "poeple", "person", "persons", "prson", "prsons",
    "pilgrim", "pilgrims", "pligrim", "pligrims",
    "devotee", "devotees", "member", "members", "membrs",
    "passenger", "passengers",
    "traveller", "travellers", "traveler", "travelers", "travellr", "travlr"
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def classify_traveller_word(word: str) -> Optional[str]:
    """
    Classifies a candidate word into 'adult', 'child', or 'general' traveller category,
    using bounded fuzzy matching tolerant to common typos.
    Returns None if the word is non-traveller related or unrecognized.
    """
    w = word.strip().lower()
    if not w or len(w) < 3:
        return None
    if w in RESERVED_NON_TRAVELLER_WORDS:
        return None

    # 1. Exact match check
    if w in CANONICAL_ADULT_WORDS:
        return "adult"
    if w in CANONICAL_CHILD_WORDS:
        return "child"
    if w in CANONICAL_GENERAL_WORDS:
        return "general"

    # 2. Bounded fuzzy match check:
    # First letter must match to prevent cross-word collisions (e.g. 'temple' vs 'people').
    # Edit distance threshold: <= 1 for words <= 4 chars, <= 2 for words >= 5 chars.
    for target in CANONICAL_ADULT_WORDS:
        if w[0] == target[0] and abs(len(w) - len(target)) <= 2:
            max_dist = 1 if len(w) <= 4 else 2
            if levenshtein_distance(w, target) <= max_dist:
                return "adult"

    for target in CANONICAL_CHILD_WORDS:
        if w[0] == target[0] and abs(len(w) - len(target)) <= 2:
            max_dist = 1 if len(w) <= 4 else 2
            if levenshtein_distance(w, target) <= max_dist:
                return "child"

    for target in CANONICAL_GENERAL_WORDS:
        if w[0] == target[0] and abs(len(w) - len(target)) <= 2:
            max_dist = 1 if len(w) <= 4 else 2
            if levenshtein_distance(w, target) <= max_dist:
                return "general"

    return None


def normalize_duration_expression(text: str) -> str:
    """
    Normalizes missing-space and formatting variations in duration expressions:
    - 'a3 day trip' -> 'a 3 day trip'
    - 'a3-day trip' -> 'a 3-day trip'
    - 'a 3day trip' -> 'a 3 day trip'
    - 'a4.5 day trip' -> 'a 4.5 day trip'
    - 'a4.5day trip' -> 'a 4.5 day trip'
    Only applies where context explicitly indicates a duration keyword (days/nights/hours/weeks).
    """
    if not text:
        return text
    # 1. 'a3 day', 'a3-day', 'a4.5 day' -> 'a 3 day', 'a 3-day', 'a 4.5 day'
    s = re.sub(r"\ba(\d+(?:\.\d+)?)\s*[- ]*(days?|nights?|hours?|weeks?)\b", r"a \1 \2", text, flags=re.IGNORECASE)
    # 2. '3day', '3days', '4.5day' -> '3 day', '3 days', '4.5 day'
    s = re.sub(r"\b(\d+(?:\.\d+)?)(days?|nights?)\b", r"\1 \2", s, flags=re.IGNORECASE)
    return s


def extract_traveller_party_size(query: str) -> Tuple[Optional[int], Optional[int], Optional[int], int, bool]:
    """
    Deterministically extracts party size with typo-tolerant fuzzy matching.
    Returns: (adults, children, seniors, total_pax, pax_assumed)
    
    Examples:
    - '2 adults and 2 children and 1 senior citizen' -> (2, 2, 1, 5, False)
    - '2 adults and 2 children and 1 elderly person' -> (2, 2, 1, 5, False)
    - '2 adulta and 1 child' -> (2, 1, None, 3, False)
    - '2 adlts and 1 chld' -> (2, 1, None, 3, False)
    - '3 adult and 2 children' -> (3, 2, None, 5, False)
    - '4 people' -> (None, None, None, 4, False)
    - '2 persons' -> (None, None, None, 2, False)
    - '2 adults' -> (2, None, None, 2, False)
    - '2 adults and 1 kid' -> (2, 1, None, 3, False)
    """
    q = query.lower()

    found_seniors: Optional[int] = None

    # Matches senior citizen / elderly descriptions:
    senior_pattern = re.compile(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|single|double|a|an)\s+"
        r"(?:seni?o?rs?\s+citiz?e?ns?|elderly\s*(?:persons?|peple|people)?|seni?o?rs?|older\s+(?:parents?|adults?)|elders?)\b",
        re.IGNORECASE
    )
    for m in senior_pattern.finditer(q):
        raw_num = m.group(1).lower()
        count = WORD_TO_NUM.get(raw_num, int(raw_num) if raw_num.isdigit() else 0)
        if count > 0:
            found_seniors = (found_seniors or 0) + count

    # Mask out the matched senior phrases in a working copy to prevent 'adults' in 'older adults' from being counted twice
    q_work = senior_pattern.sub(" ", q)

    found_adults: Optional[int] = None
    found_children: Optional[int] = None
    found_general: Optional[int] = None

    # Matches numbers/num_words followed optionally by space and a word: e.g. "2 adulta", "1 child", "4people"
    pattern = re.compile(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|single|double|a|an)\s*([a-zA-Z]+)\b"
    )
    for m in pattern.finditer(q_work):
        raw_num, raw_word = m.group(1), m.group(2)
        count = WORD_TO_NUM.get(raw_num, int(raw_num) if raw_num.isdigit() else 0)
        if count <= 0:
            continue
        category = classify_traveller_word(raw_word)
        if category == "adult":
            found_adults = (found_adults or 0) + count
        elif category == "child":
            found_children = (found_children or 0) + count
        elif category == "general":
            found_general = (found_general or 0) + count

    if found_adults is not None or found_children is not None or found_seniors is not None:
        total = (found_adults or 0) + (found_children or 0) + (found_seniors or 0)
        return found_adults, found_children, found_seniors, max(1, total), False

    if found_general is not None:
        return None, None, None, max(1, found_general), False

    # Check idiomatic phrases: 'family of 4', 'group of 3', 'couple', etc.
    fam_m = re.search(r"\b(?:family\s+of|group\s+of)\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", q)
    if fam_m:
        raw_val = fam_m.group(1).lower()
        val = WORD_TO_NUM.get(raw_val, int(raw_val) if raw_val.isdigit() else 1)
        return None, None, None, max(1, val), False

    if re.search(r"\b(?:couple|two\s+of\s+us|(?:with|for)\s+(?:2|two)(?!\s*(?:days?|nights?|hours?|weeks?|am|pm|k\b|00\b)))\b", q):
        return None, None, None, 2, False
    if re.search(r"\b(?:three\s+of\s+us|(?:with|for)\s+(?:3|three)(?!\s*(?:days?|nights?|hours?|weeks?|am|pm|k\b|00\b)))\b", q):
        return None, None, None, 3, False
    if re.search(r"\b(?:four\s+of\s+us|(?:with|for)\s+(?:4|four)(?!\s*(?:days?|nights?|hours?|weeks?|am|pm|k\b|00\b)))\b", q):
        return None, None, None, 4, False
    if re.search(r"\b(?:five\s+of\s+us|(?:with|for)\s+(?:5|five)(?!\s*(?:days?|nights?|hours?|weeks?|am|pm|k\b|00\b)))\b", q):
        return None, None, None, 5, False
    if re.search(r"\b(?:alone|solo|myself)\b", q):
        return None, None, None, 1, False

    return None, None, None, 1, True



@dataclass
class CanonicalTripConstraints:
    raw_query: str
    budget_limit: Optional[float] = None
    budget_currency: str = "INR"
    adults: Optional[int] = None
    children: Optional[int] = None
    seniors: Optional[int] = None
    pax: int = 1
    pax_assumed: bool = True
    duration_days: float = 1.0
    fractional_days: bool = False
    days_assumed: bool = True
    nights: int = 0
    nights_assumed: bool = True
    is_day_trip: bool = True
    destinations: List[str] = field(default_factory=list)
    destination_order: List[str] = field(default_factory=list)
    origin: Optional[str] = "Hyderabad"
    origin_assumed: bool = True
    travel_mode: str = ""
    rooms: int = 0
    rooms_assumed: bool = True
    is_followup: bool = False
    inherited_fields: List[str] = field(default_factory=list)
    is_budget_request: bool = False
    accom_requested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "budget_limit": self.budget_limit,
            "budget_currency": self.budget_currency,
            "adults": self.adults,
            "children": self.children,
            "seniors": self.seniors,
            "senior_citizens": self.seniors,
            "pax": self.pax,
            "pax_assumed": self.pax_assumed,
            "duration_days": self.duration_days,
            "days": self.duration_days,
            "fractional_days": self.fractional_days,
            "days_assumed": self.days_assumed,
            "nights": self.nights,
            "nights_assumed": self.nights_assumed,
            "is_day_trip": self.is_day_trip,
            "destinations": list(self.destinations),
            "destination_order": list(self.destination_order),
            "temples": list(self.destinations),
            "origin": self.origin,
            "origin_assumed": self.origin_assumed,
            "transport_mode": self.travel_mode,
            "rooms": self.rooms,
            "rooms_assumed": self.rooms_assumed,
            "accommodation_required": self.accom_requested or (self.nights > 0 and not self.is_day_trip),
            "is_followup": self.is_followup,
            "inherited_fields": list(self.inherited_fields),
            "is_budget_request": self.is_budget_request,
            "accom_requested": self.accom_requested
        }


def parse_budget_value(text: str) -> Optional[float]:
    """Deterministically parses numeric, k-suffix, or word budget values."""
    if not text:
        return None
    s = text.strip().lower().replace(",", "")
    
    # 1. Check for 'k' or 'K' suffix e.g. "2k", "2.5k", "3.8k", "10k"
    m_k = re.search(r"^(\d+(?:\.\d+)?)\s*k$", s)
    if m_k:
        try:
            return float(m_k.group(1)) * 1000.0
        except ValueError:
            pass

    # 2. Check for "thousand" e.g. "2 thousand", "5 thousand" (protect against "thousand pillar")
    m_th = re.search(r"^(\d+(?:\.\d+)?)\s*thousand$", s)
    if m_th:
        try:
            return float(m_th.group(1)) * 1000.0
        except ValueError:
            pass

    # 3. Check for "lakh" or "lac"
    m_lakh = re.search(r"^(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?)$", s)
    if m_lakh:
        try:
            return float(m_lakh.group(1)) * 100000.0
        except ValueError:
            pass

    # 4. Standard float / integer
    m_num = re.search(r"^(\d+(?:\.\d+)?)$", s)
    if m_num:
        try:
            return float(m_num.group(1))
        except ValueError:
            pass

    # 5. Words
    word_map = {
        "one thousand": 1000.0,
        "two thousand": 2000.0,
        "three thousand": 3000.0,
        "four thousand": 4000.0,
        "five thousand": 5000.0,
        "six thousand": 6000.0,
        "seven thousand": 7000.0,
        "eight thousand": 8000.0,
        "nine thousand": 9000.0,
        "ten thousand": 10000.0,
        "two k": 2000.0,
        "three k": 3000.0,
        "five k": 5000.0,
        "eight k": 8000.0,
        "three thousand eight hundred": 3800.0,
        "two thousand five hundred": 2500.0,
        "four thousand seven hundred fifty": 4750.0
    }
    for w_phrase, w_val in word_map.items():
        if w_phrase in s:
            return w_val

    return None


def is_trip_followup(
    query: str,
    session_trip_context: Optional[Dict[str, Any]],
    detected_temples: List[str]
) -> bool:
    """
    Determines whether the query is demonstrably a follow-up to the existing trip in session memory,
    rather than an independent new trip or unrelated query.
    """
    if not session_trip_context or not session_trip_context.get("destinations"):
        return False

    existing_destinations = session_trip_context.get("destinations", [])

    # If the current query explicitly mentions any temple that is NOT in existing_destinations,
    # it is unequivocally a new destination / new trip!
    if detected_temples:
        for t in detected_temples:
            if t not in existing_destinations:
                return False

    q_lower = query.lower()

    # If the user explicitly asks to "plan a X day trip to..." with temples,
    # this is an explicit new trip definition even if at the same temple.
    is_explicit_new_plan = bool(re.search(r"\b(plan|itinerary|tour)\s+(?:a\s+)?(?:\d+|\w+)\s*[- ]*days?\b", q_lower))
    if is_explicit_new_plan and detected_temples:
        return False

    # Check for genuine follow-up indicators
    budget_followup = bool(re.search(r"\b(?:change|update|set|make|increase|decrease|reduce|my)?\s*budget\s*(?:to|is|of|as|at)?\s*([0-9.,]+[kK]?|\w+)\b", q_lower))
    _, _, _, _, p_assumed = extract_traveller_party_size(q_lower)
    pax_followup = (not p_assumed) and bool(re.search(r"\b(?:what\s+about|how\s+about|for|make\s+it|update\s+to|change\s+to)\b", q_lower))
    duration_followup = bool(re.search(r"\b(?:make\s+it|change\s+to|what\s+about)\s+(\d+(?:\.5)?)\s*days?\b", q_lower))
    transport_followup = bool(re.search(r"\b(?:what\s+about\s+by|how\s+about\s+by|by)\s+(?:train|bus|car|cab|flight)\b", q_lower))
    anaphoric = bool(re.search(r"\b(?:this|that|the)\s+(?:trip|tour|itinerary|plan)\b|\bthere\b|\bit\b", q_lower))
    topic_followup = bool(re.search(r"\b(?:what\s+about|how\s+about|and|also)\s+(?:the\s+)?(?:sevas?|timings?|rooms?|accommodation|budget|cost|price|darshan)\b", q_lower))

    return budget_followup or pax_followup or duration_followup or transport_followup or anaphoric or topic_followup


def extract_canonical_constraints(
    query: str,
    session_trip_context: Optional[Dict[str, Any]] = None,
    detected_temples: Optional[List[str]] = None
) -> CanonicalTripConstraints:
    """
    Creates a fresh, request-scoped CanonicalTripConstraints object for the current request.
    Strictly prevents stale state from previous requests from leaking into independent new requests.
    """
    q_norm = normalize_duration_expression(query.lower().strip())
    temples = list(detected_temples) if detected_temples is not None else []

    is_followup = is_trip_followup(query, session_trip_context, temples)
    inherited_fields: List[str] = []

    # 1. Budget extraction
    budget_limit: Optional[float] = None
    is_budget_request = False

    # Match explicit budget patterns avoiding "thousand pillar"
    b_match = re.search(
        r"\b(?:budget\s+(?:to\s+|of\s+|is\s+|limit\s+(?:is\s+|of\s+|to\s+)?)?|under\s+(?:a\s+budget\s+of\s+)?|within\s+(?:a\s+budget\s+of\s+)?|below\s+(?:a\s+budget\s+of\s+)?|have\s+)(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)*(?:\.\d+)?\s*[kK]?|\d+\s*thousand(?![\w\s]*pillar))\b",
        q_norm
    )
    if b_match:
        budget_limit = parse_budget_value(b_match.group(1))
        if budget_limit is not None:
            is_budget_request = True
    else:
        b_match2 = re.search(
            r"\b(?:rs\.?|inr|₹)\s*(\d+(?:,\d+)*(?:\.\d+)?\s*[kK]?|\d+\s*thousand(?![\w\s]*pillar))\s*(?:budget|limit|for)\b",
            q_norm
        )
        if b_match2:
            budget_limit = parse_budget_value(b_match2.group(1))
            if budget_limit is not None:
                is_budget_request = True
        else:
            b_match3 = re.search(
                r"\b(?:with|under|within|in)\s+(?:a\s+)?(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)*(?:\.\d+)?\s*[kK]?|\d+\s*thousand(?![\w\s]*pillar))\s*(?:budget|limit|rs|inr|rupees?)?\b",
                q_norm
            )
            if b_match3:
                budget_limit = parse_budget_value(b_match3.group(1))
                if budget_limit is not None:
                    is_budget_request = True
            else:
                b_match4 = re.search(
                    r"\b(\d+(?:,\d+)*(?:\.\d+)?\s*[kK]?|\d+\s*thousand(?![\w\s]*pillar))\s*(?:budget|limit|rupees?)\b",
                    q_norm
                )
                if b_match4:
                    budget_limit = parse_budget_value(b_match4.group(1))
                    if budget_limit is not None:
                        is_budget_request = True

    # If not explicitly supplied, inherit ONLY if demonstrably a follow-up
    if budget_limit is None:
        if is_followup and session_trip_context and "budget_limit" in session_trip_context:
            budget_limit = session_trip_context["budget_limit"]
            if budget_limit is not None:
                inherited_fields.append("budget_limit")
                is_budget_request = True

    # 2. Party Size: Adults, Children, Seniors and Pax
    adults, children, seniors, pax, pax_assumed = extract_traveller_party_size(q_norm)

    # Inherit pax only if demonstrably a follow-up
    if pax_assumed:
        if is_followup and session_trip_context and "pax" in session_trip_context:
            pax = int(session_trip_context["pax"])
            pax_assumed = False
            adults = session_trip_context.get("adults")
            children = session_trip_context.get("children")
            seniors = session_trip_context.get("seniors")
            inherited_fields.append("pax")

    # 3. Duration: Days & Nights
    days = 1.0
    days_assumed = True
    nights = 0
    nights_assumed = True
    fractional_days = False
    is_day_trip = True

    half_day_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven)\s+(?:and\s+)?(?:a\s+)?half\s+days?\b", q_norm)
    dec_day_match = re.search(r"\b(\d+)\.5\s*[- ]*days?\b", q_norm)
    night_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[- ]*(?:nights?|overnight)\b", q_norm)
    day_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[- ]*(?:days?)\b", q_norm)

    if half_day_match:
        raw_hd = half_day_match.group(1).lower()
        base_d = WORD_TO_NUM.get(raw_hd, int(raw_hd) if raw_hd.isdigit() else 1)
        days = float(base_d) + 0.5
        nights = base_d
        days_assumed = False
        nights_assumed = False
        fractional_days = True
        is_day_trip = False
    elif dec_day_match:
        base_d = int(dec_day_match.group(1))
        days = float(base_d) + 0.5
        nights = base_d
        days_assumed = False
        nights_assumed = False
        fractional_days = True
        is_day_trip = False
    elif night_match:
        raw_n = night_match.group(1).lower()
        nights = WORD_TO_NUM.get(raw_n, int(raw_n) if raw_n.isdigit() else 1)
        nights = max(1, nights)
        days = float(nights + 1)
        days_assumed = False
        nights_assumed = False
        is_day_trip = False
    elif day_match:
        raw_d = day_match.group(1).lower()
        base_d = WORD_TO_NUM.get(raw_d, int(raw_d) if raw_d.isdigit() else 1)
        days = float(max(1, base_d))
        days_assumed = False
        if days == 1.0:
            nights = 0
            nights_assumed = False
            is_day_trip = True
        else:
            is_day_trip = False
            if re.search(r'\b(?:accommodation|stay|room|rooms|hotel|lodge|cottage)\b.*?\b\d+\s*days?\b', q_norm) or re.search(r'\b\d+\s*days?\b.*?\b(?:accommodation|stay|room|rooms|hotel|lodge|cottage)\b', q_norm):
                nights = int(days)
            else:
                nights = max(1, int(days - 1))
            nights_assumed = False
    elif "weekend" in q_norm:
        days = 2.0
        nights = 1
        days_assumed = False
        nights_assumed = False
        is_day_trip = False
    elif any(re.search(pat, q_norm) for pat in [
        r"(?<!\d\s)(?<!\d)(?<!two\s)(?<!three\s)(?<!four\s)(?<!five\s)\bday\s*(?:trip|visit)\b",
        r"\bsingle\s+day\b",
        r"\bsame\s+day\b",
        r"\b1\s*day\b",
        r"\bone\s*day\b"
    ]):
        days = 1.0
        nights = 0
        days_assumed = False
        nights_assumed = False
        is_day_trip = True

    # Inherit duration only if demonstrably a follow-up
    if days_assumed:
        if is_followup and session_trip_context and "duration_days" in session_trip_context:
            days = float(session_trip_context["duration_days"])
            days_assumed = False
            fractional_days = (days % 1 != 0)
            nights = int(session_trip_context.get("nights", math.floor(days) if fractional_days else max(0, int(days - 1))))
            nights_assumed = False
            is_day_trip = bool(session_trip_context.get("is_day_trip", days <= 1.0))
            inherited_fields.append("duration_days")

    # 4. Destinations and Order
    destinations: List[str] = list(temples)
    if not destinations:
        if is_followup and session_trip_context and session_trip_context.get("destinations"):
            destinations = list(session_trip_context["destinations"])
            inherited_fields.append("destinations")

    destination_order = list(destinations)

    # 5. Origin
    origin = "Hyderabad"
    origin_assumed = True
    orig_match = re.search(r"\bfrom\s+([A-Za-z\s]+?)(?:\s+(?:to|for|with|under|within|on|by|in)\b|[,;.]|$)", q_norm)
    if orig_match:
        cand = orig_match.group(1).strip()
        cand_clean = " ".join([w for w in cand.split() if w not in ["a", "the", "my", "our", "this"]])
        if cand_clean:
            origin = cand_clean.title()
            origin_assumed = False

    if origin_assumed:
        if is_followup and session_trip_context and "origin" in session_trip_context and not session_trip_context.get("origin_assumed", True):
            origin = session_trip_context["origin"]
            origin_assumed = False
            inherited_fields.append("origin")

    # 6. Travel Mode
    travel_mode = ""
    mode_patterns = [
        (r"\b(?:by\s+)?(train|railway|rail)\b", "train"),
        (r"\b(?:by\s+)?(bus|rtc|tsrtc|apsrtc)\b", "bus"),
        (r"\b(?:by\s+)?(car|cab|taxi|drive|driving)\b", "cab"),
        (r"\b(?:by\s+)?(flight|air|plane)\b", "flight"),
        (r"\b(?:by\s+)?(bike|motorcycle|two-wheeler)\b", "two-wheeler"),
    ]
    for pat, mode in mode_patterns:
        if re.search(pat, q_norm):
            travel_mode = mode
            break

    if not travel_mode:
        if is_followup and session_trip_context and session_trip_context.get("transport_mode"):
            travel_mode = session_trip_context["transport_mode"]
            inherited_fields.append("transport_mode")

    # 7. Rooms
    room_match = re.search(r"\b(\d+|one|two|three|four)\s*rooms?\b", q_norm)
    if room_match:
        raw_r = room_match.group(1).lower()
        rooms = WORD_TO_NUM.get(raw_r, int(raw_r) if raw_r.isdigit() else 1)
        rooms = max(1, rooms)
        rooms_assumed = False
    else:
        rooms_assumed = True
        if is_day_trip or nights == 0:
            rooms = 0
        else:
            if adults is not None and children is not None and adults <= 2 and children <= 1:
                rooms = 1
            else:
                rooms = max(1, math.ceil(pax / 2.0))

    accom_requested = bool(re.search(r"\b(hotel|hotels|stay|lodging|lodge|rooms?|cottage|resort)\b", q_norm))

    return CanonicalTripConstraints(
        raw_query=query,
        budget_limit=budget_limit,
        budget_currency="INR",
        adults=adults,
        children=children,
        seniors=seniors,
        pax=pax,
        pax_assumed=pax_assumed,
        duration_days=days,
        fractional_days=fractional_days,
        days_assumed=days_assumed,
        nights=nights,
        nights_assumed=nights_assumed,
        is_day_trip=is_day_trip,
        destinations=destinations,
        destination_order=destination_order,
        origin=origin,
        origin_assumed=origin_assumed,
        travel_mode=travel_mode,
        rooms=rooms,
        rooms_assumed=rooms_assumed,
        is_followup=is_followup,
        inherited_fields=inherited_fields,
        is_budget_request=is_budget_request,
        accom_requested=accom_requested
    )


def validate_and_synchronize_response(
    canonical: CanonicalTripConstraints,
    response: AgentResponse
) -> AgentResponse:
    """
    Validates and ensures the final aggregated AgentResponse conforms strictly
    to the current request's canonical parameters.
    """
    if not response or not isinstance(response, AgentResponse):
        return response

    if response.metadata is None:
        response.metadata = {}

    # Attach canonical constraints dict to metadata
    response.metadata["structured_constraints"] = canonical.to_dict()

    # Synchronize budget limit in metadata
    if canonical.budget_limit is not None:
        response.metadata["budget_limit"] = canonical.budget_limit
        if "parameters" in response.metadata and isinstance(response.metadata["parameters"], dict):
            response.metadata["parameters"]["budget_limit"] = canonical.budget_limit

    # Synchronize pax & adult/child/senior counts in metadata
    response.metadata["pax"] = canonical.pax
    response.metadata["pax_assumed"] = canonical.pax_assumed
    if canonical.adults is not None:
        response.metadata["adults"] = canonical.adults
    if canonical.children is not None:
        response.metadata["children"] = canonical.children
    if canonical.seniors is not None:
        response.metadata["seniors"] = canonical.seniors
        response.metadata["senior_citizens"] = canonical.seniors

    # Synchronize duration and destinations
    response.metadata["duration_days"] = canonical.duration_days
    response.metadata["days"] = canonical.duration_days
    response.metadata["nights"] = canonical.nights
    response.metadata["destinations"] = canonical.destinations

    return response
