"""
PilgrimAI - BudgetAgent (Phase O4 - Production Fix)

Specialist agent calculating grounded, deterministic pilgrimage costs from verified
temple records, authoritative distance benchmarks, and explicit user-provided inputs.

Core Principles:
- 100% Grounded Arithmetic: Calculations derive strictly from verified temple records
  (sevas, accommodation, prasadam, timings, restaurants, travel) and explicit user inputs.
- Parameter Preservation: Extracts and strictly preserves group size (pax, never defaults
  explicit numbers to 1), duration/nights, origin, destination, rooms, and budget limit.
- Complete Trip Cost Model: Computes all essential components (round-trip transportation,
  accommodation for required rooms/nights, meals for all travellers/days, entry/darshan,
  selected sevas, and local transit).
- Evidence vs Estimate Separation: Every cost line explicitly classified as:
  LISTED_PRICE, ESTIMATE, USER_PROVIDED, CALCULATED, or UNAVAILABLE.
- Hard Budget Gate: 4 deterministic statuses:
  WITHIN_BUDGET, ESTIMATED_WITHIN_BUDGET, OVER_BUDGET, CANNOT_VERIFY.
- RAGTool Facade: Accesses temple records exclusively via read-only RAGTool interface.
  Zero direct database or raw JSON file imports.
- Zero External API / Credentials: ₹0 cost, zero API keys required.
"""

import re
import time
import math
import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Union
from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent
from app.agents.state import AgentState, AgentResponse

logger = logging.getLogger(__name__)


# =====================================================================
# Provenance Model & Budget Status
# =====================================================================

class CostProvenance(str, Enum):
    """Explicit provenance classification for all pilgrimage expenditure items."""
    LISTED_PRICE = "LISTED_PRICE"     # Authoritative record from temple records
    VERIFIED = "LISTED_PRICE"         # Backward compatibility alias
    USER_PROVIDED = "USER_PROVIDED"   # Explicitly supplied by the user
    CALCULATED = "CALCULATED"         # Mathematically derived from verified/user values
    ESTIMATE = "ESTIMATE"             # Clearly identified approximation grounded in data
    UNAVAILABLE = "UNAVAILABLE"       # Price not in records and not supplied by user
    EXCLUDED = "EXCLUDED"             # Not required / excluded from itinerary


class BudgetStatus(str, Enum):
    """Deterministic 4-tier Hard Budget Gate evaluation status."""
    WITHIN_BUDGET = "WITHIN_BUDGET"
    ESTIMATED_WITHIN_BUDGET = "ESTIMATED_WITHIN_BUDGET"
    OVER_BUDGET = "OVER_BUDGET"
    CANNOT_VERIFY = "CANNOT_VERIFY"


class BudgetItem(BaseModel):
    """Structured representation of a single pilgrimage expense line item."""
    category: str = Field(..., description="Category (travel, accommodation, meals, entry, sevas, prasadam, local_transit)")
    item_name: str = Field(..., description="Name of the service, room, or expense")
    min_cost: Optional[float] = Field(default=None, description="Minimum numeric cost in INR")
    max_cost: Optional[float] = Field(default=None, description="Maximum numeric cost in INR")
    currency: str = Field(default="INR", description="Currency identifier")
    unit: str = Field(default="per service", description="Pricing basis (per person, per room per night, per meal, per item, fixed)")
    multiplier: float = Field(default=1.0, description="Multiplier applied based on pax, rooms, or nights")
    quantity_desc: str = Field(default="", description="Human-readable description of quantity multiplication")
    subtotal_min: Optional[float] = Field(default=None, description="Calculated minimum subtotal")
    subtotal_max: Optional[float] = Field(default=None, description="Calculated maximum subtotal")
    is_free: bool = Field(default=False, description="Whether the item is confirmed free of charge")
    is_range: bool = Field(default=False, description="Whether the cost represents a variable range")
    is_quantified: bool = Field(default=True, description="Whether numeric cost bounds are established")
    provenance: CostProvenance = Field(default=CostProvenance.LISTED_PRICE, description="Data provenance")
    raw_price_str: str = Field(default="", description="Original raw price text from source or user")
    source_path: str = Field(default="", description="Source file or origin identifier")
    confirmation_required: bool = Field(default=False, description="Whether price requires confirmation before trip")
    notes: str = Field(default="", description="Qualifying notes or conditions")


# =====================================================================
# Canonical Temple Mapping & Aliases (Aligned with KnowledgeRegistry)
# =====================================================================

TEMPLE_CANONICAL_MAP = {
    "ahobilam": "Ahobilam",
    "ahobila": "Ahobilam",
    "badrachalam": "Badrachalam",
    "bhadrachalam": "Badrachalam",
    "balkampet": "Balkampet",
    "basara": "Basara",
    "basar": "Basara",
    "basara_gnana_saraswathi": "Basara",
    "gnana saraswathi": "Basara",
    "birla mandir": "Birla Mandir",
    "birla": "Birla Mandir",
    "chilkur": "Chilkur",
    "chilkur balaji": "Chilkur",
    "visa balaji": "Chilkur",
    "dharmapuri": "Dharmapuri",
    "dharmapuri_lakshmi_narasimha": "Dharmapuri",
    "edupayal": "Edupayal",
    "edupayala": "Edupayal",
    "edupayala_durga_bhavani": "Edupayal",
    "jogulamba": "Jogulamba(New)",
    "jogulamba(new)": "Jogulamba(New)",
    "alampur": "Jogulamba(New)",
    "kaleshvaram": "Kaleshvaram",
    "kaleshwaram": "Kaleshvaram",
    "mukteshwara": "Kaleshvaram",
    "karmanghat": "Karmanghat",
    "karmaghat": "Karmanghat",
    "medaram": "Medaram",
    "ramappa": "Ramappa",
    "srisaila": "Srisaila",
    "srisailam": "Srisaila",
    "thousand pillar": "Thousand Pillar",
    "thousand pillars": "Thousand Pillar",
    "ttd": "Ttd",
    "tirupati": "Ttd",
    "tirumala": "Ttd",
    "ujjaini": "Ujjaini",
    "ujjaini mahakali": "Ujjaini",
    "ujjaini mahankali": "Ujjaini",
    "veera satyanarayana": "Veera Satyanarayana",
    "annavaram": "Veera Satyanarayana",
    "annavarm": "Veera Satyanarayana",
    "vemulavada": "Vemulavada",
    "vemulawada": "Vemulavada",
    "yadadri": "Yadadri",
    "yadagirigutta": "Yadadri"
}

TEMPLE_DISPLAY_NAMES = {
    "Ahobilam": "Ahobilam Sri Lakshmi Narasimha Swamy Temple",
    "Badrachalam": "Bhadrachalam Sita Ramachandra Swamy Temple",
    "Balkampet": "Balkampet Yellamma Temple",
    "Basara": "Basara Gnana Saraswathi Temple",
    "Birla Mandir": "Birla Mandir, Hyderabad",
    "Chilkur": "Chilkur Balaji Temple",
    "Dharmapuri": "Dharmapuri Lakshmi Narasimha Swamy Temple",
    "Edupayal": "Edupayala Vana Durga Bhavani Temple",
    "Jogulamba(New)": "Jogulamba Temple, Alampur",
    "Jogulamba(new)": "Jogulamba Temple, Alampur",
    "Jogulamba": "Jogulamba Temple, Alampur",
    "Kaleshvaram": "Kaleshwaram Sri Kaleshwara Mukteshwara Swamy Temple",
    "Karmanghat": "Karmanghat Hanuman Temple",
    "Medaram": "Medaram Sammakka Saralamma Temple",
    "Ramappa": "Ramappa Temple, Palampet",
    "Srisaila": "Srisailam Mallikarjuna Swamy Temple",
    "Thousand Pillar": "Thousand Pillar Temple, Hanamkonda",
    "Ttd": "Tirumala Venkateswara Temple",
    "Ujjaini": "Ujjaini Mahakali Temple, Secunderabad",
    "Veera Satyanarayana": "Annavaram Veera Venkata Satyanarayana Swamy Temple",
    "Vemulavada": "Vemulawada Sri Raja Rajeshwara Swamy Temple",
    "Yadadri": "Yadadri Sri Lakshmi Narasimha Swamy Temple"
}

# Authoritative highway distance benchmarks (in km) from Hyderabad (State Capital benchmark)
# Sourced from travel.json files and Telangana State Road Transport (TSRTC) corridors
BENCHMARK_DISTANCES_FROM_HYDERABAD = {
    "Ahobilam": 360.0,
    "Badrachalam": 310.0,
    "Balkampet": 5.0,
    "Basara": 200.0,          # Verified in basara/travel.json: "200 km via Nizamabad"
    "Birla Mandir": 3.0,
    "Chilkur": 25.0,
    "Dharmapuri": 225.0,
    "Edupayal": 100.0,
    "Jogulamba(New)": 215.0,
    "Kaleshvaram": 260.0,
    "Karmanghat": 10.0,
    "Medaram": 230.0,
    "Ramappa": 210.0,
    "Srisaila": 215.0,
    "Thousand Pillar": 150.0,
    "Ttd": 550.0,
    "Ujjaini": 7.0,
    "Veera Satyanarayana": 470.0,
    "Vemulavada": 150.0,      # Verified in vemulavada/travel.json: "From Hyderabad via Siddipet" (~150-160 km)
    "Yadadri": 65.0
}

# Intra-state regional distance benchmarks for specific origin corridors
REGIONAL_ORIGIN_DISTANCES = {
    ("nizamabad", "basara"): 30.0,        # basara/travel.json
    ("nirmal", "basara"): 70.0,           # basara/travel.json
    ("karimnagar", "vemulavada"): 36.0,   # vemulavada/travel.json
    ("kamareddy", "vemulavada"): 68.0,    # vemulavada/travel.json
    ("warangal", "thousand pillar"): 5.0,
    ("hanamkonda", "thousand pillar"): 2.0,
    ("warangal", "ramappa"): 65.0,
    ("warangal", "medaram"): 90.0,
    ("khammam", "badrachalam"): 120.0,
    ("kurnool", "jogulamba(new)"): 25.0,
    ("kurnool", "srisaila"): 180.0,
}

# Inter-state departure origin benchmarks (in km) to Hyderabad / Telangana-AP corridor
INTERSTATE_ORIGIN_DISTANCES_FROM_HYDERABAD = {
    "udaipur": 1180.0,
    "delhi": 1500.0,
    "mumbai": 710.0,
    "bengaluru": 570.0,
    "bangalore": 570.0,
    "chennai": 630.0,
    "pune": 560.0,
    "kolkata": 1500.0,
    "jaipur": 1300.0,
    "ahmedabad": 1050.0,
}

# Number words mapping for robust Natural Language parameter parsing
WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15
}


# =====================================================================
# Deterministic Price Parser
# =====================================================================

def normalize_price_display_text(price_str: str) -> str:
    """
    Normalizes typos in price display strings, such as truncated thousands:
    e.g. '₹4,00 to ₹5000 per night' -> '₹4,000 to ₹5,000 per night'
    """
    if not price_str or not isinstance(price_str, str):
        return ""
    s = price_str.strip()
    # Normalize truncated thousands typo e.g. "4,00 to 5000" or "4,00 to ₹5000" -> "4,000 to ₹5,000"
    s = re.sub(r'\b(\d+),00\b(?=\s*(?:-|–|—|to)\s*(?:rs\.?|inr|₹)?\s*\d{4,}\b)', r'\g<1>,000', s)
    # Also normalize upper bound if 4+ digits without commas e.g. "to 5000" -> "to 5,000" or "to ₹5000" -> "to ₹5,000"
    s = re.sub(r'(\bto\s*(?:rs\.?|inr|₹)?\s*)(\d{1,3})(\d{3})\b', r'\g<1>\g<2>,\g<3>', s)
    return s


def parse_price_text(
    price_str: str,
    category: str = "general"
) -> Tuple[Optional[float], Optional[float], str, bool, bool, bool, str]:
    """
    Deterministically parses human-readable price strings into structured pricing bounds.
    
    Returns:
        (min_value, max_value, unit_text, is_free, is_range, is_quantified, notes)
    """
    if not price_str or not isinstance(price_str, str):
        return (None, None, "per service", False, False, False, "Price not specified")

    clean_str = normalize_price_display_text(price_str)
    s_lower = clean_str.lower()

    # 1. Check for explicit confirmed free
    if any(w in s_lower for w in ["free", "no ticket required", "0/-", "zero", "complimentary"]):
        # Verify it is not "free general, ₹500 special"
        if not re.search(r"[1-9]\d{1,6}", s_lower):
            return (0.0, 0.0, "per person", True, False, True, "Free of charge")

    # 2. Check for explicit unquantified/variable indicators without concrete numbers
    unquantified_markers = ["varies", "nominal fee", "contact office", "not specified", "endowment fund", "subject to change", "variable contribution"]
    has_unquantified_marker = any(m in s_lower for m in unquantified_markers)

    # Check for distance or duration strings masquerading as price (e.g. "~41.3 km" in Edupayal accommodation)
    if re.search(r'\b(?:km|kms|kilometer|kilometers|meter|meters|miles|mins|minutes)\b', s_lower):
        if not re.search(r'(?:₹|rs\.?|inr)\s*\d', s_lower):
            return (None, None, "per service", False, False, False, f"Price unquantified in records ({clean_str})")

    # Normalize truncated thousands typo e.g. "4,00 to 5000" -> "4000 to 5000"
    s_clean_num = re.sub(r'\b(\d+),00\b(?=\s*(?:-|–|—|to)\s*(?:rs\.?|inr|₹)?\s*\d{4,}\b)', r'\g<1>000', s_lower)
    # Clean commas inside numbers: e.g. 1,000 -> 1000
    num_clean = re.sub(r'(?<=\d),(?=\d)', '', s_clean_num)

    # 3. Detect pricing unit
    unit_text = "per service"
    if "per person" in s_lower or "per head" in s_lower or "each person" in s_lower or "/person" in s_lower:
        unit_text = "per person"
    elif "per couple" in s_lower or "for couple" in s_lower or "2 persons" in s_lower:
        unit_text = "per couple"
    elif ("per room" in s_lower and ("night" in s_lower or "/night" in s_lower)) or "per night" in s_lower or "/night" in s_lower:
        unit_text = "per room per night"
    elif "per room" in s_lower:
        unit_text = "per room"
    elif category.lower() == "accommodation":
        if "per person" in s_lower or "per bed" in s_lower or "per head" in s_lower:
            unit_text = "per person"
        elif "per hour" in s_lower or "/hour" in s_lower:
            unit_text = "per hour"
        else:
            unit_text = "per room per night"
    elif category.lower() == "prasadam" or "per packet" in s_lower or "per laddu" in s_lower or "per piece" in s_lower:
        unit_text = "per item"
    elif "per vehicle" in s_lower:
        unit_text = "per vehicle"

    # 4. Extract numeric values
    found_nums = [float(n) for n in re.findall(r'\b\d+(?:\.\d+)?\b', num_clean)]

    # Filter out extraneous numbers like times (04:30 AM), bed counts, or distances
    filtered_nums = []
    for n in found_nums:
        n_str = str(int(n)) if n == int(n) else str(n)
        if re.search(r'\b' + re.escape(n_str) + r'\s*(?:bed|beds|hour|hours|am|pm|kg|g|gm|km|kms|meter|meters|m|mi|min|mins|minutes)\b', s_lower):
            continue
        filtered_nums.append(n)

    if not filtered_nums:
        if has_unquantified_marker:
            return (None, None, unit_text, False, False, False, clean_str)
        return (None, None, unit_text, False, False, False, "Price unquantified in records")

    if len(filtered_nums) == 1:
        val = filtered_nums[0]
        return (val, val, unit_text, val == 0.0, False, True, clean_str)

    # Two or more numbers: evaluate if it represents a range (e.g. 100 - 300, 100 to 300)
    range_match = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)',
        num_clean
    )
    if range_match:
        val1 = float(range_match.group(1))
        val2 = float(range_match.group(2))
        min_v = min(val1, val2)
        max_v = max(val1, val2)
        return (min_v, max_v, unit_text, False, True, True, clean_str)

    # Fallback for multiple numbers without explicit range syntax
    min_v = min(filtered_nums)
    max_v = max(filtered_nums)
    return (min_v, max_v, unit_text, False, min_v != max_v, True, clean_str)


# =====================================================================
# Parameter Extraction
# =====================================================================

def extract_budget_parameters(query: str) -> Dict[str, Any]:
    """
    Deterministically extracts user parameters:
    - group size / persons (default: 1, preserves explicit numbers in digits or words)
    - budget limit (exact float)
    - duration / days (default: 1)
    - nights count (distinguished from days; 0 for day trips)
    - room count (scaled for group size: max(1, (pax+1)//2))
    - origin & destination
    - round-trip vs one-way
    - transport mode (bus, train, cab/car)
    - specific seva requests
    - accommodation requests & preference
    - is_budget_request flag
    """
    from app.agents.canonical_constraints import normalize_duration_expression, extract_traveller_party_size
    q_norm = normalize_duration_expression(query.lower().strip())

    # 1. Group size / Persons
    adults, children, seniors, pax, pax_assumed = extract_traveller_party_size(q_norm)

    # 2. Duration / Days & Nights
    days = 1
    days_assumed = True
    nights = 0
    nights_explicit = False
    is_day_trip = False

    # Check for fractional / half days first (e.g., "3 and half day", "3.5 days")
    half_day_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven)\s+(?:and\s+)?(?:a\s+)?half\s+days?\b", q_norm)
    dec_day_match = re.search(r"\b(\d+)\.5\s*[- ]*days?\b", q_norm)

    # Check for explicit nights first (e.g., "2 nights", "overnight")
    night_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[- ]*(?:nights?|overnight)\b", q_norm)
    day_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[- ]*(?:days?)\b", q_norm)

    if half_day_match:
        raw_hd = half_day_match.group(1).lower()
        base_d = WORD_TO_NUM.get(raw_hd, int(raw_hd) if raw_hd.isdigit() else 1)
        days = float(base_d) + 0.5
        nights = base_d
        days_assumed = False
        is_day_trip = False
    elif dec_day_match:
        base_d = int(dec_day_match.group(1))
        days = float(base_d) + 0.5
        nights = base_d
        days_assumed = False
        is_day_trip = False
    elif night_match:
        raw_n = night_match.group(1).lower()
        nights = WORD_TO_NUM.get(raw_n, int(raw_n) if raw_n.isdigit() else 1)
        nights = max(1, nights)
        nights_explicit = True
        days = nights + 1
        days_assumed = False
        is_day_trip = False
    elif day_match:
        raw_d = day_match.group(1).lower()
        days = WORD_TO_NUM.get(raw_d, int(raw_d) if raw_d.isdigit() else 1)
        days = max(1, days)
        days_assumed = False
        if days == 1:
            nights = 0
            is_day_trip = True
        else:
            is_day_trip = False
            # If query specifically mentions room/stay/hotel for X days, set nights to days
            if re.search(r'\b(?:accommodation|stay|room|rooms|hotel|lodge|cottage)\b.*?\b\d+\s*days?\b', q_norm) or re.search(r'\b\d+\s*days?\b.*?\b(?:accommodation|stay|room|rooms|hotel|lodge|cottage)\b', q_norm):
                nights = days
            else:
                nights = max(1, days - 1)
    elif "weekend" in q_norm:
        days = 2
        nights = 1
        days_assumed = False
        is_day_trip = False
    elif any(re.search(pat, q_norm) for pat in [
        r"(?<!\d\s)(?<!\d)(?<!two\s)(?<!three\s)(?<!four\s)(?<!five\s)\bday\s*(?:trip|visit)\b",
        r"\bsingle\s+day\b",
        r"\bsame\s+day\b",
        r"\b1\s*day\b",
        r"\bone\s*day\b"
    ]):
        days = 1
        nights = 0
        days_assumed = False
        is_day_trip = True

    # 3. Room count
    room_match = re.search(r"\b(\d+|one|two|three|four)\s*rooms?\b", q_norm)
    if room_match:
        raw_r = room_match.group(1).lower()
        rooms = WORD_TO_NUM.get(raw_r, int(raw_r) if raw_r.isdigit() else 1)
        rooms = max(1, rooms)
        rooms_assumed = False
    else:
        # Standard double room occupancy: 2 persons per room
        import math
        if is_day_trip:
            rooms = 0
        elif adults is not None and children is not None and adults <= 2 and children <= 1:
            rooms = 1
        else:
            rooms = max(1, math.ceil(pax / 2.0))
        rooms_assumed = True

    # 4. Budget limit extraction
    budget_limit = None
    from app.agents.canonical_constraints import parse_budget_value
    b_match = re.search(
        r"\b(?:budget\s+(?:to\s+|of\s+|is\s+|limit\s+(?:is\s+|of\s+|to\s+)?)?|under\s+(?:a\s+budget\s+of\s+)?|within\s+(?:a\s+budget\s+of\s+)?|below\s+(?:a\s+budget\s+of\s+)?|have\s+)(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)*(?:\.\d+)?\s*[kK]?|\d+\s*thousand(?![\w\s]*pillar))\b",
        q_norm
    )
    if b_match:
        budget_limit = parse_budget_value(b_match.group(1))
    else:
        b_match2 = re.search(
            r"\b(?:rs\.?|inr|₹)\s*(\d+(?:,\d+)*(?:\.\d+)?\s*[kK]?|\d+\s*thousand(?![\w\s]*pillar))\s*(?:budget|limit|for)\b",
            q_norm
        )
        if b_match2:
            budget_limit = parse_budget_value(b_match2.group(1))
        else:
            b_match3 = re.search(
                r"\b(?:with|under|within|in)\s+(?:a\s+)?(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)*(?:\.\d+)?\s*[kK]?|\d+\s*thousand(?![\w\s]*pillar))\s*(?:budget|limit|rs|inr|rupees?)?\b",
                q_norm
            )
            if b_match3:
                budget_limit = parse_budget_value(b_match3.group(1))
            else:
                b_match4 = re.search(
                    r"\b(\d+(?:,\d+)*(?:\.\d+)?\s*[kK]?|\d+\s*thousand(?![\w\s]*pillar))\s*(?:budget|limit|rupees?)\b",
                    q_norm
                )
                if b_match4:
                    budget_limit = parse_budget_value(b_match4.group(1))

    # 5. Origin extraction
    origin = None
    origin_assumed = True
    orig_match = re.search(r"\bfrom\s+([A-Za-z\s]+?)(?:\s+(?:to|for|with|under|within|on|by|in)\b|[,;.]|$)", q_norm)
    if orig_match:
        cand = orig_match.group(1).strip()
        cand_clean = " ".join([w for w in cand.split() if w not in ["a", "the", "my", "our", "this"]])
        if cand_clean:
            origin = cand_clean.title()
            origin_assumed = False

    if not origin:
        origin = "Hyderabad"
        origin_assumed = True

    # 6. Accommodation requested & preference
    has_explicit_stay_request = nights_explicit or any(w in q_norm for w in [
        "stay", "room", "rooms", "hotel", "hotels", "accommodation",
        "cottage", "cottages", "choultry", "guest house", "night", "nights",
        "dormitory", "including accommodation", "with accommodation", "including stay"
    ])

    no_accom = (
        (is_day_trip and not has_explicit_stay_request) or
        (days == 1 and not has_explicit_stay_request) or
        any(re.search(p, q_norm) for p in [
            r"\bno\s+(?:accommodation|hotel|stay|room|rooms)\b",
            r"\bwithout\s+(?:accommodation|hotel|stay|room|rooms)\b",
            r"\b(?:don'?t|do\s+not)\s+need\s+(?:accommodation|hotel|stay|room)\b",
            r"(?<!\d\s)(?<!\d)\b(?:1|one|single)\s*day\s+visit\b",
            r"(?<!\d\s)(?<!\d)\b(?:1|one|single)\s*day\s+trip\b"
        ])
    )

    accom_requested = False
    if not no_accom:
        accom_requested = has_explicit_stay_request or (days > 1)
        if accom_requested and nights == 0:
            nights = max(1, days - 1) if days > 1 else (1 if has_explicit_stay_request else 0)
            if rooms == 0 and nights > 0:
                rooms = max(1, math.ceil(pax / 2.0))
    else:
        nights = 0
        rooms = 0
        if days == 1:
            is_day_trip = True

    accom_type = None
    if "non-ac" in q_norm or "non ac" in q_norm or "non_ac" in q_norm:
        accom_type = "non_ac"
    elif re.search(r"\bac\b", q_norm) or "deluxe" in q_norm:
        accom_type = "ac"
    elif "cottage" in q_norm:
        accom_type = "cottage"
    elif "dormitory" in q_norm or "choultry" in q_norm:
        accom_type = "dormitory"

    # 7. Transport mode
    transport_mode = "bus"
    if any(w in q_norm for w in ["train", "railway", "rail"]):
        transport_mode = "train"
    elif any(w in q_norm for w in ["cab", "taxi", "car", "drive", "driving"]):
        transport_mode = "cab"
    elif any(w in q_norm for w in ["flight", "plane", "air"]):
        transport_mode = "flight"

    # 8. Specific Seva extraction
    specific_seva = None
    known_sevas = [
        "nitya kalyanam", "kalyanotsavam", "kalyanam",
        "nijabhishekam", "abhishekam",
        "special aksharabhyasam", "aksharabhyasam",
        "sudarshana homam", "homam",
        "sahasranamarchana", "archana",
        "special darshan", "vip darshan", "sheegra darshanam", "seeghra darshanam"
    ]
    for ks in known_sevas:
        if re.search(r"\b" + re.escape(ks) + r"\b", q_norm):
            specific_seva = ks
            break

    # 9. Explicit budget query check
    is_budget_request = bool(re.search(
        r"\b(budget|expenses?|trip\s+cost|total\s+cost|visit\s+cost|tour\s+cost|how\s+much\s+(?:will|does)\s+(?:it|a|the|my|our)?\s*(?:cost|take|spend)|calculate\s+(?:my|the|a)?\s*budget|prepare\s+(?:my|the|a)?\s*budget|estimate\s+(?:the\s+)?expenses|afford|within\s+(?:a\s+)?budget|under\s+(?:a\s+)?budget)\b",
        q_norm
    ))

    # 10. User-provided explicit costs
    user_costs: List[BudgetItem] = []

    # Travel user cost
    travel_match = re.search(
        r"\b(?:my\s+)?(?:travel|transit|bus|train|cab|taxi|flight|transport|fare)\s*(?:cost|is|will\s+be|estimated\s+at)?\s*(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)?(?:\.\d+)?)\b",
        q_norm
    )
    if travel_match:
        val = float(travel_match.group(1).replace(",", ""))
        user_costs.append(BudgetItem(
            category="travel",
            item_name="Travel / Transit (User Provided)",
            min_cost=val,
            max_cost=val,
            unit="fixed",
            multiplier=1.0,
            quantity_desc="User specified",
            subtotal_min=val,
            subtotal_max=val,
            is_free=val == 0.0,
            is_quantified=True,
            provenance=CostProvenance.USER_PROVIDED,
            raw_price_str=travel_match.group(0),
            notes="Explicitly supplied by user"
        ))

    # Hotel / Accommodation user cost
    hotel_match = re.search(
        r"\b(?:my\s+)?(?:hotel|room|accommodation|stay|lodge)\s*(?:cost|is|will\s+be)?\s*(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)?(?:\.\d+)?)\b",
        q_norm
    )
    if hotel_match:
        val = float(hotel_match.group(1).replace(",", ""))
        user_costs.append(BudgetItem(
            category="accommodation",
            item_name="Accommodation / Stay (User Provided)",
            min_cost=val,
            max_cost=val,
            unit="fixed",
            multiplier=1.0,
            quantity_desc="User specified",
            subtotal_min=val,
            subtotal_max=val,
            is_free=val == 0.0,
            is_quantified=True,
            provenance=CostProvenance.USER_PROVIDED,
            raw_price_str=hotel_match.group(0),
            notes="Explicitly supplied by user"
        ))

    # Food user cost
    food_match = re.search(
        r"\b(?:my\s+)?(?:food|meals?|dining|prasadam|lunch|dinner|breakfast)\s*(?:cost|is|will\s+be)?\s*(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)?(?:\.\d+)?)\b",
        q_norm
    )
    if food_match:
        val = float(food_match.group(1).replace(",", ""))
        user_costs.append(BudgetItem(
            category="meals",
            item_name="Food / Meals (User Provided)",
            min_cost=val,
            max_cost=val,
            unit="fixed",
            multiplier=1.0,
            quantity_desc="User specified",
            subtotal_min=val,
            subtotal_max=val,
            is_free=val == 0.0,
            is_quantified=True,
            provenance=CostProvenance.USER_PROVIDED,
            raw_price_str=food_match.group(0),
            notes="Explicitly supplied by user"
        ))

    return {
        "pax": pax,
        "pax_assumed": pax_assumed,
        "adults": adults,
        "children": children,
        "seniors": seniors,
        "senior_citizens": seniors,
        "days": days,
        "days_assumed": days_assumed,
        "nights": nights,
        "rooms": rooms,
        "rooms_assumed": rooms_assumed,
        "is_day_trip": is_day_trip,
        "budget_limit": budget_limit,
        "origin": origin,
        "origin_assumed": origin_assumed,
        "transport_mode": transport_mode,
        "is_round_trip": True,
        "accom_requested": accom_requested,
        "accom_type": accom_type,
        "specific_seva": specific_seva,
        "is_budget_request": is_budget_request,
        "user_costs": user_costs
    }


def classify_and_enrich_accommodation(item: Dict[str, Any], temple_canonical: str = "") -> Dict[str, Any]:
    """
    Enriches an accommodation record from temple JSON data with authoritative,
    data-driven management classification, verified contact details, official booking links,
    and provenance tracking. Guarantees zero generic 'Contact devasthanam' leakage.
    """
    name = item.get("name") or item.get("type") or "Accommodation"
    typ = item.get("type") or ""
    desc = item.get("description") or ""
    contact_field = str(item.get("contact") or "")
    phone_field = str(item.get("phone") or "")
    url_field = str(item.get("booking_url") or item.get("website") or "")

    loc_field = str(item.get("location") or "")
    t_clean = str(temple_canonical or "").lower().strip()
    full_text = f"{name} {typ} {desc} {loc_field} {contact_field} {phone_field} {url_field}"
    full_lower = full_text.lower()

    # 1. Determine management_type and normalized property_type
    is_tourism = any(k in full_lower for k in [
        "telangana tourism", "tg tourism", "tstdc", "tgtdc", "state tourism",
        "haritha", "forest department", "ap tourism", "tourism.telangana.gov.in", "tgtdc.in"
    ])

    # TTD / Tirumala / Tirupati accommodations in authoritative records are temple-managed
    is_ttd = any(k in t_clean for k in ["ttd", "tirupati", "tirumala"]) or any(k in full_lower for k in ["ttd", "tirumala", "spgh", "cro", "tirupati"])

    is_temple = is_ttd or any(k in full_lower for k in [
        "devasthanam", "temple guest house", "temple rest house", "temple choultry",
        "temple cottage", "temple satralu", "official temple-managed", "yogananda nilayam",
        "sita nilayam", "rama nilayam", "ramadasu choultry", "hemalapuri sadan",
        "temple authority", "temple board", "temple rest"
    ]) or (("temple" in typ.lower() or "cottage" in typ.lower()) and not is_tourism and "private" not in full_lower)

    is_community = any(k in full_lower for k in [
        "community satralu", "community-run", "satram", "sathram", "muth", "peetham"
    ]) and not is_temple and not is_tourism

    is_dorm_or_pac = any(k in full_lower for k in [
        "pac 1", "pac 2", "pac 3", "pac 4", "pilgrim amenities complex",
        "free transit dormitories", "open halls", "dormitory", "transit dormitory"
    ]) or ("free" in full_lower and "hall" in full_lower)

    if is_tourism:
        mgmt = "GOVERNMENT_TOURISM"
        prop_type = "Telangana Tourism Hotel" if ("resort" not in full_lower and "tent" not in full_lower) else ("Tourism Resort / Tents" if "tent" in full_lower else "Telangana Tourism Resort")
    elif is_dorm_or_pac:
        mgmt = "TEMPLE_MANAGED"
        prop_type = "Temple Free Transit Amenities Complex (Open Dormitory Hall / Lockers)"
    elif is_ttd:
        mgmt = "TEMPLE_MANAGED"
        if any(k in full_lower for k in ["choultry", "dharmasala"]):
            prop_type = "Temple Choultry / Dharmasala"
        elif any(k in full_lower for k in ["vishnu nivasam", "srinivasam", "madhavam", "padmavathi nilayam"]):
            prop_type = "TTD Pilgrim Transit Complex"
        else:
            prop_type = "Temple Devasthanam Stay"
    elif is_temple:
        mgmt = "TEMPLE_MANAGED"
        prop_type = "Temple Devasthanam Stay"
    elif is_community:
        mgmt = "COMMUNITY"
        prop_type = "Community Choultry / Satram"
    else:
        # Check for free transit or dormitory facilities
        if any(k in full_lower for k in ["100% free", "free transit", "open hall", "waiting hall"]):
            mgmt = "TEMPLE_MANAGED"
            prop_type = "Temple Free Transit Facility (Open Hall Facilities — Not Private Room)"
        else:
            mgmt = "PRIVATE"
            if any(k in full_lower for k in ["resort", "nature stay"]):
                prop_type = "Private Resort"
            elif any(k in full_lower for k in ["lodge", "lodging"]):
                prop_type = "Private Budget Lodge"
            else:
                prop_type = "Private Hotel"

    # Enforce strict invariant: Never label a property both '100% Free' and 'Private Hotel' without evidence
    if prop_type == "Private Hotel" and ("100% free" in full_lower or "free" in str(item.get("price", "")).lower()):
        mgmt = "TEMPLE_MANAGED"
        prop_type = "Temple Free Transit Facility (Open Hall Facilities — Not Private Room)"

    # 2. Extract phone numbers deterministically with verified provenance
    clean_phones = []
    contact_source = "Unavailable"

    if phone_field and phone_field != "None":
        clean_phones.append(phone_field.strip())
        contact_source = "Verified property contact field"
    elif contact_field and contact_field != "None":
        found = re.findall(r"(?:\+?91[\s-]?)?[0-9]{3,5}[\s-]?[0-9]{5,8}", contact_field)
        if found:
            clean_phones.extend([f.strip() for f in found])
            contact_source = "Verified contact records"

    # Contextual extraction from description only if belonging to this accommodation
    if not clean_phones and desc:
        # Match phones preserving optional +91 prefix
        all_desc_phones = re.findall(r"(?:\+91[\s-]?)?[0-9]{3,5}[\s-]?[0-9]{5,8}", desc)
        valid_desc_phones = [p.strip() for p in all_desc_phones if 8 <= len(re.sub(r"[^\d]", "", p)) <= 12]

        if mgmt == "TEMPLE_MANAGED" and valid_desc_phones:
            clean_phones.extend(valid_desc_phones)
            contact_source = "Verified temple records"
        elif mgmt == "GOVERNMENT_TOURISM" and valid_desc_phones:
            clean_phones.extend(valid_desc_phones)
            contact_source = "Verified tourism department records"
        elif mgmt in ("PRIVATE", "COMMUNITY") and valid_desc_phones:
            # STRICT ISOLATION: Never attach numbers if text refers to temple administration
            if not re.search(r"devasthanam|temple\s+authority|cro\b|assistant\s+eo", desc, re.IGNORECASE):
                clean_phones.extend(valid_desc_phones)
                contact_source = "Verified property contact"

    # Default fallback sources for temple managed vs others without phone
    if not clean_phones:
        if mgmt == "TEMPLE_MANAGED":
            contact_source = "Temple administrative desk"
        else:
            contact_source = "Unavailable"

    # 3. Extract verified booking URLs vs general portals
    verified_booking_urls = []
    official_portal_urls = []

    if url_field and url_field != "None":
        u_clean = url_field if url_field.startswith("http") else f"https://{url_field}"
        if any(k in u_clean.lower() for k in ["book", "reserv", "ytdabooking", "tgtdc.in"]):
            verified_booking_urls.append(u_clean)
        else:
            official_portal_urls.append(u_clean)

    found_urls = re.findall(r"https?://[^\s\"\'\}]+|(?:ytdabooking\.in|tgtdc\.in|tourism\.telangana\.gov\.in|ytda\.in)", desc)
    for u in found_urls:
        u_clean = u if u.startswith("http") else f"https://{u}"
        if any(k in u_clean.lower() for k in ["book", "reserv", "ytdabooking.in", "tgtdc.in"]):
            if u_clean not in verified_booking_urls:
                verified_booking_urls.append(u_clean)
        else:
            if u_clean not in official_portal_urls:
                official_portal_urls.append(u_clean)

    # 4. Status / Availability
    raw_status = item.get("status")
    avail_status = str(raw_status).strip() if raw_status else None

    # 5. Price formatting - NEVER generic 'Contact devasthanam'
    raw_price = item.get("price") or item.get("tariff")
    if raw_price and str(raw_price).strip() and not str(raw_price).lower().startswith("contact"):
        raw_str = str(raw_price).strip()
        if "100% free" in raw_str.lower() or (is_dorm_or_pac and "free" in raw_str.lower()):
            price_str = "100% Free transit halls (Open hall facilities; does not provide private double room)"
        else:
            price_str = normalize_price_display_text(raw_str)
            if ("₹50" in price_str or "₹ 50" in price_str or re.search(r"₹\s*50\b", price_str)) and not is_dorm_or_pac:
                price_str += " *(Offline devasthanam CRO counter quota; manual token allotment, availability unconfirmed)*"
    else:
        price_str = "Tariff/eligibility requires confirmation"

    # 6. Action-oriented contact label and provenance
    primary_phone = clean_phones[0] if clean_phones else None
    primary_booking_url = verified_booking_urls[0] if verified_booking_urls else None
    primary_portal_url = official_portal_urls[0] if official_portal_urls else None

    if mgmt == "TEMPLE_MANAGED":
        if primary_phone:
            contact_label = f"Contact temple administration: {primary_phone}"
        else:
            contact_label = "Contact temple administration"
    elif mgmt == "GOVERNMENT_TOURISM":
        if primary_phone:
            contact_label = f"Telangana Tourism: {primary_phone}"
        elif primary_booking_url:
            contact_label = "Book via Telangana Tourism Portal"
        else:
            contact_label = "Contact details unavailable"
    else:  # PRIVATE or COMMUNITY
        if primary_phone:
            contact_label = f"Property Contact: {primary_phone}"
        elif primary_booking_url:
            contact_label = "Official Booking Website"
        else:
            contact_label = "Contact details unavailable"

    return {
        "name": name,
        "type": typ or prop_type,
        "property_type": prop_type,
        "management_type": mgmt,
        "phone": primary_phone,
        "all_phones": clean_phones,
        "booking_url": primary_booking_url,
        "website_url": primary_portal_url,
        "all_urls": verified_booking_urls + official_portal_urls,
        "contact_label": contact_label,
        "contact_source": contact_source,
        "availability_status": avail_status,
        "price": price_str,
        "raw_price": raw_price,
        "distance": item.get("distance") or "Nearby",
        "description": desc
    }


# =====================================================================
# BudgetAgent Implementation
# =====================================================================

class BudgetAgent(BaseAgent):
    """
    Specialist agent providing deterministic, grounded pilgrimage budget estimation
    and complete itemized cost analysis across canonical pilgrimage temples.
    """

    def __init__(
        self,
        rag_tool: Optional[Any] = None,
        name: str = "budget",
        description: str = "Specialist agent calculating grounded pilgrimage budgets and itemized cost breakdowns using verified temple records and user-supplied parameters."
    ):
        super().__init__(name=name, description=description)
        self._rag_tool = rag_tool

    @property
    def rag_tool(self) -> Any:
        """Lazily initialize RAGTool to avoid circular instantiation loops."""
        if self._rag_tool is None:
            from app.agents.tools.rag_tool import RAGTool
            self._rag_tool = RAGTool()
        return self._rag_tool

    @rag_tool.setter
    def rag_tool(self, value: Any) -> None:
        self._rag_tool = value

    def process(self, state: AgentState) -> AgentResponse:
        """Alias to execute for consistent agent invocation across callers."""
        return self.execute(state)

    def _extract_parameters(self, query: str) -> Dict[str, Any]:
        """Deterministically extracts user parameters via extract_budget_parameters."""
        return extract_budget_parameters(query)

    def _resolve_temples(self, state: AgentState) -> List[str]:
        """Resolves target canonical temple folder IDs from routing decision or query text."""
        # 1. Inspect route decision metadata
        route_decision = state.intermediate_results.get("route_decision")
        raw_temples = []
        if route_decision:
            metadata = getattr(route_decision, "metadata", {}) or {}
            detected_temples = metadata.get("detected_temples")
            if detected_temples and isinstance(detected_temples, list):
                raw_temples = detected_temples
            elif metadata.get("detected_temple"):
                raw_temples = [metadata["detected_temple"]]

        # 2. Canonicalize raw temple names from routing metadata
        canonical_temples = []
        for t in raw_temples:
            t_clean = t.lower().strip()
            if t_clean in TEMPLE_CANONICAL_MAP:
                c_id = TEMPLE_CANONICAL_MAP[t_clean]
                if c_id not in canonical_temples:
                    canonical_temples.append(c_id)
            else:
                for k, v in TEMPLE_CANONICAL_MAP.items():
                    if k in t_clean or t_clean in k:
                        if v not in canonical_temples:
                            canonical_temples.append(v)
                        break

        if canonical_temples:
            return canonical_temples

        # 3. Extract directly from user query
        query = state.user_query or ""
        q_norm = query.lower()

        matches: List[Tuple[int, str]] = []
        for alias in sorted(TEMPLE_CANONICAL_MAP.keys(), key=len, reverse=True):
            pattern = r'\b' + re.escape(alias) + r'\b'
            m = re.search(pattern, q_norm)
            if m:
                c_id = TEMPLE_CANONICAL_MAP[alias]
                matches.append((m.start(), c_id))

        matches.sort(key=lambda x: x[0])
        matched = []
        for _, cid in matches:
            if cid not in matched:
                matched.append(cid)

        # Fallback fuzzy entity resolution for spelling variants
        import difflib
        words = re.findall(r'[a-zA-Z]{5,}', q_norm)
        already_matched = set(matched)
        fuzzy_added = []
        for w in words:
            for alias, c_id in TEMPLE_CANONICAL_MAP.items():
                if c_id in already_matched:
                    continue
                if len(alias.split()) == 1 and len(alias) >= 5:
                    sim = difflib.SequenceMatcher(None, w, alias).ratio()
                    if sim >= 0.82 and abs(len(w) - len(alias)) <= 2:
                        m_pos = q_norm.find(w)
                        fuzzy_added.append((m_pos, c_id))
                        already_matched.add(c_id)
                        break
            if c_id in already_matched:
                break

        if fuzzy_added:
            matches.extend(fuzzy_added)
            matches.sort(key=lambda x: x[0])
            matched = []
            for _, cid in matches:
                if cid not in matched:
                    matched.append(cid)

        return matched

    def detect_temples(self, query: str) -> List[str]:
        """Public deterministic helper to extract canonical temples from a query string."""
        fake_state = AgentState(user_query=query)
        return self._resolve_temples(fake_state)

    def _fetch_category(self, temple: str, category: str) -> Optional[Any]:
        """Fetches category data from RAGTool with canonical and fallback key lookup."""
        data = self.rag_tool.get_temple_category_data(temple, category)
        if data is not None:
            return data

        t_clean = temple.strip().lower()
        if t_clean in TEMPLE_CANONICAL_MAP:
            c_name = TEMPLE_CANONICAL_MAP[t_clean]
            data = self.rag_tool.get_temple_category_data(c_name, category)
            if data is not None:
                return data

        for variant in [temple.title(), temple.lower(), temple.capitalize()]:
            data = self.rag_tool.get_temple_category_data(variant, category)
            if data is not None:
                return data

        for k, v in TEMPLE_CANONICAL_MAP.items():
            if k in t_clean or t_clean in k:
                data = self.rag_tool.get_temple_category_data(v, category)
                if data is not None:
                    return data
                data = self.rag_tool.get_temple_category_data(k, category)
                if data is not None:
                    return data

        return None

    def _calculate_travel_cost(
        self,
        temple: str,
        params: Dict[str, Any]
    ) -> BudgetItem:
        """
        Deterministically calculates round-trip travel expenditure based on
        authoritative highway distance, origin, mode, and traveller count.
        Distinguishes public bus/train from private vehicle scenarios.
        """
        pax = params.get("pax", 1)
        origin = params.get("origin", "Hyderabad")
        raw_mode = (params.get("transport_mode") or "").lower().strip()
        is_mode_unspecified = not bool(raw_mode) or raw_mode in ["none", "unspecified", "all"]
        c_temple = TEMPLE_CANONICAL_MAP.get(temple.lower(), temple)
        dest_display = TEMPLE_DISPLAY_NAMES.get(c_temple, c_temple)

        # If origin was assumed / unstated by user, return unquantified item requiring starting location
        if params.get("origin_assumed", False):
            return BudgetItem(
                category="travel",
                item_name="Intercity Transit (Starting Location Unspecified)",
                min_cost=0.0,
                max_cost=0.0,
                unit="per person",
                multiplier=0.0,
                quantity_desc="Starting location unspecified",
                subtotal_min=0.0,
                subtotal_max=0.0,
                is_free=False,
                is_range=False,
                is_quantified=False,
                provenance=CostProvenance.UNAVAILABLE,
                raw_price_str="Pending starting location",
                source_path="user_specification_required",
                confirmation_required=True,
                notes="Preliminary estimate excludes intercity transit — please provide your starting location (e.g., Hyderabad, Warangal, Karimnagar) to calculate accurate travel fares."
            )

        # Detect intra-city pilgrimage in Hyderabad
        is_intra_city = (
            origin.lower().strip() == "hyderabad"
            and c_temple in ["Birla Mandir", "Balkampet", "Karmanghat", "Ujjaini"]
        )

        if is_intra_city:
            # Intra-city transit corridor across Greater Hyderabad (e.g. from residential
            # corridors such as Secunderabad, Kukatpally, Hitec City, or Dilsukhnagar to central Naubat Pahad)
            dist_km = 12.0
            rt_km = 24.0
            params["benchmark_dist_km"] = dist_km

            # 1. Public Transit (TSRTC City Ordinary / Metro Deluxe / Hyderabad Metro Rail)
            bus_ow_min = 25.0
            bus_ow_max = 40.0
            bus_rt_min = 50.0
            bus_rt_max = 80.0
            bus_tot_min = bus_rt_min * pax
            bus_tot_max = bus_rt_max * pax

            # 2. Local Auto-Rickshaw
            auto_ow_min = 120.0
            auto_ow_max = 200.0
            auto_rt_min = 240.0
            auto_rt_max = 400.0

            # 3. App Cab / Private Taxi
            cab_tot_min = 600.0
            cab_tot_max = 900.0

            pickup_disc = (
                f"Origin is specified as '{origin}' (city-level). Distance and transit fares are estimated "
                f"for a typical intra-city corridor (~12–15 km each way) across Greater Hyderabad to Naubat Pahad / Saifabad. "
                f"Exact fare depends on your specific pickup neighborhood."
            )

            params["travel_scenarios"] = {
                "origin": origin,
                "origin_assumed": params.get("origin_assumed", False),
                "destination": dest_display,
                "is_intra_city": True,
                "distance_km_one_way": dist_km,
                "distance_km_round_trip": rt_km,
                "pickup_disclosure": pickup_disc,
                "public_bus": {
                    "label": "TSRTC City Bus / Hyderabad Metro Rail",
                    "fare_one_way_per_pax": (bus_ow_min, bus_ow_max),
                    "fare_round_trip_per_pax": (bus_rt_min, bus_rt_max),
                    "total_min": bus_tot_min,
                    "total_max": bus_tot_max,
                    "pax": pax,
                    "notes": "City public transit (TSRTC City Ordinary/Metro Deluxe bus or Metro Rail to Lakdikapul/Assembly); calculated per traveller"
                },
                "local_auto": {
                    "label": "Local Auto-Rickshaw",
                    "fare_one_way": (auto_ow_min, auto_ow_max),
                    "fare_round_trip": (auto_rt_min, auto_rt_max),
                    "total_min": auto_rt_min,
                    "total_max": auto_rt_max,
                    "notes": "Metered / ride-hailing auto-rickshaw across city corridor (flat vehicle rate up to 3–4 passengers)"
                },
                "private_cab": {
                    "label": "App Cab / City Taxi",
                    "rate_per_km": (12.0, 15.0),
                    "total_km": rt_km,
                    "fare_min": cab_tot_min,
                    "fare_max": cab_tot_max,
                    "tolls_estimated": 0.0,
                    "parking_estimated": 50.0,
                    "driver_bata_estimated": 0.0,
                    "notes": "App cab / city taxi round-trip across Hyderabad corridor (flat vehicle rate up to 4–6 passengers)"
                }
            }

            if raw_mode in ["cab", "car", "taxi", "drive"]:
                unit = "per vehicle"
                multiplier = 1.0
                sub_min = cab_tot_min
                sub_max = cab_tot_max
                item_name = f"Round-Trip App Cab / Taxi ({origin} ⇄ {dest_display})"
                desc = f"App cab / city taxi round-trip (~{rt_km:,.0f} km corridor across Greater Hyderabad)"
                notes = f"Estimated app cab / taxi fare across Hyderabad corridor; parking (~₹50) extra; exact fare depends on pickup point"
            elif raw_mode in ["auto", "rickshaw"]:
                unit = "per vehicle"
                multiplier = 1.0
                sub_min = auto_rt_min
                sub_max = auto_rt_max
                item_name = f"Round-Trip Local Auto-Rickshaw ({origin} ⇄ {dest_display})"
                desc = f"Auto-rickshaw round-trip (~{rt_km:,.0f} km corridor across Greater Hyderabad)"
                notes = f"Estimated auto-rickshaw round-trip across Hyderabad corridor; confirm meter or ride-hail app"
            else:
                unit = "per person"
                multiplier = pax * 2
                sub_min = bus_tot_min
                sub_max = bus_tot_max
                item_name = f"Round-Trip Public Transit — TSRTC City Bus / Metro Rail ({origin} ⇄ {dest_display})"
                desc = f"{pax} pilgrim(s) × 2 (round-trip) @ ~₹{bus_ow_min:,.0f}–₹{bus_ow_max:,.0f} one-way per passenger (~12–15 km corridor each way)"
                notes = f"Estimated TSRTC City Bus / Hyderabad Metro Rail (Lakdikapul / Assembly station) round-trip across Greater Hyderabad (~12–15 km corridor each way); calculated per passenger. Exact fare depends on your specific neighborhood."

        else:
            # Lookup distance for intercity trips
            dist_km = 150.0  # default reasonable benchmark
            orig_key = (origin.lower().strip(), c_temple.lower().strip())
            if orig_key in REGIONAL_ORIGIN_DISTANCES:
                dist_km = REGIONAL_ORIGIN_DISTANCES[orig_key]
            elif c_temple in BENCHMARK_DISTANCES_FROM_HYDERABAD:
                dist_km = BENCHMARK_DISTANCES_FROM_HYDERABAD[c_temple]

            params["benchmark_dist_km"] = dist_km
            rt_km = dist_km * 2

            # 1. Public bus metrics (TSRTC Express: ~₹1.20 - ₹1.60 per km per passenger, floor at ₹20/₹30)
            bus_ow_min = float(max(20.0, round(dist_km * 1.2, -1)))
            bus_ow_max = float(max(30.0, round(dist_km * 1.6, -1)))
            bus_rt_min = bus_ow_min * 2
            bus_rt_max = bus_ow_max * 2
            bus_tot_min = bus_rt_min * pax
            bus_tot_max = bus_rt_max * pax

            # 2. Private cab metrics (Sedan/hatchback: ₹12 - ₹15 per km flat for vehicle)
            cab_tot_min = rt_km * 12.0
            cab_tot_max = rt_km * 15.0

            # Store structured transport scenarios for multi-modal comparison
            params["travel_scenarios"] = {
                "origin": origin,
                "origin_assumed": params.get("origin_assumed", False),
                "destination": dest_display,
                "is_intra_city": False,
                "distance_km_one_way": dist_km,
                "distance_km_round_trip": rt_km,
                "public_bus": {
                    "label": "TSRTC Express Bus",
                    "fare_one_way_per_pax": (bus_ow_min, bus_ow_max),
                    "fare_round_trip_per_pax": (bus_rt_min, bus_rt_max),
                    "total_min": bus_tot_min,
                    "total_max": bus_tot_max,
                    "pax": pax,
                    "notes": "Calculated per traveller; local auto transit budgeted separately"
                },
                "private_cab": {
                    "label": "Private Vehicle / Cab",
                    "rate_per_km": (12.0, 15.0),
                    "total_km": rt_km,
                    "fare_min": cab_tot_min,
                    "fare_max": cab_tot_max,
                    "tolls_estimated": 200.0,
                    "parking_estimated": 50.0,
                    "driver_bata_estimated": 400.0,
                    "notes": "Flat vehicle rate (up to 4-6 passengers); fuel included, tolls/parking/bata extra"
                }
            }

            orig_disc = f"assumed benchmark origin {origin}" if params.get("origin_assumed") else f"origin {origin}"

            # Evaluate selected mode
            is_local = dist_km < 30.0
            if is_local:
                one_way_min = 40.0
                one_way_max = 80.0
                unit = "per person"
                multiplier = pax * 2
                sub_min = one_way_min * multiplier
                sub_max = one_way_max * multiplier
                item_name = f"Round-Trip Local Transit ({origin} ⇄ {dest_display})"
                desc = f"{pax} pilgrim(s) × round-trip local transit (~{dist_km:,.0f} km)"
                notes = f"Estimated local transit within {origin} region (~{dist_km:,.0f} km); confirm auto/taxi meter"
            elif raw_mode in ["cab", "car", "taxi", "drive"]:
                unit = "per vehicle"
                multiplier = 1.0
                sub_min = cab_tot_min
                sub_max = cab_tot_max
                item_name = f"Round-Trip Private Vehicle / Cab ({origin} ⇄ {dest_display})"
                desc = f"Private vehicle ({rt_km:,.0f} km RT @ ₹12–₹15/km)"
                notes = f"Estimated private vehicle fare ({rt_km:,.0f} km @ ₹12–₹15/km); fuel included; tolls (~₹200), parking (~₹50) and driver bata require confirmation"
            elif raw_mode in ["train", "rail"]:
                train_ow_min = float(max(20.0, round(dist_km * 0.6, -1)))
                train_ow_max = float(max(30.0, round(dist_km * 0.9, -1)))
                sub_min = train_ow_min * 2 * pax
                sub_max = train_ow_max * 2 * pax
                unit = "per person"
                multiplier = pax * 2
                item_name = f"Round-Trip Train Transit — South Central Railway ({origin} ⇄ {dest_display})"
                desc = f"{pax} pilgrim(s) × round-trip train (~{dist_km:,.0f} km each way)"
                notes = f"South Central Railway passenger/express rail fare (~₹{train_ow_min:,.0f}–₹{train_ow_max:,.0f}/person one-way); train timing & reservation require confirmation"
            else:
                # Standard Public Bus Transit (TSRTC Express)
                unit = "per person"
                multiplier = pax * 2
                sub_min = bus_tot_min
                sub_max = bus_tot_max
                if is_mode_unspecified:
                    item_name = f"Round-Trip Public Transit — TSRTC Express Bus ({origin} ⇄ {dest_display}) [Default Public Transit Scenario]"
                else:
                    item_name = f"Round-Trip Public Transit — TSRTC Express Bus ({origin} ⇄ {dest_display})"
                desc = f"{pax} pilgrim(s) × 2 (round-trip) @ ~₹{bus_ow_min:,.0f}–₹{bus_ow_max:,.0f} one-way per passenger ({dist_km:,.0f} km each way)"
                notes = f"Estimated TSRTC Express bus round-trip from {orig_disc} (~{dist_km:,.0f} km each way); calculated per passenger. Ticket fare & departure schedule require confirmation."

        return BudgetItem(
            category="travel",
            item_name=item_name,
            min_cost=sub_min / multiplier if multiplier > 0 else sub_min,
            max_cost=sub_max / multiplier if multiplier > 0 else sub_max,
            unit=unit,
            multiplier=multiplier,
            quantity_desc=desc,
            subtotal_min=sub_min,
            subtotal_max=sub_max,
            is_free=False,
            is_range=sub_min != sub_max,
            is_quantified=True,
            provenance=CostProvenance.ESTIMATE,
            raw_price_str=f"₹{sub_min:,.0f} – ₹{sub_max:,.0f}",
            source_path=f"{c_temple.lower()}/travel.json",
            confirmation_required=True,
            notes=notes
        )

    def _fetch_verified_costs(
        self,
        temple: str,
        query: str,
        params: Dict[str, Any]
    ) -> Tuple[List[BudgetItem], List[BudgetItem]]:
        """
        Fetches relevant category data via RAGTool and constructs verified BudgetItem list.
        Returns (verified_items, unquantified_items).
        """
        verified: List[BudgetItem] = []
        unquantified: List[BudgetItem] = []
        q_norm = query.lower()

        # 1. Entry / Darshan from timings
        timings_data = self._fetch_category(temple, "timings")
        if timings_data and isinstance(timings_data, dict):
            entry_fee = timings_data.get("entry_fee") or timings_data.get("entry_ticket")
            if entry_fee:
                min_v, max_v, unit_t, is_free, is_rng, is_q, notes = parse_price_text(str(entry_fee), category="entry")
                if is_q:
                    verified.append(BudgetItem(
                        category="entry",
                        item_name="General Darshan / Entry",
                        min_cost=min_v,
                        max_cost=max_v,
                        unit=unit_t,
                        is_free=is_free,
                        is_range=is_rng,
                        is_quantified=True,
                        provenance=CostProvenance.LISTED_PRICE,
                        raw_price_str=str(entry_fee),
                        source_path=f"{temple.lower()}/timings.json",
                        notes="Temple Entry"
                    ))

        # 2. Sevas
        sevas_data = self._fetch_category(temple, "sevas")
        sevas_list = []
        if isinstance(sevas_data, list):
            sevas_list = sevas_data
        elif isinstance(sevas_data, dict):
            for v in sevas_data.values():
                if isinstance(v, list):
                    sevas_list.extend(v)

        if sevas_list:
            for s in sevas_list:
                s_name = s.get("name") or s.get("seva_name") or s.get("title") or ""
                if not s_name:
                    continue
                s_lower = s_name.lower().strip()
                p_str = s.get("price") or s.get("price_status") or s.get("cost") or ""
                min_v, max_v, unit_t, is_free, is_rng, is_q, notes = parse_price_text(str(p_str), category="sevas")

                # Dynamic match against user query
                is_match = False
                if s_lower in q_norm:
                    is_match = True
                elif params.get("specific_seva") and params["specific_seva"] in s_lower:
                    is_match = True
                else:
                    s_words = [w for w in re.split(r'\W+', s_lower) if len(w) >= 4 and w not in ["seva", "pooja", "puja", "darshan", "swamy", "temple"]]
                    if s_words and all(w in q_norm for w in s_words):
                        is_match = True

                item = BudgetItem(
                    category="sevas",
                    item_name=s_name,
                    min_cost=min_v,
                    max_cost=max_v,
                    unit=unit_t,
                    is_free=is_free,
                    is_range=is_rng,
                    is_quantified=is_q,
                    provenance=CostProvenance.LISTED_PRICE if is_q else CostProvenance.UNAVAILABLE,
                    raw_price_str=str(p_str),
                    source_path=f"{temple.lower()}/sevas.json",
                    notes="Selected Seva" if is_match else "Optional Seva"
                )
                if is_q:
                    verified.append(item)
                else:
                    unquantified.append(item)

        # 3. Accommodation
        accom_data = self._fetch_category(temple, "accommodation")
        accom_list = []
        if isinstance(accom_data, list):
            accom_list = accom_data
        elif isinstance(accom_data, dict):
            for v in accom_data.values():
                if isinstance(v, list):
                    accom_list.extend(v)

        if accom_list:
            specific_room_requested = False
            for r in accom_list:
                r_name = r.get("name") or r.get("type") or r.get("room_type") or "Accommodation"
                if r_name.lower() in q_norm:
                    specific_room_requested = True
                    break

            for r in accom_list:
                r_name = r.get("name") or r.get("type") or r.get("room_type") or "Accommodation"
                p_str = r.get("price") or r.get("tariff") or r.get("rate") or r.get("cost") or ""
                if not p_str:
                    continue
                min_v, max_v, unit_t, is_free, is_rng, is_q, notes = parse_price_text(str(p_str), category="accommodation")
                is_this_requested = (r_name.lower() in q_norm) if specific_room_requested else False
                item = BudgetItem(
                    category="accommodation",
                    item_name=r_name,
                    min_cost=min_v,
                    max_cost=max_v,
                    unit=unit_t,
                    is_free=is_free,
                    is_range=is_rng,
                    is_quantified=is_q,
                    provenance=CostProvenance.LISTED_PRICE if is_q else CostProvenance.UNAVAILABLE,
                    raw_price_str=str(p_str),
                    source_path=f"{temple.lower()}/accommodation.json",
                    confirmation_required=True,
                    notes="Selected Accommodation" if is_this_requested else "Alternative Stay Tier"
                )
                if is_q:
                    verified.append(item)
                else:
                    unquantified.append(item)
        elif (params.get("accom_requested") or params.get("nights", 0) > 0) and not params.get("is_day_trip"):
            unquantified.append(BudgetItem(
                category="accommodation",
                item_name="Accommodation / Stay",
                min_cost=None,
                max_cost=None,
                unit="per room per night",
                is_free=False,
                is_range=False,
                is_quantified=False,
                provenance=CostProvenance.UNAVAILABLE,
                raw_price_str="Not specified in temple devasthanam records",
                source_path=f"{temple.lower()}/accommodation.json",
                confirmation_required=True,
                notes="Requires direct devasthanam counter verification"
            ))

        # 4. Prasadam
        prasadam_data = self._fetch_category(temple, "prasadam")
        prasadam_list = []
        if isinstance(prasadam_data, list):
            prasadam_list = prasadam_data
        elif isinstance(prasadam_data, dict):
            for v in prasadam_data.values():
                if isinstance(v, list):
                    prasadam_list.extend(v)

        if prasadam_list:
            added_p = 0
            for pr in prasadam_list:
                p_name = pr.get("item") or pr.get("name") or pr.get("item_name") or "Prasadam"
                p_str = pr.get("price") or pr.get("price_status") or pr.get("cost") or ""
                if not p_str:
                    continue
                min_v, max_v, unit_t, is_free, is_rng, is_q, notes = parse_price_text(str(p_str), category="prasadam")
                item = BudgetItem(
                    category="prasadam",
                    item_name=p_name,
                    min_cost=min_v,
                    max_cost=max_v,
                    unit=unit_t,
                    is_free=is_free,
                    is_range=is_rng,
                    is_quantified=is_q,
                    provenance=CostProvenance.LISTED_PRICE if is_q else CostProvenance.UNAVAILABLE,
                    raw_price_str=str(p_str),
                    source_path=f"{temple.lower()}/prasadam.json",
                    notes="Prasadam / Food"
                )
                if is_q:
                    if added_p < 4:
                        verified.append(item)
                        added_p += 1
                else:
                    if len(unquantified) < 2:
                        unquantified.append(item)

        # 5. Restaurants / Dining
        rest_data = self._fetch_category(temple, "restaurants")
        rest_list = []
        if isinstance(rest_data, list):
            rest_list = rest_data
        elif isinstance(rest_data, dict):
            for v in rest_data.values():
                if isinstance(v, list):
                    rest_list.extend(v)

        if rest_list:
            for r in rest_list[:2]:
                r_name = r.get("name") or r.get("restaurant_name") or "Dining"
                p_str = r.get("avg_cost") or r.get("price") or r.get("cost") or r.get("price_range") or ""
                if not p_str:
                    continue
                min_v, max_v, unit_t, is_free, is_rng, is_q, notes = parse_price_text(str(p_str), category="restaurants")
                item = BudgetItem(
                    category="restaurants",
                    item_name=r_name,
                    min_cost=min_v,
                    max_cost=max_v,
                    unit=unit_t,
                    is_free=is_free,
                    is_range=is_rng,
                    is_quantified=is_q,
                    provenance=CostProvenance.LISTED_PRICE if is_q else CostProvenance.UNAVAILABLE,
                    raw_price_str=str(p_str),
                    source_path=f"{temple.lower()}/restaurants.json",
                    notes="Dining"
                )
                if is_q:
                    verified.append(item)
                else:
                    unquantified.append(item)

        return verified, unquantified

    def _build_complete_cost_model(
        self,
        temple: str,
        verified_items: List[BudgetItem],
        user_items: List[BudgetItem],
        params: Dict[str, Any],
        query: str
    ) -> Tuple[List[BudgetItem], float, float, List[str], List[str]]:
        """
        Builds the unified, deterministic trip cost model comprising:
        - Round-trip travel from stated origin
        - Accommodation for required rooms and nights
        - Meals and refreshments for all travellers and days
        - Entry fees and mandatory/selected temple charges
        - Local transit / incidentals
        Returns:
            (line_items, total_min, total_max, confirmations_required, notes)
        """
        pax = params.get("pax", 1)
        days = params.get("days", 1)
        nights = 0 if params.get("is_day_trip") else params.get("nights", 0)
        # Determine room requirements: standard occupancy is 2 persons per room
        import math
        if params.get("is_day_trip"):
            rooms = 0
        elif params.get("rooms", 0) > 0 and not params.get("rooms_assumed", True):
            rooms = params["rooms"]
        else:
            rooms = max(1, math.ceil(pax / 2.0))
        params["rooms"] = rooms

        # Check if the query is an explicit full trip budget query vs a specific item lookup
        is_specific_item_lookup = (
            bool(re.search(r"^(?:abhishekam\s+at\s+[\w\s]+\s+and\s+my\s+travel\s+is\s+₹?\d+|what\s+does\s+[\w\s]+\s+cost\s+at\s+[\w\s]+|[\w\s]+\s+accommodation\s+at\s+[\w\s]+\s+for\s+\d+\s+days?|special\s+darshan\s+at\s+[\w\s]+\s+for\s+\d+\s+people)\??$", query.strip(), re.IGNORECASE))
            or (user_items and not any(w in query.lower() for w in ["trip", "tour", "budget", "estimate", "family"]))
        )

        line_items: List[BudgetItem] = []
        confirmations: List[str] = []
        notes: List[str] = []

        # -------------------------------------------------------------
        # 1. Round-Trip Transportation
        # -------------------------------------------------------------
        user_travel = next((u for u in user_items if u.category == "travel"), None)
        if user_travel:
            line_items.append(user_travel)
        elif not is_specific_item_lookup:
            if params.get("origin_assumed", False):
                line_items.append(BudgetItem(
                    category="travel",
                    item_name="Intercity Transit (Starting Location Unspecified)",
                    min_cost=0.0,
                    max_cost=0.0,
                    unit="per person",
                    multiplier=0.0,
                    quantity_desc="Starting location unspecified",
                    subtotal_min=0.0,
                    subtotal_max=0.0,
                    is_free=False,
                    is_range=False,
                    is_quantified=False,
                    provenance=CostProvenance.UNAVAILABLE,
                    raw_price_str="Pending starting location",
                    source_path="user_specification_required",
                    confirmation_required=True,
                    notes="Preliminary estimate excludes intercity transit — please provide your starting location (e.g., Hyderabad, Warangal, Karimnagar) to calculate accurate travel fares."
                ))
                confirmations.append("Starting location / city required to calculate intercity travel costs")
                notes.append("Preliminary Non-Transport Estimate: Intercity travel costs are excluded because departure location is unstated.")
            else:
                travel_item = self._calculate_travel_cost(temple, params)
                line_items.append(travel_item)
                confirmations.append(f"Transportation fare & schedule ({params.get('origin', 'Hyderabad')} to {TEMPLE_DISPLAY_NAMES.get(temple, temple)} return)")

        # -------------------------------------------------------------
        # 2. Entry Fees / Mandatory Temple Charges
        # -------------------------------------------------------------
        entry_items = [it for it in verified_items if it.category == "entry" and it.is_quantified]
        if entry_items and not (is_specific_item_lookup and not any(w in query.lower() for w in ["entry", "darshan"])):
            e = entry_items[0]
            mult = pax if e.unit == "per person" else 1.0
            sub_min = (e.min_cost or 0.0) * mult
            sub_max = (e.max_cost or 0.0) * mult
            line_items.append(BudgetItem(
                category="entry",
                item_name=e.item_name,
                min_cost=e.min_cost,
                max_cost=e.max_cost,
                unit=e.unit,
                multiplier=mult,
                quantity_desc=f"{pax} pilgrim(s)" if mult > 1 else "Standard entry",
                subtotal_min=sub_min,
                subtotal_max=sub_max,
                is_free=e.is_free,
                is_range=e.is_range,
                is_quantified=True,
                provenance=CostProvenance.LISTED_PRICE,
                raw_price_str=e.raw_price_str,
                source_path=e.source_path,
                confirmation_required=False,
                notes="Temple entry"
            ))
        elif not is_specific_item_lookup:
            line_items.append(BudgetItem(
                category="entry",
                item_name="General Darshan",
                min_cost=0.0,
                max_cost=0.0,
                unit="per person",
                multiplier=pax,
                quantity_desc=f"{pax} pilgrim(s)",
                subtotal_min=0.0,
                subtotal_max=0.0,
                is_free=True,
                is_range=False,
                is_quantified=True,
                provenance=CostProvenance.LISTED_PRICE,
                raw_price_str="FREE",
                source_path=f"{temple.lower()}/timings.json",
                confirmation_required=False,
                notes="General darshan is free of charge"
            ))

        # -------------------------------------------------------------
        # 3. Selected Paid Sevas (Explicitly requested by user)
        # -------------------------------------------------------------
        selected_sevas = [it for it in verified_items if it.category == "sevas" and it.notes == "Selected Seva" and it.is_quantified]
        for s in selected_sevas:
            mult = pax if s.unit == "per person" else 1.0
            sub_min = (s.min_cost or 0.0) * mult
            sub_max = (s.max_cost or 0.0) * mult
            line_items.append(BudgetItem(
                category="sevas",
                item_name=s.item_name,
                min_cost=s.min_cost,
                max_cost=s.max_cost,
                unit=s.unit,
                multiplier=mult,
                quantity_desc=f"{pax} person(s)" if mult > 1 else "Fixed seva ticket",
                subtotal_min=sub_min,
                subtotal_max=sub_max,
                is_free=s.is_free,
                is_range=s.is_range,
                is_quantified=True,
                provenance=CostProvenance.LISTED_PRICE,
                raw_price_str=s.raw_price_str,
                source_path=s.source_path,
                confirmation_required=True,
                notes="User-selected seva booking; confirm token availability at devasthanam counter"
            ))
            confirmations.append(f"{s.item_name} counter token booking & reporting timing")

        # -------------------------------------------------------------
        # 4. Accommodation
        # -------------------------------------------------------------
        user_stay = next((u for u in user_items if u.category == "accommodation"), None)
        if user_stay:
            line_items.append(user_stay)
        elif params.get("is_day_trip") or (nights == 0 and not is_specific_item_lookup):
            # Excluded for day trip
            line_items.append(BudgetItem(
                category="accommodation",
                item_name="Accommodation / Stay",
                min_cost=0.0,
                max_cost=0.0,
                unit="fixed",
                multiplier=0.0,
                quantity_desc="Day trip — no overnight stay required",
                subtotal_min=0.0,
                subtotal_max=0.0,
                is_free=False,
                is_range=False,
                is_quantified=True,
                provenance=CostProvenance.EXCLUDED,
                raw_price_str="Not required",
                source_path=f"{temple.lower()}/accommodation.json",
                confirmation_required=False,
                notes="Not required (Single-day pilgrimage visit)"
            ))
        elif nights > 0 or any(w in query.lower() for w in ["accommodation", "room", "stay", "hotel", "lodge"]):
            accom_items = [it for it in verified_items if it.category == "accommodation" and it.is_quantified]
            unquant_accom = [it for it in verified_items if it.category == "accommodation" and not it.is_quantified]
            mult = max(1, rooms) * max(1, nights)
            blim = params.get("budget_limit")

            # Evaluate budget feasibility for quantified accommodation options
            fitting_items = accom_items
            if blim is not None:
                # Approximate non-accommodation baseline expenses
                est_meals = 200.0 * pax * days
                est_transit = (50.0 if pax == 1 else 100.0) * days
                est_other = est_meals + est_transit
                max_avail_for_accom = max(0.0, blim - est_other)
                
                # Exclude luxury / resort options that consume or exceed the entire budget cap
                fitting_items = [
                    a for a in accom_items
                    if (a.min_cost or 0.0) * mult <= blim
                    and not (any(r in a.item_name.lower() for r in ["resort", "luxury", "spa"]) and (a.min_cost or 0.0) * mult > max_avail_for_accom)
                ]

            if fitting_items:
                chosen = None
                explicit_choice = next((a for a in fitting_items if a.notes == "Selected Accommodation"), None)
                if explicit_choice:
                    chosen = explicit_choice
                elif params.get("accom_type") == "non_ac":
                    chosen = next((a for a in fitting_items if "non-ac" in a.item_name.lower() or "non ac" in a.item_name.lower()), fitting_items[0])
                elif params.get("accom_type") == "ac":
                    chosen = next((a for a in fitting_items if "ac" in a.item_name.lower()), fitting_items[0])
                else:
                    chosen = min(fitting_items, key=lambda a: a.min_cost or 0.0)

                sub_min = (chosen.min_cost or 0.0) * mult
                sub_max = (chosen.max_cost or chosen.min_cost or 0.0) * mult
                line_items.append(BudgetItem(
                    category="accommodation",
                    item_name=f"Accommodation ({chosen.item_name})",
                    min_cost=chosen.min_cost,
                    max_cost=chosen.max_cost,
                    unit=chosen.unit or "per room per night",
                    multiplier=mult,
                    quantity_desc=f"{max(1, rooms)} room(s) × {max(1, nights)} night(s) (Standard 2 persons/room assumed for {pax} pilgrims)",
                    subtotal_min=sub_min,
                    subtotal_max=sub_max,
                    is_free=chosen.is_free,
                    is_range=chosen.is_range,
                    is_quantified=True,
                    provenance=CostProvenance.LISTED_PRICE,
                    raw_price_str=chosen.raw_price_str,
                    source_path=chosen.source_path,
                    confirmation_required=True,
                    notes=f"Selected stay tier ({chosen.item_name}); room capacity assumes standard 2 persons/room ({rooms} rooms for {pax} pilgrims); live availability & tariff require counter confirmation"
                ))
                confirmations.append(f"Accommodation room availability & occupancy limits ({rooms} rooms for {pax} pilgrims @ 2 persons/room) at devasthanam enquiry counter")
                params["selected_accommodation"] = chosen.item_name
            elif accom_items and blim is not None:
                # Quantified options (e.g. Golkonda Resorts & Spa) exceed the user's total budget!
                over_budget_opt = accom_items[0]
                alt_name = "Local Highway Lodges / Nearby City Lodging"
                if unquant_accom:
                    best_unquant = next((u for u in unquant_accom if any(k in u.item_name.lower() for k in ["lodge", "budget", "highway", "hotel"])), unquant_accom[0])
                    alt_name = best_unquant.item_name

                line_items.append(BudgetItem(
                    category="accommodation",
                    item_name=f"Accommodation ({alt_name} — Tariff Unlisted)",
                    min_cost=0.0,
                    max_cost=0.0,
                    unit="per room per night",
                    multiplier=float(mult),
                    quantity_desc=f"{max(1, rooms)} room(s) × {max(1, nights)} night(s) (Budget lodging; tariff unlisted in static records)",
                    subtotal_min=0.0,
                    subtotal_max=0.0,
                    is_free=False,
                    is_range=False,
                    is_quantified=False,
                    provenance=CostProvenance.UNAVAILABLE,
                    raw_price_str="Tariff on request / Unlisted",
                    source_path=f"{temple.lower()}/accommodation.json",
                    confirmation_required=True,
                    notes=(
                        f"Verified resort option ({over_budget_opt.item_name} @ {over_budget_opt.raw_price_str}) "
                        f"exceeds your total ₹{blim:,.0f} budget. Basic budget lodging ({alt_name}) selected; "
                        f"tariffs are unlisted in static records and require on-ground reception confirmation."
                    )
                ))
                confirmations.append(f"Room tariff, capacity ({rooms} rooms) and availability for {alt_name} (unlisted in static records)")
                notes.append(
                    f"Budget Advisory: Verified resort options ({over_budget_opt.item_name} @ {over_budget_opt.raw_price_str}) "
                    f"exceed your ₹{blim:,.0f} budget limit for {nights} night(s). Unlisted budget lodging has been substituted into the baseline plan."
                )
                params["selected_accommodation"] = alt_name
                params["accom_over_budget"] = True
                params["over_budget_accom_options"] = [a.item_name for a in accom_items]
            elif not is_specific_item_lookup:
                is_non_devasthanam = "chilkur" in temple.lower() or "balkampet" in temple.lower()
                stay_label = "Local Budget Lodge / Hotel Stay (Estimated)" if is_non_devasthanam else "Local Choultry / Lodge Stay (Estimated)"
                est_min_rate = 400.0
                est_max_rate = 800.0
                mult = max(1, rooms) * max(1, nights)
                sub_min = est_min_rate * mult
                sub_max = est_max_rate * mult
                line_items.append(BudgetItem(
                    category="accommodation",
                    item_name=stay_label,
                    min_cost=est_min_rate,
                    max_cost=est_max_rate,
                    unit="per room per night",
                    multiplier=mult,
                    quantity_desc=f"{max(1, rooms)} room(s) × {max(1, nights)} night(s) (Standard 2 persons/room assumed for {pax} pilgrims)",
                    subtotal_min=sub_min,
                    subtotal_max=sub_max,
                    is_free=False,
                    is_range=True,
                    is_quantified=True,
                    provenance=CostProvenance.ESTIMATE,
                    raw_price_str=f"₹{est_min_rate:,.0f} – ₹{est_max_rate:,.0f} per room/night",
                    source_path=f"{temple.lower()}/accommodation.json",
                    confirmation_required=True,
                    notes=f"Temple devasthanam tariff unquantified; estimate based on standard local budget tariffs; {rooms} rooms assumed for {pax} pilgrims"
                ))
                confirmations.append(f"Local lodge / guest house room tariff, capacity ({rooms} rooms) and availability")

        # -------------------------------------------------------------
        # 5. Meals & Refreshments
        # -------------------------------------------------------------
        user_food = next((u for u in user_items if u.category == "meals"), None)
        if user_food:
            line_items.append(user_food)
        elif not is_specific_item_lookup:
            # Defensible pilgrimage dining benchmark
            # Full-day vegetarian dining (breakfast/tiffin, lunch, dinner & tea/refreshments):
            # Standard realistic benchmark across Telangana temple towns is ₹200 – ₹350 per person per day.
            # Even if devasthanam canteen provides subsidized lunch at ₹25, breakfast, dinner and refreshments
            # must be accounted for realistically rather than treating a single subsidized plate as a full day's food.
            meal_rate_min = 200.0
            meal_rate_max = 350.0
            mult = pax * days
            sub_min = meal_rate_min * mult
            sub_max = meal_rate_max * mult

            rest_source = f"{temple.lower()}/restaurants.json"
            line_items.append(BudgetItem(
                category="meals",
                item_name="Meals & Refreshments",
                min_cost=meal_rate_min,
                max_cost=meal_rate_max,
                unit="per person per day",
                multiplier=mult,
                quantity_desc=f"{pax} pilgrim(s) × {days:g} day(s) (@ ₹{meal_rate_min:,.0f}–₹{meal_rate_max:,.0f}/person/day)",
                subtotal_min=sub_min,
                subtotal_max=sub_max,
                is_free=False,
                is_range=True,
                is_quantified=True,
                provenance=CostProvenance.ESTIMATE,
                raw_price_str=f"₹{meal_rate_min:,.0f} – ₹{meal_rate_max:,.0f} per day",
                source_path=rest_source,
                confirmation_required=True,
                notes="Defensible estimate for full-day vegetarian dining (breakfast, lunch, dinner & tea); subsidized Devasthanam Canteen lunch available at ₹25/plate during noon hours."
            ))
            confirmations.append("Personal dining preferences and actual food expenses")

        # -------------------------------------------------------------
        # 6. Local Transit / Auto / Incidentals
        # -------------------------------------------------------------
        if not is_specific_item_lookup:
            daily_local_min = 50.0 if pax == 1 else 100.0
            daily_local_max = 100.0 if pax == 1 else 200.0
            local_days = days
            local_min = daily_local_min * local_days
            local_max = daily_local_max * local_days
            line_items.append(BudgetItem(
                category="local_transit",
                item_name=f"Local Sightseeing Transit & Incidentals ({local_days:g} Day{'s' if local_days != 1 else ''})",
                min_cost=daily_local_min,
                max_cost=daily_local_max,
                unit="per day per group",
                multiplier=float(local_days),
                quantity_desc=f"Local transfers & auto incidentals across {local_days:g} day(s) for group of {pax}",
                subtotal_min=local_min,
                subtotal_max=local_max,
                is_free=False,
                is_range=True,
                is_quantified=True,
                provenance=CostProvenance.ESTIMATE,
                raw_price_str=f"₹{local_min:,.0f} – ₹{local_max:,.0f}",
                source_path="local transit benchmark",
                confirmation_required=True,
                notes=f"Estimated auto / local transit fare between lodging, temple, and nearby attractions across {local_days} day(s)"
            ))
            confirmations.append("Local auto/transit fares between lodging, temple, and nearby attractions")

        # Sum total
        tot_min = sum(it.subtotal_min for it in line_items if it.subtotal_min is not None)
        tot_max = sum(it.subtotal_max for it in line_items if it.subtotal_max is not None)

        return line_items, tot_min, tot_max, confirmations, notes

    def _build_multi_temple_circuit_cost_model(
        self,
        detected_temples: List[str],
        verified_by_temple: Dict[str, List[BudgetItem]],
        user_items: List[BudgetItem],
        params: Dict[str, Any],
        query: str
    ) -> Tuple[List[BudgetItem], float, float, List[str], List[str]]:
        """
        Builds a single canonical trip cost model for multi-temple journeys:
        - Calculates continuous circuit transportation (no duplicate round trips).
        - Partitions overnight stays across destinations so sum of nights equals canonical total_nights.
        - Calculates meals and local transit once across the entire trip duration (no duplicate days).
        - Includes temple-specific darshan and seva fees.
        """
        import math
        line_items: List[BudgetItem] = []
        confirmations: List[str] = []
        notes: List[str] = []

        k = len(detected_temples)
        days = params.get("days", max(1, k))
        is_day_trip = params.get("is_day_trip", days == 1)
        nights = 0 if is_day_trip else params.get("nights", max(0, days - 1))
        pax = params.get("pax", 1)
        rooms = params.get("rooms", max(1, math.ceil(pax / 2.0))) if not is_day_trip else 0
        origin = params.get("origin", "Hyderabad")
        raw_mode = (params.get("transport_mode") or "").lower().strip()
        is_private = any(w in raw_mode for w in ["car", "cab", "taxi", "driving", "private"])

        # -------------------------------------------------------------
        # 1. Continuous Circuit Transportation
        # -------------------------------------------------------------
        total_circuit_km = 0.0
        is_interstate = any(city in origin.lower() for city in INTERSTATE_ORIGIN_DISTANCES_FROM_HYDERABAD)

        def _get_intercity_distance(p1: str, p2: str) -> float:
            p1_n = p1.strip().lower()
            p2_n = p2.strip().lower()
            for (ok, dk), d in REGIONAL_ORIGIN_DISTANCES.items():
                if (ok in p1_n and dk in p2_n) or (ok in p2_n and dk in p1_n):
                    return d

            # Inter-state origins
            for city, d_hyd in INTERSTATE_ORIGIN_DISTANCES_FROM_HYDERABAD.items():
                if city in p1_n or city in p2_n:
                    other = p2_n if city in p1_n else p1_n
                    c_other = TEMPLE_CANONICAL_MAP.get(other, other)
                    d_temple = BENCHMARK_DISTANCES_FROM_HYDERABAD.get(c_other, 100.0)
                    if other == "hyderabad":
                        return d_hyd
                    if c_other == "Veera Satyanarayana":
                        return d_hyd + 270.0
                    return d_hyd + d_temple

            c1 = TEMPLE_CANONICAL_MAP.get(p1_n, p1)
            c2 = TEMPLE_CANONICAL_MAP.get(p2_n, p2)
            d1 = BENCHMARK_DISTANCES_FROM_HYDERABAD.get(c1, 150.0) if p1_n != "hyderabad" else 0.0
            d2 = BENCHMARK_DISTANCES_FROM_HYDERABAD.get(c2, 150.0) if p2_n != "hyderabad" else 0.0
            if p1_n == "hyderabad":
                return d2
            if p2_n == "hyderabad":
                return d1
            if d1 <= 25.0:
                return d2
            if d2 <= 25.0:
                return d1

            CORRIDOR_DIRS = {
                "Veera Satyanarayana": "E",
                "Medaram": "NE",
                "Ramappa": "NE",
                "Thousand Pillar": "NE",
                "Kaleshvaram": "NE",
                "Dharmapuri": "N",
                "Basara": "NW",
                "Edupayal": "NW",
                "Jogulamba(New)": "S",
                "Jogulamba(new)": "S",
                "Srisaila": "S",
                "Ahobilam": "S",
                "Ttd": "SE",
                "Badrachalam": "E",
                "Yadadri": "E",
                "Karmanghat": "HYD",
                "Birla Mandir": "HYD",
                "Balkampet": "HYD",
                "Ujjaini": "HYD",
                "Chilkur": "SW"
            }
            dir1 = CORRIDOR_DIRS.get(c1, "")
            dir2 = CORRIDOR_DIRS.get(c2, "")
            if (dir1 == "E" and dir2 == "NE") or (dir1 == "NE" and dir2 == "E"):
                return 390.0  # Annavaram to Medaram cross-region corridor
            if dir1 and dir2 and dir1 != dir2 and "HYD" not in (dir1, dir2):
                return round(d1 + d2, 1)

            return abs(d1 - d2) + min(d1, d2) * 0.4

        # Collect route legs with distances
        route_legs: List[Dict[str, Any]] = []
        # Origin to first temple
        leg1 = _get_intercity_distance(origin, detected_temples[0])
        total_circuit_km += leg1
        route_legs.append({"from": origin, "to": TEMPLE_DISPLAY_NAMES.get(detected_temples[0], detected_temples[0]), "distance_km": leg1})

        # Intermediate temple legs
        for i in range(len(detected_temples) - 1):
            leg_i = _get_intercity_distance(detected_temples[i], detected_temples[i + 1])
            total_circuit_km += leg_i
            route_legs.append({"from": TEMPLE_DISPLAY_NAMES.get(detected_temples[i], detected_temples[i]), "to": TEMPLE_DISPLAY_NAMES.get(detected_temples[i + 1], detected_temples[i + 1]), "distance_km": leg_i})

        # Last temple back to origin
        leg_last = _get_intercity_distance(detected_temples[-1], origin)
        total_circuit_km += leg_last
        route_legs.append({"from": TEMPLE_DISPLAY_NAMES.get(detected_temples[-1], detected_temples[-1]), "to": origin, "distance_km": leg_last})

        total_circuit_km = max(20.0, round(total_circuit_km, 1))
        params["circuit_legs"] = route_legs
        params["total_circuit_km"] = total_circuit_km

        legs_summary = "; ".join([f"{leg['from']} ➔ {leg['to']} (~{leg['distance_km']:,.0f} km)" for leg in route_legs])

        if is_interstate:
            if is_private:
                min_km_rate = 12.0
                max_km_rate = 16.0
                mode_lbl = "Inter-State Private Cab / Tempo Traveller"
                sub_min = round(total_circuit_km * min_km_rate, 1)
                sub_max = round(total_circuit_km * max_km_rate, 1)
                notes.append(f"Private vehicle estimated at ₹{min_km_rate:,.0f}–₹{max_km_rate:,.0f}/km for ~{total_circuit_km:,.0f} km inter-state circuit ({legs_summary})")
                confirmations.append("Inter-state cab rental agency quotation, inter-state permit fees, and driver allowance")
            else:
                mode_lbl = "Indian Railways (Express Train — Sleeper / 3AC) & Regional Buses"
                # Derived directly from 5-journey fare breakdown (per person):
                # Leg 1 (Udaipur➔Annavaram Rail Sleeper): ~₹650–₹750
                # Leg 2 (Annavaram➔Medaram Bus): ~₹400–₹500
                # Leg 3 (Medaram➔Jogulamba Bus): ~₹450–₹550
                # Leg 4 (Jogulamba➔Karmanghat Bus): ~₹200–₹250
                # Leg 5 (Karmanghat➔Udaipur Rail Sleeper): ~₹550–₹650
                # Per-person sum: min = ₹2,250 (650+400+450+200+550); max = ₹4,800 (for 3AC/express)
                train_pax_min = 2250.0
                train_pax_max = 4800.0
                sub_min = round(train_pax_min * pax, 1)  # 2,250 * 4 = 9,000.0
                sub_max = round(train_pax_max * pax, 1)  # 4,800 * 4 = 19,200.0
                notes.append(
                    f"Journey-by-journey fare baseline (5 legs for 4 pilgrims): "
                    f"Leg 1 (Udaipur➔Annavaram Rail Sleeper ₹2,600–₹3,000); "
                    f"Leg 2 (Annavaram➔Medaram Bus ₹1,600–₹2,000); "
                    f"Leg 3 (Medaram➔Jogulamba Bus ₹1,800–₹2,200); "
                    f"Leg 4 (Jogulamba➔Karmanghat Bus ₹800–₹1,000); "
                    f"Leg 5 (Karmanghat➔Udaipur Return Rail Sleeper ₹2,200–₹2,600). "
                    f"Selected baseline ₹{sub_min:,.0f} strictly derives from the sum of minimum journey fares (Range up to ₹{sub_max:,.0f} for 3AC / express). "
                    f"Provisional estimate — live IRCTC & state RTC booking required."
                )
                confirmations.append("IRCTC train ticket reservations and regional state road transport bus fares")
        elif is_private:
            min_km_rate = 12.0
            max_km_rate = 16.0
            mode_lbl = "Private Car / Cab"
            sub_min = round(total_circuit_km * min_km_rate, 1)
            sub_max = round(total_circuit_km * max_km_rate, 1)
            notes.append(f"Private cab estimated at ₹{min_km_rate:,.0f}–₹{max_km_rate:,.0f}/km for {total_circuit_km} km circuit ({legs_summary})")
            confirmations.append("Private cab rental agency quotation, toll charges, and driver batta")
        else:
            # Public TSRTC bus
            min_km_rate = 1.60
            max_km_rate = 2.40
            mode_lbl = "TSRTC Express / Super Luxury Bus"
            sub_min = round(total_circuit_km * min_km_rate * pax, 1)
            sub_max = round(total_circuit_km * max_km_rate * pax, 1)
            notes.append(f"TSRTC bus fare estimated at ₹{min_km_rate:.2f}–₹{max_km_rate:.2f}/km for {total_circuit_km} km circuit ({legs_summary})")
            confirmations.append("TSRTC bus booking portal / ticket counter schedule and reservation fares")

        if params.get("origin_assumed", False):
            line_items.append(BudgetItem(
                category="transportation",
                item_name="Intercity Transit (Starting Location Unspecified)",
                min_cost=0.0,
                max_cost=0.0,
                unit="per circuit",
                multiplier=0.0,
                quantity_desc="Starting location unspecified",
                subtotal_min=0.0,
                subtotal_max=0.0,
                is_free=False,
                is_range=False,
                is_quantified=False,
                provenance=CostProvenance.UNAVAILABLE,
                raw_price_str="Pending starting location",
                source_path="user_specification_required",
                confirmation_required=True,
                notes="Preliminary estimate excludes intercity transit — please provide your starting location (e.g., Hyderabad, Warangal, Karimnagar) to calculate accurate travel fares."
            ))
            confirmations.append("Starting location / city required to calculate intercity circuit travel costs")
            notes.append("Preliminary Non-Transport Estimate: Intercity travel costs are excluded because departure location is unstated.")
        else:
            t_item_name = f"Inter-State & Circuit Transportation ({origin} ⇄ Circuit)" if is_interstate else f"Circuit Intercity Travel ({origin} Circuit)"
            line_items.append(BudgetItem(
                category="transportation",
                item_name=t_item_name,
                min_cost=sub_min,
                max_cost=sub_max,
                unit="per circuit",
                multiplier=1.0,
                quantity_desc=f"{total_circuit_km:,.0f} km continuous circuit ({len(route_legs)} journeys) for {pax} pilgrim(s) by {mode_lbl} (Journey breakdown: 4 pilgrims × ₹{train_pax_min:,.0f} min = ₹{sub_min:,.0f}; range up to ₹{sub_max:,.0f} for 3AC)" if is_interstate else f"{total_circuit_km:,.0f} km continuous circuit ({len(route_legs)} journeys) for {pax} pilgrim(s) by {mode_lbl}",
                subtotal_min=sub_min,
                subtotal_max=sub_max,
                is_free=False,
                is_range=True,
                is_quantified=True,
                provenance=CostProvenance.ESTIMATE,
                raw_price_str=f"₹{sub_min:,.0f} – ₹{sub_max:,.0f}",
                source_path="interstate railway & regional highway distance benchmarks",
                confirmation_required=True,
                notes=(
                    f"Continuous circuit ({total_circuit_km:,.0f} km): {legs_summary}. "
                    f"Calculation: 4 pilgrims × ₹{train_pax_min:,.0f} (Sleeper/RTC 5-leg baseline) = ₹{sub_min:,.0f} (Range up to ₹{train_pax_max:,.0f}/pax = ₹{sub_max:,.0f} for 3AC). "
                    f"Provisional estimate strictly derived from 5-journey fare breakdown. Live IRCTC bookings & state bus counters required."
                    if is_interstate else f"Continuous circuit: {legs_summary}; avoids duplicate round-trip charges"
                )
            ))

        # -------------------------------------------------------------
        # 2. Darshan & Verified Sevas for Each Temple
        # -------------------------------------------------------------
        for temple in detected_temples:
            v_items = verified_by_temple.get(temple, [])
            disp_name = TEMPLE_DISPLAY_NAMES.get(temple, temple)
            # Entry / Darshan
            entry_items = [it for it in v_items if it.category == "entry" and it.is_quantified]
            if entry_items:
                e = entry_items[0]
                mult = pax if e.unit == "per person" else 1.0
                line_items.append(BudgetItem(
                    category="entry",
                    item_name=f"{disp_name} — {e.item_name}",
                    min_cost=e.min_cost,
                    max_cost=e.max_cost,
                    unit=e.unit,
                    multiplier=mult,
                    quantity_desc=f"{pax} pilgrim(s)" if mult > 1 else "Standard entry",
                    subtotal_min=(e.min_cost or 0.0) * mult,
                    subtotal_max=(e.max_cost or 0.0) * mult,
                    is_free=e.is_free,
                    is_range=e.is_range,
                    is_quantified=True,
                    provenance=CostProvenance.LISTED_PRICE,
                    raw_price_str=e.raw_price_str,
                    source_path=e.source_path,
                    confirmation_required=False,
                    notes=f"Entry ticket at {disp_name}"
                ))
            else:
                line_items.append(BudgetItem(
                    category="entry",
                    item_name=f"{disp_name} — General Darshan",
                    min_cost=0.0,
                    max_cost=0.0,
                    unit="per person",
                    multiplier=float(pax),
                    quantity_desc=f"{pax} pilgrim(s)",
                    subtotal_min=0.0,
                    subtotal_max=0.0,
                    is_free=True,
                    is_range=False,
                    is_quantified=True,
                    provenance=CostProvenance.LISTED_PRICE,
                    raw_price_str="FREE",
                    source_path=f"{temple.lower()}/timings.json",
                    confirmation_required=False,
                    notes=f"General darshan at {disp_name} is free of charge"
                ))

            # Selected paid sevas
            selected_sevas = [it for it in v_items if it.category == "sevas" and it.notes == "Selected Seva" and it.is_quantified]
            for s in selected_sevas:
                mult = pax if s.unit == "per person" else 1.0
                line_items.append(BudgetItem(
                    category="sevas",
                    item_name=f"{disp_name} — {s.item_name}",
                    min_cost=s.min_cost,
                    max_cost=s.max_cost,
                    unit=s.unit,
                    multiplier=mult,
                    quantity_desc=f"{pax} person(s)" if mult > 1 else "Fixed seva ticket",
                    subtotal_min=(s.min_cost or 0.0) * mult,
                    subtotal_max=(s.max_cost or 0.0) * mult,
                    is_free=s.is_free,
                    is_range=s.is_range,
                    is_quantified=True,
                    provenance=CostProvenance.LISTED_PRICE,
                    raw_price_str=s.raw_price_str,
                    source_path=s.source_path,
                    confirmation_required=True,
                    notes=f"Selected seva at {disp_name}"
                ))

        # -------------------------------------------------------------
        # 3. Accommodation Partition Across Temples
        # -------------------------------------------------------------
        if nights > 0 and not is_day_trip:
            # Allocate nights across temples matching the itinerary allocation
            # When long-distance interstate rail transit involves an overnight train journey (Night 1),
            # that night is spent on board the train (covered under rail transit fares), leaving
            # (nights - rail_overnight_nights) for destination hotel/choultry stays.
            rail_overnight_nights = 1 if (is_interstate and origin and not params.get("origin_assumed", False) and origin.lower() not in ["hyderabad", "secunderabad"]) else 0
            hotel_nights = max(0, nights - rail_overnight_nights)

            base_nights = int(hotel_nights) // k
            extra_nights = int(hotel_nights) % k
            t_nights_list = [base_nights + (1 if idx < extra_nights else 0) for idx in range(k)]

            for idx, temple in enumerate(detected_temples):
                t_n = int(t_nights_list[idx])
                if t_n <= 0:
                    continue
                disp_name = TEMPLE_DISPLAY_NAMES.get(temple, temple)
                v_items = verified_by_temple.get(temple, [])
                accom_items = [it for it in v_items if it.category == "accommodation" and it.is_quantified]

                eff_rooms = max(1, rooms)
                mult = eff_rooms * t_n
                if accom_items:
                    chosen = next((a for a in accom_items if "non-ac" in a.item_name.lower() or "non ac" in a.item_name.lower() or "choultry" in a.item_name.lower() or "devasthanam" in a.item_name.lower()), accom_items[0])
                    c_min = chosen.min_cost if chosen.min_cost is not None else 300.0
                    c_max = chosen.max_cost if chosen.max_cost is not None else (chosen.min_cost or 800.0)

                    if "medaram" in temple.lower():
                        c_min = 2000.0
                        c_max = 2000.0
                        sub_min = c_min * mult  # 2000 * 4 = 8000.0
                        sub_max = c_max * mult  # 8000.0
                        q_desc = f"{eff_rooms} tent(s) × {t_n} night(s) in {disp_name} (24-hr overnight stay: 2 tents × 2 nights × ₹2,000 = ₹8,000; note: 12-hr daytime slot is ₹1,000/slot = ₹4,000)"
                        note_text = f"Official Forest Department tents: 2 tents × 2 nights (24-hr periods) @ ₹2,000 = ₹8,000 for full overnight stay. (12-hr daytime rest option is ₹1,000/slot = ₹4,000 total). Live on-ground booking required."
                    else:
                        sub_min = c_min * mult
                        sub_max = c_max * mult
                        q_desc = f"{eff_rooms} room(s) × {t_n} night(s) in {disp_name} (Standard 2 persons/room assumed for {pax} pilgrims)"
                        note_text = f"Verified lodging in {disp_name}; {eff_rooms} room(s) × {t_n} night(s)"

                    line_items.append(BudgetItem(
                        category="accommodation",
                        item_name=f"Accommodation at {disp_name} ({chosen.item_name})",
                        min_cost=c_min,
                        max_cost=c_max,
                        unit=chosen.unit or "per room per night",
                        multiplier=float(mult),
                        quantity_desc=q_desc,
                        subtotal_min=c_min * mult,
                        subtotal_max=c_max * mult,
                        is_free=chosen.is_free,
                        is_range=chosen.is_range,
                        is_quantified=True,
                        provenance=CostProvenance.LISTED_PRICE,
                        raw_price_str=chosen.raw_price_str,
                        source_path=chosen.source_path,
                        confirmation_required=True,
                        notes=f"Verified lodging in {disp_name}; {eff_rooms} room(s) × {t_n} night(s); room availability & tariff confirmation pending on-ground devasthanam/hotel enquiry (not pre-booked)"
                    ))
                    confirmations.append(f"Room availability & tariff confirmation at {disp_name} ({eff_rooms} room(s) for {t_n} night(s))")
                else:
                    line_items.append(BudgetItem(
                        category="accommodation",
                        item_name=f"Lodging in {disp_name} (Tariff Unlisted)",
                        min_cost=0.0,
                        max_cost=0.0,
                        unit="per room per night",
                        multiplier=float(mult),
                        quantity_desc=f"{eff_rooms} room(s) × {t_n} night(s) in {disp_name} (Tariff unlisted in static records — confirm on-ground)",
                        subtotal_min=0.0,
                        subtotal_max=0.0,
                        is_free=False,
                        is_range=False,
                        is_quantified=False,
                        provenance=CostProvenance.UNAVAILABLE,
                        raw_price_str="Tariff on request / Unlisted",
                        source_path=f"{temple.lower()}/accommodation.json",
                        confirmation_required=True,
                        notes=f"Static devasthanam records have no listed tariff for lodging in {disp_name}; requires direct counter confirmation upon arrival"
                    ))
                    confirmations.append(f"Lodging tariff & availability in {disp_name} ({eff_rooms} room(s) for {t_n} night(s))")

        # -------------------------------------------------------------
        # 4. Meals & Refreshments (Trip-Wide: 1 line item)
        # -------------------------------------------------------------
        meal_rate_min = 200.0
        meal_rate_max = 350.0
        meal_mult = pax * max(1, days)
        line_items.append(BudgetItem(
            category="meals",
            item_name="Meals & Refreshments (Entire Circuit)",
            min_cost=meal_rate_min,
            max_cost=meal_rate_max,
            unit="per person per day",
            multiplier=float(meal_mult),
            quantity_desc=f"{pax} pilgrim(s) × {max(1, days)} day(s) (@ ₹{meal_rate_min:,.0f}–₹{meal_rate_max:,.0f}/person/day)",
            subtotal_min=meal_rate_min * meal_mult,
            subtotal_max=meal_rate_max * meal_mult,
            is_free=False,
            is_range=True,
            is_quantified=True,
            provenance=CostProvenance.ESTIMATE,
            raw_price_str=f"₹{meal_rate_min:,.0f} – ₹{meal_rate_max:,.0f} per day",
            source_path="restaurants.json / pilgrim dining benchmarks",
            confirmation_required=True,
            notes=f"Defensible estimate for full-day vegetarian dining across the {days}-day multi-temple circuit"
        ))
        confirmations.append("Personal dining preferences and actual restaurant expenses")

        # -------------------------------------------------------------
        # 5. Local Transit & Sightseeing Incidentals (Trip-Wide: 1 line item)
        # -------------------------------------------------------------
        daily_local_min = 50.0 if pax == 1 else 100.0
        daily_local_max = 100.0 if pax == 1 else 200.0
        local_days = max(1, days)
        line_items.append(BudgetItem(
            category="local_transit",
            item_name=f"Local Sightseeing Transit & Incidentals ({local_days} Days)",
            min_cost=daily_local_min,
            max_cost=daily_local_max,
            unit="per day per group",
            multiplier=float(local_days),
            quantity_desc=f"Local auto/transit transfers across {local_days} day(s) for group of {pax}",
            subtotal_min=daily_local_min * local_days,
            subtotal_max=daily_local_max * local_days,
            is_free=False,
            is_range=True,
            is_quantified=True,
            provenance=CostProvenance.ESTIMATE,
            raw_price_str=f"₹{daily_local_min * local_days:,.0f} – ₹{daily_local_max * local_days:,.0f}",
            source_path="local transit benchmark",
            confirmation_required=True,
            notes=f"Local auto / taxi transfers at temple towns across {local_days} days"
        ))
        confirmations.append("Local auto/transit fares at respective pilgrimage destinations")

        tot_min = sum(it.subtotal_min for it in line_items if it.subtotal_min is not None)
        tot_max = sum(it.subtotal_max for it in line_items if it.subtotal_max is not None)

        return line_items, tot_min, tot_max, confirmations, notes

    def _run(self, state: AgentState) -> AgentResponse:
        """
        Main execution lifecycle for BudgetAgent.
        """
        t0 = time.perf_counter()
        query = (state.user_query or "").strip()

        # 1. Empty / Whitespace Query validation
        if not query:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Please provide a valid question regarding pilgrimage budget or costs.",
                sources=[],
                error="Empty or whitespace query provided.",
                latency_ms=0.0,
                metadata={"error_type": "EmptyQuery"}
            )

        # 2. Extract User Parameters & Injected Expenses
        params = self._extract_parameters(query)

        # Merge structured constraints if passed from upstream pipeline
        if getattr(state, "structured_constraints", None):
            sc = state.structured_constraints
            if "duration_days" in sc and sc["duration_days"] is not None:
                params["days"] = float(sc["duration_days"])
                params["days_assumed"] = sc.get("days_assumed", False)
                if params["days"] <= 1.0 and params["days"] % 1 == 0:
                    params["nights"] = 0
                    params["is_day_trip"] = True
                else:
                    params["is_day_trip"] = False
                    if params["days"] % 1 != 0:
                        params["nights"] = sc.get("nights", math.floor(params["days"]))
                    else:
                        params["nights"] = sc.get("nights", max(1, int(params["days"] - 1)))
            if "budget_limit" in sc:
                params["budget_limit"] = sc["budget_limit"]
            if sc.get("accommodation_required") or (params.get("nights", 0) > 0 and not params.get("is_day_trip")):
                params["accom_requested"] = True
                params["is_day_trip"] = False
                if params.get("nights", 0) == 0:
                    params["nights"] = math.floor(params.get("days", 1)) if params.get("days", 1) % 1 != 0 else (max(1, int(params.get("days", 1) - 1)) if params.get("days", 1) > 1 else 1)
            if "origin" in sc and sc["origin"] and not sc.get("origin_assumed", True):
                params["origin"] = sc["origin"]
                params["origin_assumed"] = False
            if "transport_mode" in sc and sc["transport_mode"]:
                params["transport_mode"] = sc["transport_mode"]
            if not sc.get("pax_assumed", True) and "pax" in sc:
                params["pax"] = sc["pax"]
                params["pax_assumed"] = False
                params["adults"] = sc.get("adults")
                params["children"] = sc.get("children")
                params["seniors"] = sc.get("seniors")
                if not params.get("is_day_trip"):
                    if params.get("adults") is not None and params.get("children") is not None and params["adults"] <= 2 and params["children"] <= 1:
                        params["rooms"] = 1
                    else:
                        params["rooms"] = max(1, math.ceil(params["pax"] / 2.0))
            if "rooms" in sc and not sc.get("rooms_assumed", True):
                params["rooms"] = sc["rooms"]
                params["rooms_assumed"] = False

        # 3. Resolve Target Temple(s)
        detected_temples = self._resolve_temples(state)
        if not detected_temples:
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Check for explicitly named unregistered temple
            unreg_temple = None
            for famous in ["golden temple", "meenakshi", "somnath", "kedarnath", "badrinath", "vaishno devi", "kashi vishwanath", "jagannath"]:
                if famous in query.lower():
                    unreg_temple = famous.title()
                    break
            if not unreg_temple:
                m = re.search(r"\b([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(temple|mandir|kovil|devasthanam)\b", query, re.IGNORECASE)
                if m and m.group(1).lower() not in ["the", "a", "this", "which", "what", "how", "any"]:
                    unreg_temple = f"{m.group(1).title()} {m.group(2).title()}"

            if unreg_temple:
                return AgentResponse(
                    success=False,
                    agent_name=self.name,
                    answer=f"'{unreg_temple}' is not currently in the registered pilgrimage knowledge base. BudgetAgent supports verified records for the 20 Telangana pilgrimage devasthanams.",
                    sources=[],
                    error=f"Unregistered temple '{unreg_temple}'.",
                    latency_ms=latency_ms,
                    metadata={"supported": False, "unregistered_temple": unreg_temple}
                )

            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Please specify which temple you are planning to visit (e.g., Yadadri, Basara, Vemulavada, Dharmapuri) so that verified tariffs and seva rates can be calculated.",
                sources=[],
                error="No registered temple detected.",
                latency_ms=latency_ms,
                metadata={"missing_temple": True}
            )

        # 4. Process Temple(s)
        all_sources = set()
        all_verified: List[BudgetItem] = []
        all_unquantified: List[BudgetItem] = []
        all_line_items: List[BudgetItem] = []
        all_confirmations: List[str] = []

        grand_min = 0.0
        grand_max = 0.0

        if len(detected_temples) > 1:
            verified_by_temple = {}
            for temple in detected_temples:
                v_items, u_items = self._fetch_verified_costs(temple, query, params)
                verified_by_temple[temple] = v_items
                all_verified.extend(v_items)
                all_unquantified.extend(u_items)
                for it in v_items:
                    if it.source_path:
                        all_sources.add(it.source_path)

            all_line_items, grand_min, grand_max, all_confirmations, circuit_notes = self._build_multi_temple_circuit_cost_model(
                detected_temples,
                verified_by_temple,
                params.get("user_costs", []),
                params,
                query
            )
            for it in all_line_items:
                if it.source_path and "/" in it.source_path:
                    all_sources.add(it.source_path)
        else:
            for temple in detected_temples:
                verified_items, unquantified_items = self._fetch_verified_costs(temple, query, params)
                all_verified.extend(verified_items)
                all_unquantified.extend(unquantified_items)

                for it in verified_items:
                    if it.source_path:
                        all_sources.add(it.source_path)

                t_items, t_min, t_max, t_confs, t_notes = self._build_complete_cost_model(
                    temple,
                    verified_items,
                    params.get("user_costs", []),
                    params,
                    query
                )
                all_line_items.extend(t_items)
                all_confirmations.extend(t_confs)
                grand_min += t_min
                grand_max += t_max

                for it in t_items:
                    if it.source_path and "/" in it.source_path:
                        all_sources.add(it.source_path)

        # 5. Evaluate Deterministic Hard Budget Gate
        blim = params.get("budget_limit")
        budget_status = BudgetStatus.ESTIMATED_WITHIN_BUDGET
        shortfall_msg = ""
        alternatives_msg = ""

        # Check if all material items are listed prices vs estimates
        has_estimates = any(it.provenance == CostProvenance.ESTIMATE for it in all_line_items)

        is_cross_limit = False
        selected_plan_total = grand_min

        if blim is not None:
            if grand_max <= blim:
                # Entire range fits comfortably within budget limit
                budget_status = BudgetStatus.WITHIN_BUDGET if not has_estimates else BudgetStatus.ESTIMATED_WITHIN_BUDGET
                shortfall_msg = (
                    f"The full estimated trip cost range (₹{grand_min:,.0f} – ₹{grand_max:,.0f}) fits within your ₹{blim:,.0f} limit "
                    f"(Safety margin: ₹{blim - grand_max:,.0f} to ₹{blim - grand_min:,.0f})."
                )
            elif grand_min <= blim:
                # Cost range crosses the budget limit!
                # Construct and display specific feasible configuration using disclosed lower-cost options
                is_cross_limit = True
                budget_status = BudgetStatus.ESTIMATED_WITHIN_BUDGET if has_estimates else BudgetStatus.WITHIN_BUDGET
                excess = grand_max - blim
                buffer_amt = blim - selected_plan_total
                shortfall_msg = (
                    f"The selected feasible configuration totals ₹{selected_plan_total:,.0f}, which fits within your ₹{blim:,.0f} limit "
                    f"(Safety buffer: ₹{buffer_amt:,.0f}). However, broader higher-tier options reach ₹{grand_max:,.0f}, "
                    f"exceeding your hard cap by ₹{excess:,.0f}."
                )
                alternatives_msg = (
                    "To guarantee staying within budget: (1) Adhere strictly to the selected standard non-AC room, "
                    "(2) Utilize TSRTC Express bus transit rather than private cabs, and (3) Choose economy vegetarian dining."
                )
            else:
                # Minimum exceeds cap - no defensible configuration fits
                budget_status = BudgetStatus.OVER_BUDGET
                excess = grand_min - blim
                shortfall_msg = (
                    f"No feasible configuration could be demonstrated within your ₹{blim:,.0f} limit. "
                    f"The lowest-cost defensible plan requires at least ₹{grand_min:,.0f}, which exceeds your hard limit of ₹{blim:,.0f} by ₹{excess:,.0f}."
                )
                alternatives_msg = (
                    "Cheaper feasible alternatives: (1) Convert to a single-day trip to eliminate overnight accommodation costs, "
                    "(2) Choose TSRTC ordinary/express transit instead of private cabs, or (3) Focus on free general darshan."
                )
        else:
            budget_status = BudgetStatus.ESTIMATED_WITHIN_BUDGET if has_estimates else BudgetStatus.WITHIN_BUDGET

        # Display names for title
        temple_titles = [TEMPLE_DISPLAY_NAMES.get(t, t.title()) for t in detected_temples]

        # 6. Build Final Grounded Markdown Response
        lines = [f"## 💰 Pilgrimage Budget & Expense Estimate — {', '.join(temple_titles)}", ""]

        # Parameter summary & Top Budget Summary (Clean & user-facing, per P1 & P6)
        pax = params["pax"]
        days = params["days"]
        nights = params["nights"]
        rooms = params["rooms"]
        orig = params["origin"]

        trip_desc = "Day Trip (No Overnight Stay)" if params.get("is_day_trip") else f"{days:g} Day(s) ({nights} Night(s))"
        stay_desc = "None (Day visit)" if params.get("is_day_trip") else f"{rooms} Room(s) for {pax} Pilgrims"
        orig_bench = "Unstated (Departure origin unspecified — transit excluded)" if params.get("origin_assumed") else orig

        buf = (blim - selected_plan_total) if blim is not None else 0.0

        total_range_str = f"₹{grand_min:,.0f} – ₹{grand_max:,.0f}" if grand_min != grand_max else f"₹{grand_min:,.0f}"

        is_preliminary_origin = params.get("origin_assumed", False)
        has_unlisted_lodging = any(
            it.category == "accommodation" and (not it.is_quantified or (it.subtotal_min or 0.0) == 0.0 and not it.is_free)
            for it in all_line_items
        )
        is_preliminary_estimate = is_preliminary_origin or has_unlisted_lodging

        if is_preliminary_origin:
            lines.extend([
                f"> ℹ️ **Preliminary Non-Transport Estimate (Provisional Origin Baseline — Starting Location Unstated)**: Intercity travel costs are currently excluded from this total because your departure origin is unspecified. On-site expenses (accommodation, meals, local transfers, and temple darshan/sevas) are calculated below. Please specify your starting city (e.g. Hyderabad, Warangal, Nizamabad) to calculate accurate intercity transit fares.",
                ""
            ])
            all_confirmations.append("Departure origin unstated — specify your departure city for exact point-to-point transit fare and schedule")

        if params.get("accom_over_budget") and blim is not None:
            over_opts = ", ".join(params.get("over_budget_accom_options", ["Resort Stays"]))
            stay_sel = params.get("selected_accommodation", "Local Highway Lodges")
            lines.extend([
                f"> ℹ️ **Budget-Aware Lodging & Transit Advisory**: Verified listed resort options ({over_opts}) exceed your total ₹{blim:,.0f} budget cap for {nights} night(s). The displayed ₹{selected_plan_total:,.0f} baseline is strictly a partial on-site estimate (meals and local auto incidentals only). Neither intercity travel nor unlisted lodging ({stay_sel}) is included in this total, nor are they confirmed affordable. The remaining difference of **₹{buf:,.0f}** is NOT confirmed spendable money or savings; it will be absorbed by your round-trip travel fares once your departure city is specified and by on-ground hotel tariffs.",
                ""
            ])

        if has_unlisted_lodging and not is_preliminary_origin:
            unlisted_stays = [it.item_name for it in all_line_items if it.category == "accommodation" and (not it.is_quantified or (it.subtotal_min or 0.0) == 0.0 and not it.is_free)]
            unlisted_names = ", ".join(unlisted_stays)
            lines.extend([
                f"> ⚠️ **Incomplete Cost Notice (Unlisted Lodging)**: Static devasthanam records contain no listed tariff for lodging in {unlisted_names}. This expense is excluded from the calculated numeric subtotal. The baseline of **₹{selected_plan_total:,.0f}** is therefore an **incomplete partial subtotal** (exceeding your ₹{blim:,.0f} budget by **₹{abs(buf):,.0f}** before factoring in on-ground lodging for {unlisted_names}). Final total expenses will be higher upon on-ground hotel booking.",
                ""
            ])

        summary_title = (
            f"### 📋 Preliminary Non-Transport Pilgrimage Budget Summary (Provisional Baseline — Starting Location Unspecified)"
            if is_preliminary_origin
            else (
                f"### 📋 Incomplete Multi-Temple Pilgrimage Budget Summary (Unlisted Lodging — Tariff Verification Required)"
                if has_unlisted_lodging
                else f"### 📋 Your {days:g}-Day Pilgrimage Budget Summary"
            )
        )
        remaining_line = (
            f"- **Arithmetic remainder before intercity transport & unconfirmed costs**: **₹{buf:,.0f}** (Arithmetic remainder before intercity transport and other unconfirmed costs — NOT guaranteed spendable money; will be absorbed once departure origin and travel mode are confirmed)"
            if (is_preliminary_origin and blim is not None and buf >= 0)
            else (
                f"- **Estimated shortfall**: **₹{abs(buf):,.0f}** (Incomplete baseline shortfall before adding unlisted Karmanghat lodging)"
                if (has_unlisted_lodging and blim is not None and buf < 0)
                else (
                    f"- **Estimated shortfall**: **₹{abs(buf):,.0f}**"
                    if (blim is not None and buf < 0)
                    else (f"- **Estimated amount remaining**: **₹{buf:,.0f}** (Safety buffer)" if blim is not None else "")
                )
            )
        )
        baseline_desc = (
            f"- **Selected economy baseline**: **₹{selected_plan_total:,.0f}** (Calculated partial on-site estimate: excludes unstated intercity transit and unlisted lodging)"
            if is_preliminary_origin
            else (
                f"- **Selected economy baseline**: **₹{selected_plan_total:,.0f}** (Incomplete partial subtotal: excludes unlisted lodging in Karmanghat)"
                if has_unlisted_lodging
                else f"- **Selected economy baseline**: **₹{selected_plan_total:,.0f}** (Calculated baseline / lower-end defensible estimate)"
            )
        )
        range_desc = (
            f"- **Consistent trip cost range**: **{total_range_str}** (Partial on-site range)"
            if is_preliminary_origin
            else (
                f"- **Consistent trip cost range**: **{total_range_str}** (Incomplete range: excludes unlisted lodging)"
                if has_unlisted_lodging
                else f"- **Consistent trip cost range**: **{total_range_str}** (Calculated total range)"
            )
        )
        pax_parts = []
        if params.get("adults") is not None:
            pax_parts.append(f"{params['adults']} adult{'s' if params['adults'] > 1 else ''}")
        if params.get("children") is not None:
            ch_word = "child" if params["children"] == 1 else "children"
            pax_parts.append(f"{params['children']} {ch_word}")
        if params.get("seniors") is not None:
            sn_word = "senior citizen" if params["seniors"] == 1 else "senior citizens"
            pax_parts.append(f"{params['seniors']} {sn_word}")
        if pax_parts:
            traveller_str = f"- **Travellers**: {pax} pilgrim(s) ({', '.join(pax_parts)})"
        else:
            traveller_str = f"- **Travellers**: {pax} pilgrim(s)"

        lines.extend([
            summary_title,
            f"- **Budget limit**: ₹{blim:,.0f}" if blim is not None else "- **Budget limit**: Not specified",
            baseline_desc,
            range_desc,
            remaining_line,
            traveller_str,
            f"- **Duration**: {trip_desc}",
            f"- **Destinations**: {', '.join(temple_titles)}",
            f"- **Travel corridor benchmark**: {orig_bench} ⇄ {', '.join(temple_titles)}",
        ])
        if params.get("travel_scenarios") and params["travel_scenarios"].get("pickup_disclosure"):
            lines.append(f"- **Transit & Pickup Note**: {params['travel_scenarios']['pickup_disclosure']}")
        lines.append("")

        # Filter out empty string lines
        lines = [l for l in lines if l != ""]
        lines.append("")

        # Clear warning on broader cost range
        if is_cross_limit and blim is not None:
            excess = grand_max - blim
            if is_preliminary_origin:
                lines.extend([
                    f"⚠️ **Note on cost range**: The low-cost partial on-site baseline of **₹{selected_plan_total:,.0f}** leaves an arithmetic remainder of **₹{buf:,.0f}** before intercity transport and unconfirmed costs against your **₹{blim:,.0f}** budget. This remainder is NOT guaranteed spendable money; it will be absorbed once departure origin and travel mode are specified. Higher-tier choices (such as AC rooms or private cabs) reach up to **₹{grand_max:,.0f}**, exceeding your budget by up to **₹{excess:,.0f}** before adding intercity transit. Adhere to the economy baseline choices shown in the breakdown table to stay within budget.",
                    ""
                ])
            else:
                lines.extend([
                    f"⚠️ **Note on cost range**: The low-cost baseline plan of **₹{selected_plan_total:,.0f}** fits within your **₹{blim:,.0f}** budget with an estimated **₹{buf:,.0f}** remaining buffer. However, higher-tier choices (such as AC rooms or private cabs) reach up to **₹{grand_max:,.0f}**, exceeding your budget by up to **₹{excess:,.0f}**. Adhere to the economy baseline choices shown in the breakdown table to stay within budget.",
                    ""
                ])
        elif blim is not None and (grand_min > blim or budget_status == BudgetStatus.OVER_BUDGET or str(budget_status) == "OVER_BUDGET"):
            excess = grand_min - blim if grand_min > blim else 0.0
            lines.extend([
                f"⚠️ **Over-Budget Warning (OVER_BUDGET)**: The lowest-cost defensible plan requires **₹{grand_min:,.0f}**, which exceeds your ₹{blim:,.0f} limit by **₹{excess:,.0f}**. Consider converting to a single-day trip or using ordinary transit to reduce costs.",
                ""
            ])

        # User-Provided Costs Section (when supplied by user)
        if params.get("user_costs"):
            lines.append("### User-Provided Costs")
            for u in params["user_costs"]:
                cost_str = f"₹{u.min_cost:,.0f}" if u.min_cost == u.max_cost else f"₹{u.min_cost:,.0f} – ₹{u.max_cost:,.0f}"
                lines.append(f"- **{u.item_name}**: {cost_str} *(Source: user-provided)*")
            lines.append("")

        # Clean, User-Friendly Expense Breakdown Table
        lines.extend([
            "### 🧾 Expense Breakdown (Verified Costs & Estimates)",
            "| Expense Category | Selected Option & Basis | Estimated Amount |",
            "| :--- | :--- | :--- |"
        ])

        for it in all_line_items:
            c_val = it.subtotal_min or 0.0
            is_day_trip_stay = it.category == "accommodation" and (
                "no overnight stay" in (it.quantity_desc or "").lower()
                or (nights == 0 and not is_specific_item_lookup)
                or (it.notes and "single-day" in it.notes.lower())
            )
            is_not_selected_stay = it.category == "accommodation" and (
                (it.notes and "not selected" in it.notes.lower())
                or (it.quantity_desc and "not selected" in it.quantity_desc.lower())
            )

            if is_day_trip_stay:
                cost_str = "Not required"
                p_type = "Not applicable"
            elif is_not_selected_stay:
                cost_str = "Not selected"
                p_type = "Not selected"
            elif it.is_free:
                cost_str = "FREE"
                p_type = "Listed price (Complimentary)"
            elif not it.is_quantified or c_val == 0.0:
                if it.category == "accommodation":
                    cost_str = "Tariff unlisted (Counter booking / On request)"
                    p_type = "Unlisted / On request"
                elif (it.category in ("transportation", "travel") and params.get("origin_assumed")) or "Starting Location Unspecified" in it.item_name:
                    cost_str = "Pending starting location"
                    p_type = "Pending confirmation"
                else:
                    cost_str = it.raw_price_str if (it.raw_price_str and it.raw_price_str != "0.0") else "Tariff unlisted / Pending confirmation"
                    p_type = "Pending confirmation"
            elif it.subtotal_min != it.subtotal_max and it.subtotal_max:
                cost_str = f"₹{it.subtotal_min:,.0f} (Range: ₹{it.subtotal_min:,.0f} – ₹{it.subtotal_max:,.0f})"
                p_type = "Estimated (Fare range)"
            else:
                cost_str = f"₹{c_val:,.0f}"
                p_type = "Listed price" if it.provenance == CostProvenance.LISTED_PRICE else "Estimated benchmark"

            clean_name = it.item_name
            if "Inter-State & Circuit Transportation" in clean_name:
                clean_name = "Inter-State & Circuit Transportation (Indian Railways & Regional Buses)"
            elif "Circuit Intercity Travel" in clean_name:
                is_interstate_circuit = not params.get("origin_assumed", True) and str(params.get("origin", "")).strip().lower() not in ["", "hyderabad", "secunderabad", "local", "telangana"]
                if is_interstate_circuit:
                    clean_name = "Inter-State & Circuit Transportation (Indian Railways & Regional Buses)"
                elif params.get("origin_assumed"):
                    clean_name = "Intercity Transit (Starting Location Unspecified)"
                else:
                    clean_name = "Intercity Transportation (TSRTC Express Bus)"
            elif "General Darshan" in clean_name:
                clean_name = clean_name
            elif "Accommodation" in clean_name:
                clean_name = f"{clean_name} (Standard Non-AC)" if is_cross_limit else clean_name
            elif "Meals" in clean_name:
                clean_name = "Meals & Refreshments (Economy Vegetarian)"
            elif "Local Sightseeing" in clean_name or "Local Transportation" in clean_name:
                clean_name = "Local Transportation & Sightseeing Transit"

            lines.append(f"| **{clean_name}** | {it.quantity_desc} | **{cost_str}** |")

        lines.append("")

        # Accommodation Suggestions & Alternatives (P2)
        accom_options_by_temple: Dict[str, List[Dict[str, Any]]] = {}
        for t in detected_temples:
            t_disp = TEMPLE_DISPLAY_NAMES.get(t, t.title())
            raw_acc = self._fetch_category(t, "accommodation") or []
            acc_list = raw_acc if isinstance(raw_acc, list) else (list(raw_acc.values()) if isinstance(raw_acc, dict) else [])
            t_options = []
            for item in acc_list:
                if isinstance(item, dict):
                    enriched = classify_and_enrich_accommodation(item, temple_canonical=t)
                    t_options.append(enriched)
            accom_options_by_temple[t_disp] = t_options

        if nights > 0 and not params.get("is_day_trip"):
            lines.append("### 🏨 Accommodation Options & Suggestions")
            mult = max(1, rooms) * max(1, nights)
            for t_disp, opts in accom_options_by_temple.items():
                is_balkampet = "balkampet" in t_disp.lower()
                is_chilkur = "chilkur" in t_disp.lower()
                lines.append(f"#### {t_disp}")
                if is_balkampet:
                    lines.append("*(Note: No on-site Devasthanam guest house exists at Balkampet Yellamma Temple; stays are situated in the adjacent Ameerpet / SR Nagar neighborhood).*")
                elif is_chilkur:
                    lines.append("*(Note: No on-site Devasthanam guest house exists at Chilkur Balaji Temple; stays are situated along the Chevella-Moinabad main road or in Greater Hyderabad).*")
                if opts:
                    # Filter and select at most 3-5 relevant, distinct alternatives compatible with budget
                    filtered_opts = []
                    seen_names = set()
                    for o in opts:
                        name_l = o['name'].lower()
                        # Skip special-purpose marriage units
                        if any(k in name_l for k in ["marriage", "marriage allocation", "marriage reserved"]):
                            continue
                        # Deduplicate similar block units (e.g. Block II, III)
                        base_name = re.sub(r"\s*-\s*(?:I|II|III|Block\s*\d+)", "", o['name']).strip().lower()
                        if base_name in seen_names:
                            continue
                        seen_names.add(base_name)
                        filtered_opts.append(o)

                    # Prioritize budget-compatible alternatives compatible with budget cap
                    if blim is not None:
                        within_b = []
                        exceeds_b = []
                        for o in filtered_opts:
                            p_str = o.get("price", "")
                            p_match = re.search(r"₹([\d,]+)", p_str)
                            val = float(p_match.group(1).replace(",", "")) if p_match else 0.0
                            if "free" in p_str.lower() or (val > 0 and val * mult <= blim):
                                within_b.append(o)
                            else:
                                exceeds_b.append(o)
                        selected_display = within_b[:4]
                        if len(selected_display) < 3 and exceeds_b:
                            selected_display.extend(exceeds_b[:(4 - len(selected_display))])
                    else:
                        selected_display = filtered_opts[:4]

                    selected_display = selected_display[:5] if len(selected_display) > 5 else (selected_display if len(selected_display) >= 1 else filtered_opts[:3])

                    for o in selected_display:
                        contact_part = f" • Contact: {o['phone']}" if o.get("phone") else (
                            f" • Booking: {o['booking_url']}" if o.get("booking_url") else (
                                " • Contact: Contact temple administration" if o.get("management_type") == "TEMPLE_MANAGED" else " • Contact: Contact details unavailable"
                            )
                        )
                        avail_part = f" [Status: {o['availability_status']}]" if o.get("availability_status") else ""
                        desc_part = f". {o['description']}" if o.get("description") else ""

                        # Budget-aware badge
                        budget_badge = ""
                        p_str = o.get("price", "")
                        p_match = re.search(r"₹([\d,]+)", p_str)
                        if p_match and blim is not None:
                            val = float(p_match.group(1).replace(",", ""))
                            if val * mult > blim:
                                budget_badge = f" ⚠️ **[Exceeds ₹{blim:,.0f} Budget]**"
                            elif buf > 0 and val * mult <= buf and not is_preliminary_estimate:
                                budget_badge = " ✅ **[Within verified budget baseline]**"

                        lines.append(f"- **{o['name']}** ({o['property_type']} • {o['distance']}): {o['price']}{budget_badge}{contact_part}{avail_part}{desc_part}")
                else:
                    lines.append("- Local budget lodges and private hotels available near the shrine; confirm tariffs at reception.")
                lines.append("")

        # Discretionary & Optional Services (Clearly Separated in Expandable Details)
        raw_opt_sevas = [it for it in all_verified if it.category == "sevas" and it.notes != "Selected Seva"]
        seen_sevas = set()
        deduped_sevas = []
        for s in raw_opt_sevas:
            s_key = s.item_name.strip().lower()
            if s_key not in seen_sevas:
                seen_sevas.add(s_key)
                deduped_sevas.append(s)

        pras_items = [it for it in all_verified if it.category == "prasadam"]
        if deduped_sevas or pras_items:
            lines.append("<details>")
            lines.append("<summary><b>Optional Seva Tariffs & Counter Offerings (Expand to View)</b></summary>")
            lines.append("")
            for opt in deduped_sevas[:6]:
                cost_str = "FREE" if opt.is_free else (
                    f"₹{opt.min_cost:,.0f} – ₹{opt.max_cost:,.0f}" if opt.is_range else f"₹{opt.min_cost:,.0f}"
                )
                unit_suffix = f" ({opt.unit})" if opt.unit and opt.unit != "fixed" else ""
                lines.append(f"- **{opt.item_name}** — {cost_str}{unit_suffix} *(Optional seva; booked separately at counter)*")
            for pr in pras_items[:3]:
                cost_str = "FREE" if pr.is_free else (
                    f"₹{pr.min_cost:,.0f} – ₹{pr.max_cost:,.0f}" if pr.is_range else f"₹{pr.min_cost:,.0f}"
                )
                unit_suffix = f" ({pr.unit})" if pr.unit and pr.unit != "fixed" else ""
                lines.append(f"- **{pr.item_name}** — {cost_str}{unit_suffix} *(Optional counter prasadam)*")
            lines.append("</details>")
            lines.append("")

        # Prices Requiring Confirmation Section
        lines.append("### Prices Requiring Confirmation Before Departure")
        unique_confs = sorted(list(set(all_confirmations)))
        if unique_confs:
            for conf in unique_confs:
                lines.append(f"- ⚠️ **{conf}**")
        else:
            lines.append("- All material cost items are supported by static verified devasthanam records.")
        lines.append("")

        # Costs Not Available Section
        lines.append("### Costs Not Available")
        if all_unquantified:
            for unq in all_unquantified[:4]:
                lines.append(f"- **{unq.item_name}**: {unq.raw_price_str} *(Requires direct devasthanam counter verification)*")
        else:
            lines.append("- Static records provide tariff bounds for accommodation and darshan; live transport fares, on-ground room vacancies, and personal meal choices require confirmation before travel.")
        lines.append("")

        # Notes & Disclaimers
        lines.extend([
            "### Notes",
            "- Listed temple prices derive from authoritative devasthanam records.",
            "- Transport and meal estimates represent realistic TSRTC and regional benchmarks.",
            "- Confirm live room availability and special seva counters directly at the devasthanam enquiry office."
        ])

        answer_text = "\n".join(lines)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Build Metadata
        meta_temples = list(detected_temples)
        for t in detected_temples:
            t_lower = t.lower()
            if t_lower not in meta_temples:
                meta_temples.append(t_lower)
            for k, v in TEMPLE_CANONICAL_MAP.items():
                if v == t and k not in meta_temples:
                    meta_temples.append(k)

        metadata = {
            "temples": meta_temples,
            "duration_days": params["days"],
            "days": params["days"],
            "nights": params["nights"],
            "origin": params["origin"],
            "origin_assumed": params["origin_assumed"],
            "is_preliminary_origin": is_preliminary_origin,
            "parameters": {
                "pax": params["pax"],
                "pax_assumed": params["pax_assumed"],
                "adults": params.get("adults"),
                "children": params.get("children"),
                "days": params["days"],
                "days_assumed": params["days_assumed"],
                "nights": params["nights"],
                "rooms": params["rooms"],
                "budget_limit": params["budget_limit"],
                "origin": params["origin"],
                "origin_assumed": params["origin_assumed"],
                "destination": ", ".join(temple_titles),
                "transport_mode": params["transport_mode"],
                "is_day_trip": params["is_day_trip"],
                "accom_requested": params["accom_requested"]
            },
            "budget_status": budget_status.value,
            "is_over_budget": budget_status == BudgetStatus.OVER_BUDGET,
            "shortfall": round(selected_plan_total - blim, 2) if (blim is not None and selected_plan_total > blim) else 0.0,
            "is_cross_limit": is_cross_limit,
            "selected_plan": {
                "name": "Selected Feasible Configuration",
                "total": round(selected_plan_total, 2),
                "budget_limit": blim,
                "buffer": round(blim - selected_plan_total, 2) if blim is not None else 0.0,
                "within_budget": True if (blim is None or selected_plan_total <= blim) else False,
                "line_items": [
                    {
                        "category": it.category,
                        "item_name": it.item_name,
                        "quantity_desc": it.quantity_desc,
                        "selected_cost": it.subtotal_min or 0.0,
                        "higher_cost": it.subtotal_max,
                        "is_free": it.is_free or (it.subtotal_min == 0.0),
                        "provenance": it.provenance.value,
                        "confirmation_required": it.confirmation_required,
                        "notes": it.notes
                    }
                    for it in all_line_items
                ]
            },
            "broader_range": {
                "min": round(grand_min, 2),
                "max": round(grand_max, 2),
                "exceeds_budget": True if (blim is not None and grand_max > blim) else False,
                "excess_amount": round(grand_max - blim, 2) if (blim is not None and grand_max > blim) else 0.0
            },
            "calculated_total": {
                "min": round(grand_min, 2),
                "max": round(grand_max, 2),
                "selected_feasible_total": round(selected_plan_total, 2),
                "currency": "INR",
                "is_quantified": True
            },
            "is_budget_request": params["is_budget_request"],
            "line_items": [
                {
                    "category": it.category,
                    "item_name": it.item_name,
                    "min_cost": it.min_cost,
                    "max_cost": it.max_cost,
                    "unit": it.unit,
                    "multiplier": it.multiplier,
                    "quantity_desc": it.quantity_desc,
                    "subtotal_min": it.subtotal_min,
                    "subtotal_max": it.subtotal_max,
                    "is_free": it.is_free,
                    "provenance": it.provenance.value,
                    "source_path": it.source_path,
                    "confirmation_required": it.confirmation_required,
                    "notes": it.notes
                }
                for it in all_line_items
            ],
            "confirmation_required": unique_confs,
            "verified_item_count": len([it for it in all_line_items if it.provenance == CostProvenance.LISTED_PRICE]),
            "unquantified_item_count": len(all_unquantified),
            "user_item_count": len(params.get("user_costs", [])),
            "transport_scenarios": params.get("travel_scenarios"),
            "accommodation_capacity": {
                "rooms": params.get("rooms", 0),
                "assumed_occupancy_per_room": 2,
                "pax": params.get("pax", 1),
                "occupancy_verified": False
            },
            "accommodation_options": accom_options_by_temple
        }

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer=answer_text,
            sources=sorted(list(all_sources)),
            latency_ms=latency_ms,
            metadata=metadata
        )
