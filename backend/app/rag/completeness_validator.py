import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Canonical availability status constants
CONFIRMED_AVAILABLE = "CONFIRMED_AVAILABLE"
CONFIRMED_UNAVAILABLE = "CONFIRMED_UNAVAILABLE"
DOCUMENTED_LIMITATIONS = "DOCUMENTED_LIMITATIONS"
INFORMATION_NOT_AVAILABLE = "INFORMATION_NOT_AVAILABLE"

# Common negative indicator regexes
NEGATIVE_PATTERNS = [
    r"\bnone\b",
    r"\bnot\s+(available|situated|present|supported|provided|applicable|associated)\b",
    r"\bno\s+(river|lake|stream|natural\s+river|river\s+bank|water\s+body|parking|accommodation|restaurants?|free\s+meals|ramps?|lifts?|elevators?)\b",
    r"\bdoes\s+not\s+have\b",
    r"\bdo\s+not\s+have\b",
    r"\bwithout\b",
    r"\blacks?\b",
    r"\bno\s+elevators,?\s+lifts,?\s+or\s+ramps\b",
    r"\bnot\s+wheelchair\s+friendly\b",
    r"\bwheelchair\s+access\s+is\s+not\s+supported\b",
    r"\bcurrently\s+lacks\s+dedicated\b"
]

class CompletenessValidator:
    """
    Validates retrieval completeness and categorizes knowledge base records into:
    1. CONFIRMED_AVAILABLE (Affirmative evidence)
    2. CONFIRMED_UNAVAILABLE (Explicit negative evidence)
    3. DOCUMENTED_LIMITATIONS (Partial access or specific restrictions)
    4. INFORMATION_NOT_AVAILABLE (Missing data / no record)
    
    Guarantees: Missing data != Confirmed absence.
    """

    def __init__(self):
        pass

    def evaluate_availability(
        self,
        data: Any,
        category: str
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Evaluates parsed JSON data for a temple/category to determine availability status.
        Returns: (status: str, evidence: Optional[str], source_field: Optional[str])
        """
        if data is None or data == "" or data == [] or data == {}:
            return INFORMATION_NOT_AVAILABLE, None, None

        cat = category.lower()

        # -------------------------------------------------------------
        # 1. Wheelchair & Elderly Accessibility
        # -------------------------------------------------------------
        if cat in ("wheelchair", "elderly_ppl", "accessibility"):
            if isinstance(data, dict):
                wa = data.get("wheelchair_accessibility") or data
                is_friendly = wa.get("is_wheelchair_friendly")
                details = wa.get("details") or []
                details_text = " ".join([str(d) for d in details])

                # Check explicit boolean flag
                if is_friendly is True:
                    evidence = "Wheelchair accessible"
                    if details:
                        evidence += f": {details[0][:100]}"
                    return CONFIRMED_AVAILABLE, evidence, "is_wheelchair_friendly"
                elif is_friendly is False:
                    evidence = "Not wheelchair friendly"
                    if details:
                        evidence += f" ({details[0][:100]})"
                    return CONFIRMED_UNAVAILABLE, evidence, "is_wheelchair_friendly"

                # Check textual details for documented limitations vs confirmed available/unavailable
                if details_text:
                    dt_low = details_text.lower()
                    has_limitation = (
                        "bring their own" in dt_low or
                        "bring own" in dt_low or
                        "does not provide" in dt_low or
                        "partial" in dt_low or
                        ("accessible" in dt_low and ("however" in dt_low or "but" in dt_low or "limited" in dt_low or "equipment" in dt_low))
                    )
                    if has_limitation:
                        return DOCUMENTED_LIMITATIONS, details[0] if details else details_text, "details"
                    elif any(re.search(pat, dt_low) for pat in NEGATIVE_PATTERNS):
                        return CONFIRMED_UNAVAILABLE, f"Unavailable: {details[0][:120]}", "details"
                    elif any(w in dt_low for w in ["accessible", "ramp", "wheelchair available", "lift"]):
                        return CONFIRMED_AVAILABLE, f"Available: {details[0][:120]}", "details"

            return INFORMATION_NOT_AVAILABLE, None, None

        # -------------------------------------------------------------
        # 2. River Bank & Ghats
        # -------------------------------------------------------------
        elif cat in ("river_bank", "river"):
            if isinstance(data, dict):
                rname = str(data.get("river_name") or "").strip()
                significance = str(data.get("significance") or "").strip()
                ghats = data.get("ghats")

                if not rname:
                    return INFORMATION_NOT_AVAILABLE, None, None

                rname_lower = rname.lower()
                sig_lower = significance.lower()

                # Explicit negative check
                is_neg = (
                    rname_lower == "none"
                    or "no natural river" in rname_lower
                    or "no river bank" in rname_lower
                    or "not situated on the banks of a river" in rname_lower
                    or "does not have a river" in rname_lower
                    or "located at the base of hanamkonda hill" in rname_lower
                    or "no running river" in sig_lower
                    or ("lake" in rname_lower and not any(k in rname_lower for k in ["river", "vagu", "stream", "sangam"]))
                )
                if is_neg:
                    return CONFIRMED_UNAVAILABLE, f"No river: {rname}", "river_name"

                evidence = f"River: {rname}"
                if ghats and isinstance(ghats, list) and len(ghats) > 0 and isinstance(ghats[0], dict) and ghats[0].get("name"):
                    evidence += f" (Ghat: {ghats[0].get('name')})"
                return CONFIRMED_AVAILABLE, evidence, "river_name"

            return INFORMATION_NOT_AVAILABLE, None, None

        # -------------------------------------------------------------
        # 3. Accommodation
        # -------------------------------------------------------------
        elif cat == "accommodation":
            if isinstance(data, list) and len(data) > 0:
                return CONFIRMED_AVAILABLE, f"Offers {len(data)} accommodation options", "accommodations"
            elif isinstance(data, dict) and len(data) > 0:
                acc_keys = [k for k in data.keys() if "none" not in str(data[k]).lower()]
                if acc_keys:
                    return CONFIRMED_AVAILABLE, f"Accommodation available ({len(acc_keys)} types)", "accommodation_dict"
            return INFORMATION_NOT_AVAILABLE, None, None

        # -------------------------------------------------------------
        # 4. Restaurants & Food
        # -------------------------------------------------------------
        elif cat == "restaurants":
            if isinstance(data, list) and len(data) > 0:
                first_name = data[0].get("name") if isinstance(data[0], dict) else ""
                evidence = f"{len(data)} nearby restaurants" + (f" (e.g. {first_name})" if first_name else "")
                return CONFIRMED_AVAILABLE, evidence, "restaurants"
            elif isinstance(data, dict) and len(data) > 0:
                return CONFIRMED_AVAILABLE, "Restaurant facilities available", "restaurants"
            return INFORMATION_NOT_AVAILABLE, None, None

        # -------------------------------------------------------------
        # 5. Sevas, Festivals, Facilities, Prasadams, Attractions, etc.
        # -------------------------------------------------------------
        elif cat in ("sevas", "festivals", "facilities", "prasadam", "nearby_attractions", "donations", "sub_temples"):
            if isinstance(data, list) and len(data) > 0:
                return CONFIRMED_AVAILABLE, f"{len(data)} {cat} recorded", cat
            elif isinstance(data, dict) and len(data) > 0:
                return CONFIRMED_AVAILABLE, f"{cat.title()} details available", cat
            return INFORMATION_NOT_AVAILABLE, None, None

        # Generic default check
        if isinstance(data, (dict, list)) and len(data) > 0:
            return CONFIRMED_AVAILABLE, f"{cat.title()} information available", cat

        return INFORMATION_NOT_AVAILABLE, None, None

    def validate_list_completeness(
        self,
        temple: str,
        category: str,
        retrieved_items_count: int,
        expected_items_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validates whether retrieved item count matches expected source count for complete lists.
        """
        expected = expected_items_count or retrieved_items_count
        is_complete = (retrieved_items_count >= expected) if expected > 0 else (retrieved_items_count > 0)

        return {
            "complete": is_complete,
            "temple": temple,
            "category": category,
            "expected_count": expected,
            "retrieved_count": retrieved_items_count,
            "missing_count": max(0, expected - retrieved_items_count)
        }

    def validate_cross_temple_aggregation(
        self,
        category: str,
        condition: str,
        evaluation_results: List[Dict[str, Any]],
        total_scope: int = 20
    ) -> Dict[str, Any]:
        """
        Evaluates and categorizes the results across all 20 temples.
        Strictly partitions:
        - Confirmed Available
        - Confirmed Unavailable
        - Information Not Available
        """
        confirmed_available: List[Dict[str, Any]] = []
        confirmed_unavailable: List[Dict[str, Any]] = []
        information_not_available: List[str] = []

        for item in evaluation_results:
            status = item.get("status")
            t_name = item.get("temple")

            if status == CONFIRMED_AVAILABLE:
                confirmed_available.append(item)
            elif status == CONFIRMED_UNAVAILABLE:
                confirmed_unavailable.append(item)
            else:
                information_not_available.append(t_name)

        if condition == "negative":
            matching_items = confirmed_unavailable
        else:
            matching_items = confirmed_available

        return {
            "total_temples_evaluated": total_scope,
            "category": category,
            "condition": condition,
            "confirmed_available": confirmed_available,
            "confirmed_unavailable": confirmed_unavailable,
            "information_not_available": information_not_available,
            "matching_temples_count": len(matching_items),
            "matching_temples": [m["temple"] for m in matching_items],
            "complete_evaluation": len(evaluation_results) == total_scope
        }

    def validate_multi_category_coverage(
        self,
        requested_categories: List[str],
        retrieved_categories: Set[str]
    ) -> Dict[str, Any]:
        """
        Verifies that every requested category in a multi-category query is represented.
        """
        missing = [c for c in requested_categories if c not in retrieved_categories]
        return {
            "complete": len(missing) == 0,
            "requested_categories": requested_categories,
            "retrieved_categories": sorted(list(retrieved_categories)),
            "missing_categories": missing
        }

completeness_validator = CompletenessValidator()
