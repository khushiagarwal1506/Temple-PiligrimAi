import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from app.rag.category_registry import category_registry, CategoryRegistry

@dataclass
class QueryPlan:
    """
    Structured plan returned by the deterministic QueryPlanner.
    Specifies scope, target temples, categories, intent, condition, exhaustiveness,
    and recommended retrieval mode.
    """
    scope: str            # "single_temple", "cross_temple", "general"
    temples: List[str]    # List of detected canonical temple names (e.g. ["Dharmapuri"])
    categories: List[str] # List of detected canonical category keys (e.g. ["sevas"])
    intent: str           # "targeted_fact", "complete_list", "aggregation", "multi_category", "broad_information", "unsupported"
    condition: str        # "positive", "negative", "none"
    exhaustive: bool      # True if query requires exhaustive retrieval (not Top-3)
    retrieval_mode: str   # "targeted", "complete_category", "cross_temple_aggregation", "multi_category", "broad_temple", "unsupported"
    raw_query: str

class QueryPlanner:
    """
    100% Local, deterministic Query Planner for PilgrimAI.
    Analyzes natural language queries and outputs a structured QueryPlan without LLM/API calls.
    """

    def __init__(self, registry: Optional[CategoryRegistry] = None):
        self.category_registry = registry or category_registry

        # Canonical temple aliases map (all 20 temples)
        self.temple_aliases: Dict[str, List[str]] = {
            "Ahobilam": ["ahobilam", "ahobila", "narasimha ahobilam"],
            "Badrachalam": ["badrachalam", "bhadrachalam", "bhadragiri", "rama temple bhadrachalam", "lord rama bhadrachalam"],
            "Balkampet": ["balkampet", "yellamma", "balkampet yellamma"],
            "Basara": ["basara", "basar", "gnana saraswathi", "saraswathi temple"],
            "Birla Mandir": ["birla mandir", "birla"],
            "Chilkur": ["chilkur", "visa balaji", "chilkur balaji"],
            "Dharmapuri": ["dharmapuri", "narasimha swamy dharmapuri", "dharmapuri narasimha"],
            "Edupayal": ["edupayal", "edupayala", "edupayala durga", "vanadurga bhavani"],
            "Jogulamba(New)": ["jogulamba", "jogulamba(new)", "alampur", "jogulamba devi"],
            "Kaleshvaram": ["kaleshvaram", "kaleshwaram", "mukteshwara", "kaleshwara"],
            "Karmanghat": ["karmanghat", "karmaghat", "karmanghat hanuman"],
            "Medaram": ["medaram", "sammakka", "saralamma", "jatara medaram", "sammakka saralamma"],
            "Ramappa": ["ramappa", "ramalingeswara", "rudreshwara"],
            "Srisaila": ["srisaila", "srisailam", "mallikarjuna", "bhramaramba"],
            "Thousand Pillar": ["thousand pillar", "thousand pillars", "rudreswara thousand", "thousand pillar temple"],
            "Ttd": ["ttd", "tirumala", "tirupati", "venkateswara ttd", "lord venkateswara"],
            "Ujjaini": ["ujjaini", "ujjaini mahakali", "mahakali secunderabad", "secunderabad mahakali"],
            "Veera Satyanarayana": ["veera satyanarayana", "satyanarayana swamy", "annavaram", "annavarm", "veera venkata satyanarayana"],
            "Vemulavada": ["vemulavada", "vemulawada", "rajarajeshwara", "rajanne", "rajanna"],
            "Yadadri": ["yadadri", "yadagirigutta", "laxmi narasimha yadadri", "narasimha yadadri"]
        }

        # Cross-temple aggregation triggers
        self.aggregation_triggers = [
            r"\b(list\s+(all\s+)?(the\s+)?temples)\b",
            r"\b(which\s+(all\s+)?(the\s+)?temples)\b",
            r"\b(what\s+(all\s+)?(the\s+)?temples)\b",
            r"\b(show\s+(me\s+)?(all\s+)?(the\s+)?temples)\b",
            r"\b(give\s+(me\s+)?(all\s+)?(the\s+)?temples)\b",
            r"\b(all\s+(the\s+)?temples\s+(that|with|without|having|which|where|offering|providing|located))\b",
            r"\b(temples\s+(with|without|having|that\s+have|that\s+don'?t|that\s+do\s+not|offering|providing|located\s+on))\b",
            r"\b(how\s+many\s+temples\s+(have|are|offer|provide|don'?t|do\s+not))\b",
            r"\b(names\s+of\s+(all\s+)?temples)\b"
        ]

        # Negative indicators
        self.negative_indicators = [
            r"\b(don'?t\s+have|do\s+not\s+have|doesn'?t\s+have|does\s+not\s+have|without|lack|lacks|unavailable|not\s+available|not\s+provided)\b",
            r"\b(which\s+temples\s+(do\s+not|don'?t))\b",
            r"\b(temples\s+(that\s+)?(don'?t|do\s+not)\s+have)\b",
            r"\b(no\s+(wheelchair|accommodation|restaurants?|parking|river\s*bank|facilities))\b"
        ]

        # Complete-list indicators
        self.complete_list_triggers = [
            r"\b(all\s+(the\s+)?|list\s+all|what\s+are\s+all(\s+the)?|list\s+of|complete\s+list|full\s+list|entire\s+list|show\s+all|give\s+(me\s+)?all|all\s+available|every(\s+single)?|show\s+every|what\s+are\s+every|everything\s+available|complete\s+details|all\s+types|all\s+options|names\s+of\s+all)\b",
            r"\b(what|which)\s+(sevas?|poojas?|festivals?|prasadams?|facilities|restaurants?|attractions?)\s+(are\s+)?(performed|available|celebrated|there)\b",
            r"\b(sevas?|poojas?|festivals?|facilities|restaurants?)\s+(list|available|names)\b"
        ]

        # Broad information triggers
        self.broad_info_triggers = [
            r"\b(tell\s+me\s+everything\s+about|everything\s+about|all\s+about|complete\s+details\s+about|overview\s+of|complete\s+guide\s+to|full\s+details\s+of|complete\s+information\s+about)\b"
        ]

    def detect_temple(self, query: str) -> Optional[str]:
        """
        Detects explicit temple name in query using word-boundary matching.
        Returns canonical temple name or None.
        """
        q_norm = query.lower()
        for canonical_id, aliases in self.temple_aliases.items():
            for alias in sorted(aliases, key=len, reverse=True):
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, q_norm):
                    return canonical_id
        return None

    def plan(self, query: str) -> QueryPlan:
        """
        Plans the retrieval strategy for the input query.
        Returns a structured QueryPlan.
        """
        q = query.strip()
        q_lower = q.lower()

        # Step 1: Detect explicit temple entity
        detected_temple = self.detect_temple(q)

        # Step 2: Match all candidate categories via CategoryRegistry
        matched_categories = self.category_registry.match_categories(q)

        # Step 3: Detect negative condition
        is_negative = any(re.search(pat, q_lower) for pat in self.negative_indicators)
        condition = "negative" if is_negative else "positive"

        # -------------------------------------------------------------
        # Branch 1: Cross-Temple Multi-Entity Aggregation (No single temple detected)
        # -------------------------------------------------------------
        if not detected_temple:
            is_agg_trigger = any(re.search(pat, q_lower) for pat in self.aggregation_triggers)
            
            if is_agg_trigger and matched_categories:
                # Primary cross-temple aggregation path
                primary_cat = matched_categories[0]
                return QueryPlan(
                    scope="cross_temple",
                    temples=[],
                    categories=[primary_cat],
                    intent="aggregation",
                    condition=condition,
                    exhaustive=True,
                    retrieval_mode="cross_temple_aggregation",
                    raw_query=q
                )
            elif is_agg_trigger and not matched_categories:
                # Cross-temple trigger without registered category
                return QueryPlan(
                    scope="cross_temple",
                    temples=[],
                    categories=[],
                    intent="unsupported",
                    condition="none",
                    exhaustive=False,
                    retrieval_mode="unsupported",
                    raw_query=q
                )
            else:
                # General query without explicit temple
                return QueryPlan(
                    scope="general",
                    temples=[],
                    categories=matched_categories,
                    intent="targeted_fact",
                    condition="none",
                    exhaustive=False,
                    retrieval_mode="targeted",
                    raw_query=q
                )

        # -------------------------------------------------------------
        # Branch 2: Single-Temple Query
        # -------------------------------------------------------------
        # Check Broad Information Intent
        is_broad = any(re.search(pat, q_lower) for pat in self.broad_info_triggers)
        if is_broad or (len(matched_categories) == 0 and any(w in q_lower for w in ["tell me about", "about the temple", "details of"])):
            return QueryPlan(
                scope="single_temple",
                temples=[detected_temple],
                categories=[],
                intent="broad_information",
                condition="none",
                exhaustive=True,
                retrieval_mode="broad_temple",
                raw_query=q
            )

        # Check Multi-Category Intent (e.g. accommodation and restaurants)
        if len(matched_categories) >= 2:
            return QueryPlan(
                scope="single_temple",
                temples=[detected_temple],
                categories=matched_categories,
                intent="multi_category",
                condition="none",
                exhaustive=True,
                retrieval_mode="multi_category",
                raw_query=q
            )

        # Single Category or Targeted Inquiry
        if len(matched_categories) == 1:
            cat = matched_categories[0]
            is_list_trigger = any(re.search(pat, q_lower) for pat in self.complete_list_triggers)
            is_list_category = self.category_registry.is_list_oriented(cat)

            # Check if this is a complete list query for a list-oriented category
            if is_list_trigger or (is_list_category and any(w in q_lower for w in ["what are the", "which are the", "list", "all"])):
                return QueryPlan(
                    scope="single_temple",
                    temples=[detected_temple],
                    categories=[cat],
                    intent="complete_list",
                    condition="none",
                    exhaustive=True,
                    retrieval_mode="complete_category",
                    raw_query=q
                )
            else:
                # Single-temple targeted fact
                return QueryPlan(
                    scope="single_temple",
                    temples=[detected_temple],
                    categories=[cat],
                    intent="targeted_fact",
                    condition=condition if is_negative else "none",
                    exhaustive=False,
                    retrieval_mode="targeted",
                    raw_query=q
                )

        # Single temple, no recognized category (e.g. "Does Dharmapuri have a helicopter launch pad?")
        return QueryPlan(
            scope="single_temple",
            temples=[detected_temple],
            categories=[],
            intent="targeted_fact",
            condition=condition if is_negative else "none",
            exhaustive=False,
            retrieval_mode="targeted",
            raw_query=q
        )

query_planner = QueryPlanner()
