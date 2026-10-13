"""
PilgrimAI - ComparisonAgent (Phase K)

Specialist agent responsible for neutral, grounded comparisons between two or more
registered pilgrimage temples (timings, sevas, poojas, facilities, accommodation, etc.).
Strictly adheres to:
- BaseAgent lifecycle contract (Phase A).
- RAGTool read-only facade (Phase B) as the ONLY knowledge-access interface.
- Zero direct database, vector index, or embedding model imports.
- Zero subjective declarations ("better", "best", "superior", "worse", "winner").
- Explicit distinction between verified data and unrecorded data (never infer false absence).
- Strict multi-temple isolation (sources and evidence attributed per temple).
"""

import re
import logging
from typing import Optional, List, Dict, Set, Any

from app.agents.base_agent import BaseAgent
from app.agents.state import AgentState, AgentResponse
from app.agents.tools.rag_tool import RAGTool, RAGChunk
from app.rag.category_registry import category_registry, CategoryRegistry

logger = logging.getLogger(__name__)


class ComparisonAgent(BaseAgent):
    """
    Specialist agent providing grounded, neutral comparisons between two or more
    registered pilgrimage temples in Telangana.
    """

    def __init__(
        self,
        rag_tool: Optional[RAGTool] = None,
        name: str = "comparison",
        description: str = "Provides neutral, grounded comparisons between registered pilgrimage temples."
    ):
        super().__init__(name=name, description=description)
        self.rag_tool = rag_tool or RAGTool()
        self.category_registry: CategoryRegistry = category_registry
        self._registered_temples: List[str] = []
        self._temple_aliases: Dict[str, List[str]] = {}
        self._init_temple_registry()

    def _init_temple_registry(self):
        """
        Loads registered canonical temples dynamically from RAGTool and anchors aliases.
        Prevents duplicate independent temple registries.
        """
        self._registered_temples = self.rag_tool.get_registered_temples()

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
        for canonical in self._registered_temples:
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
        Uses word-boundary matching and alias length prioritization.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        matches: List[tuple[int, str]] = []

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

        return ordered_temples

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
            "compare", "between", "versus", "vs", "and"
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

    def _extract_category_data(self, temple: str, category: str) -> tuple[Optional[str], List[str]]:
        """
        Retrieves factual evidence and source filenames for a temple and category via RAGTool.
        Checks complete parent text first (Phase J compatibility), then standard retrieval.
        Returns: (content: Optional[str], sources: List[str])
        """
        sources: List[str] = []
        content: Optional[str] = None

        # 1. Try complete parent text (Phase J supported)
        try:
            parent_text = self.rag_tool.get_complete_parent_text(temple, category)
            if parent_text and parent_text.strip():
                content = parent_text.strip()
                clean_temple = temple.lower().replace(" ", "_")
                sources.append(f"{clean_temple}/{category}.json")
        except Exception as e:
            logger.warning(f"Error fetching complete parent for {temple} {category}: {e}")

        # 2. If no complete parent text, fallback to RAGTool retrieve
        if not content:
            try:
                chunks = self.rag_tool.retrieve(
                    query=f"{category} at {temple}",
                    temple=temple,
                    category=category,
                    top_k=3
                )
                if chunks:
                    texts = [c.text for c in chunks if c.text]
                    content = "\n\n".join(texts)
                    for c in chunks:
                        if c.source and c.source not in sources:
                            sources.append(c.source)
            except Exception as e:
                logger.error(f"Error retrieving {category} for {temple} via RAGTool: {e}")

        return content, sources

    def _run(self, state: AgentState) -> AgentResponse:
        """
        Executes neutral, grounded multi-temple comparison logic.
        Called automatically by BaseAgent.execute(state).
        """
        raw_query = state.user_query.strip() if state.user_query else ""

        # 1. Handle empty / whitespace input
        if not raw_query:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Please provide a valid comparison question.",
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
                    f"I don't have verified information for '{unknown_temple}' in the current PilgrimAI knowledge base. "
                    f"PilgrimAI currently covers 20 major pilgrimage temples in Telangana (such as Yadadri, Basara, Bhadrachalam, Srisailam, Dharmapuri, etc.)."
                ),
                sources=[],
                metadata={"detected_unknown_temple": unknown_temple},
                error="Unregistered temple"
            )

        # 3. Detect candidate registered temples
        detected_temples = self.detect_temples(raw_query)

        # 4. Handle ambiguous query lacking temples
        if len(detected_temples) == 0:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer=(
                    "Please specify the temples you want to compare. "
                    "For example: 'Compare Yadadri and Dharmapuri.'"
                ),
                sources=[],
                metadata={"clarification_needed": True, "detected_temples": []},
                error="Missing temple specification"
            )

        # 5. Handle query with only one temple specified
        if len(detected_temples) == 1:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer=(
                    f"Please specify the second temple you want to compare with {detected_temples[0]}. "
                    f"For example: 'Compare {detected_temples[0]} and Dharmapuri.'"
                ),
                sources=[],
                metadata={"clarification_needed": True, "detected_temples": detected_temples},
                error="Insufficient temples for comparison"
            )

        # 6. Identify target categories
        matched_categories = self.category_registry.match_categories(raw_query)

        # Determine categories to compare
        if matched_categories:
            categories_to_compare = matched_categories
        else:
            # Broad comparison default categories
            categories_to_compare = ["timings", "sevas", "facilities", "accommodation"]

        # 7. Retrieve data independently per temple and category (strictly isolated)
        temple_data: Dict[str, Dict[str, Optional[str]]] = {t: {} for t in detected_temples}
        sources_by_temple: Dict[str, List[str]] = {t: [] for t in detected_temples}
        all_sources: List[str] = []
        partial_failures: List[str] = []

        for temple in detected_temples:
            for cat in categories_to_compare:
                try:
                    content, sources = self._extract_category_data(temple, cat)
                    temple_data[temple][cat] = content
                    for s in sources:
                        # Enforce temple isolation on sources
                        t_clean = temple.lower().replace(" ", "_")
                        if t_clean in s.lower() or temple.lower() in s.lower():
                            if s not in sources_by_temple[temple]:
                                sources_by_temple[temple].append(s)
                            if s not in all_sources:
                                all_sources.append(s)
                except Exception as e:
                    logger.error(f"Failed retrieval for {temple} category {cat}: {e}")
                    temple_data[temple][cat] = None
                    partial_failures.append(f"{temple}:{cat}")

        # 8. Check for complete failure across all temples
        has_any_data = any(
            any(content is not None for content in cats.values())
            for cats in temple_data.values()
        )
        if not has_any_data:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="No verified data was found in the knowledge base for the requested temples and categories.",
                sources=[],
                error="No verified data found",
                metadata={"comparison_temples": detected_temples, "comparison_categories": categories_to_compare}
            )

        # 9. Format Neutral Comparison Output
        temple_header = " vs ".join(detected_temples)
        lines = [f"### Comparison: {temple_header}\n"]

        for cat in categories_to_compare:
            cat_display = cat.replace("_", " ").title()
            lines.append(f"#### {cat_display}")

            for temple in detected_temples:
                data = temple_data[temple].get(cat)
                if data:
                    # Summarize clean bullet points
                    summary = self._summarize_category_evidence(data, cat)
                    lines.append(f"- **{temple}**: {summary}")
                else:
                    lines.append(f"- **{temple}**: No verified {cat_display.lower()} category data available in the current knowledge base.")
            lines.append("")

        # Notes on conflicts or differing records if present
        conflict_notes = self._detect_conflicts(temple_data)
        if conflict_notes:
            lines.append("*(Note: The available records contain differing schedules/information. Please verify the current temple schedule before relying on them).*\n")

        # Source attribution preserved internally in AgentResponse.sources
        comparison_answer = "\n".join(lines).strip()

        # Success determination: successful if at least one temple has verified data
        success = True

        return AgentResponse(
            success=success,
            agent_name=self.name,
            answer=comparison_answer,
            sources=all_sources,
            metadata={
                "comparison_temples": detected_temples,
                "comparison_categories": categories_to_compare,
                "sources_by_temple": sources_by_temple,
                "partial_failures": partial_failures
            }
        )

    def _summarize_category_evidence(self, text: str, category: str) -> str:
        """
        Extracts concise, factual bullet summaries from category evidence.
        Strictly avoids subjective declarations ('better', 'best', 'superior', 'worse').
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        meaningful = [
            l for l in lines
            if not l.startswith("Temple:") and not l.startswith("Category:") and not l.startswith("File:")
        ]

        if not meaningful:
            return text[:150]

        # Extract up to 3 informative lines or items
        sample_items = []
        for l in meaningful:
            clean_l = re.sub(r"^[-*•\d.]+\s*", "", l).strip()
            if clean_l and clean_l not in sample_items:
                sample_items.append(clean_l)
            if len(sample_items) >= 3:
                break

        if sample_items:
            result = "; ".join(sample_items)
            if len(meaningful) > 3:
                result += f" (and {len(meaningful) - 3} more records)"
            return result
        return text[:150]

    def _detect_conflicts(self, temple_data: Dict[str, Dict[str, Optional[str]]]) -> bool:
        """
        Checks if any retrieved category evidence explicitly notes schedule contradictions or conflicting sources.
        Guards against triggering on ordinary operational disclaimers like 'subject to change' or 'seasonal variations'.
        """
        conflict_phrases = [
            "conflicting records",
            "differing sources",
            "discrepancy in records",
            "unverified discrepancy",
            "conflicting schedules",
            "differing schedule",
            "sources conflict",
            "unresolved discrepancy"
        ]
        for t_cats in temple_data.values():
            for text in t_cats.values():
                if text and any(p in text.lower() for p in conflict_phrases):
                    return True
        return False
