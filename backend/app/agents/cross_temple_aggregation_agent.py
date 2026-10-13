"""
PilgrimAI - CrossTempleAggregationAgent (Phase L)

Universal Category-Driven Cross-Temple Aggregation Specialist for PilgrimAI.
Refactored into the BaseAgent multi-agent framework while preserving 100% backward
compatibility with legacy RAG pipeline interfaces.

Key Capabilities:
- Dynamically evaluates all 20 registered pilgrimage temples across CategoryRegistry categories.
- Guarantees zero Top-K truncation across the 20-temple domain.
- Strictly categorizes findings into three-tier availability:
    1. CONFIRMED_AVAILABLE (Affirmative verified evidence)
    2. CONFIRMED_UNAVAILABLE (Explicit negative verified evidence)
    3. INFORMATION_NOT_AVAILABLE (Unrecorded / missing data; never inferred as absence)
- Supports positive and negative aggregation queries.
- Interfaces strictly via RAGTool facade (zero raw DB or direct filesystem access).
- Produces strongly-typed AgentResponse with execution trace and latency tracking.
- Retains legacy methods: aggregate_temples(), format_response(), detect_aggregation_intent(), validate_match().
"""

import re
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, TYPE_CHECKING

from app.agents.base_agent import BaseAgent
from app.agents.state import AgentState, AgentResponse
if TYPE_CHECKING:
    from app.agents.tools.rag_tool import RAGTool

from app.rag.category_registry import category_registry, CategoryRegistry
from app.rag.completeness_validator import (
    completeness_validator,
    CompletenessValidator,
    CONFIRMED_AVAILABLE,
    CONFIRMED_UNAVAILABLE,
    INFORMATION_NOT_AVAILABLE
)
from app.rag.context_builder import context_builder, ContextBuilder
from app.rag.query_planner import query_planner, QueryPlanner

logger = logging.getLogger(__name__)


class CrossTempleAggregationAgent(BaseAgent):
    """
    Specialist agent providing grounded, universal cross-temple aggregation
    across all 20 registered pilgrimage temples in Telangana.
    """

    def __init__(
        self,
        rag_tool: Optional[Any] = None,
        cat_registry: Optional[CategoryRegistry] = None,
        k_registry: Optional[Any] = None,
        validator: Optional[CompletenessValidator] = None,
        builder: Optional[ContextBuilder] = None,
        planner: Optional[QueryPlanner] = None,
        name: str = "cross_temple_aggregation",
        description: str = "Universal category-driven cross-temple aggregation specialist dynamically evaluating verified records across all registered pilgrimage temples."
    ):
        super().__init__(name=name, description=description)
        self._rag_tool = rag_tool
        self.category_registry = cat_registry or category_registry
        self.knowledge_registry = k_registry
        self.validator = validator or completeness_validator
        self.context_builder = builder or context_builder
        self.planner = planner or query_planner

    @property
    def rag_tool(self) -> Any:
        """Lazily initialize RAGTool to break circular instantiation with the underlying RAG service."""
        if self._rag_tool is None:
            from app.agents.tools.rag_tool import RAGTool
            self._rag_tool = RAGTool()
        return self._rag_tool

    @rag_tool.setter
    def rag_tool(self, value: Any) -> None:
        self._rag_tool = value

    # -------------------------------------------------------------------------
    # BaseAgent Lifecycle Contract Implementation
    # -------------------------------------------------------------------------

    def _run(self, state: AgentState) -> AgentResponse:
        """
        Executes grounded cross-temple aggregation across all registered temples.
        Called automatically by BaseAgent.execute(state).
        """
        raw_query = state.user_query.strip() if state.user_query else ""

        # 1. Input Validation
        if not raw_query:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer="Please provide a valid cross-temple aggregation question.",
                sources=[],
                error="Empty query"
            )

        # 2. Extract Category/Property and Condition
        property_key = None
        plan = None
        try:
            plan = self.planner.plan(raw_query)
            if plan and plan.categories:
                property_key = plan.categories[0]
        except Exception as e:
            logger.warning(f"QueryPlanner plan failed in CrossTempleAggregationAgent: {e}")

        if not property_key:
            property_key = self.resolve_property(raw_query)

        if not property_key:
            matched = self.category_registry.match_categories(raw_query)
            if matched:
                property_key = matched[0]

        if not property_key:
            return AgentResponse(
                success=False,
                agent_name=self.name,
                answer=(
                    "Could not determine which temple feature or category to search for across temples. "
                    "Please specify a feature such as river bank, wheelchair accessibility, accommodation, or annadanam."
                ),
                sources=[],
                error="Unrecognized aggregation category",
                metadata={"clarification_needed": True}
            )

        # Determine condition: positive vs negative
        condition = "positive"
        if re.search(r"\b(not|without|no\s+|lacks?|non[-\s]?|don'?t|do\s+not)\b", raw_query, re.IGNORECASE):
            condition = "negative"
        elif plan and getattr(plan, "condition", None) in ("positive", "negative"):
            condition = plan.condition

        # 3. Execute Universal Aggregation across 20 temples
        agg_res = self.aggregate_temples(property_key, condition=condition, query=raw_query)

        # 4. Format Source-Grounded Markdown Response
        formatted_answer = self.format_response(agg_res, raw_query)

        # 5. Populate Intermediate Results and Metadata
        state.intermediate_results[self.name] = agg_res

        sources = agg_res.get("sources", [])
        matches = agg_res.get("matches", [])
        confirmed_available = agg_res.get("confirmed_available", [])
        confirmed_unavailable = agg_res.get("confirmed_unavailable", [])
        information_not_available = agg_res.get("information_not_available", [])

        metadata = {
            "category": property_key,
            "condition": condition,
            "total_temples_searched": agg_res.get("total_temples_searched", 20),
            "matching_temples_count": agg_res.get("matching_temples_count", len(matches)),
            "matching_temples": [m["temple"] for m in matches if "temple" in m],
            "confirmed_available_count": len(confirmed_available),
            "confirmed_unavailable_count": len(confirmed_unavailable),
            "information_not_available_count": len(information_not_available),
            "complete": agg_res.get("complete", True)
        }

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer=formatted_answer,
            sources=sources,
            metadata=metadata
        )

    # -------------------------------------------------------------------------
    # Legacy Compatibility Methods (Preserves backward compatibility with chat.py)
    # -------------------------------------------------------------------------

    def detect_aggregation_intent(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Detects whether a query is asking for cross-temple aggregation.
        Delegates to the deterministic QueryPlanner.
        """
        plan = self.planner.plan(query)
        if plan.retrieval_mode == "cross_temple_aggregation" and plan.categories:
            return {
                "is_aggregation": True,
                "property_key": plan.categories[0],
                "condition": plan.condition,
                "query": query,
                "plan": plan
            }
        return None

    def resolve_property(self, query: str) -> Optional[str]:
        """
        Resolves the query text to a canonical category key via CategoryRegistry.
        """
        matched = self.category_registry.match_categories(query)
        return matched[0] if matched else None

    def validate_match(self, data: Any, property_key: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Backward-compatible helper inspecting JSON content to determine if property is affirmative.
        Delegates to CompletenessValidator.
        Returns: (is_match, evidence_text, field_name)
        """
        status, evidence, field = self.validator.evaluate_availability(data, property_key)
        is_match = (status == CONFIRMED_AVAILABLE)
        return is_match, evidence, field

    def aggregate_temples(
        self,
        property_key: str,
        condition: str = "positive",
        query: str = ""
    ) -> Dict[str, Any]:
        """
        Universal category-driven aggregation across ALL 20 temples.
        Iterates over all temples, inspects source records, classifies into
        CONFIRMED_AVAILABLE, CONFIRMED_UNAVAILABLE, and INFORMATION_NOT_AVAILABLE,
        and returns complete verified results.
        """
        # Handle case where caller passed query as the second argument
        actual_condition = "positive"
        actual_query = query
        if condition in ("positive", "negative"):
            actual_condition = condition
        elif isinstance(condition, str) and len(condition) > 0:
            actual_query = condition
            plan = self.planner.plan(actual_query)
            actual_condition = plan.condition if plan.condition in ("positive", "negative") else "positive"

        # Retrieve registered temples: prefer injected knowledge_registry or RAGTool
        if self.knowledge_registry and hasattr(self.knowledge_registry, "get_temples"):
            temples = self.knowledge_registry.get_temples()
        else:
            temples = self.rag_tool.get_registered_temples()

        evaluation_results: List[Dict[str, Any]] = []

        for temple_name in temples:
            if self.knowledge_registry and hasattr(self.knowledge_registry, "get_temple_raw_data"):
                data = self.knowledge_registry.get_temple_raw_data(temple_name, property_key)
            else:
                data = self.rag_tool.get_temple_category_data(temple_name, property_key)

            status, evidence, field = self.validator.evaluate_availability(data, property_key)

            # Get relative source file paths
            if self.knowledge_registry and hasattr(self.knowledge_registry, "get_source_files"):
                src_files = self.knowledge_registry.get_source_files(temple_name, property_key)
                src_path = src_files[0] if src_files else f"{temple_name.lower()}/{property_key}.json"
            else:
                clean_t = temple_name.lower().replace(" ", "_")
                src_path = f"{clean_t}/{property_key}.json"

            item_record = {
                "temple": temple_name,
                "status": status,
                "evidence": evidence,
                "source_field": field,
                "source_file": src_path,
                "property": property_key,
                "matched": (status == CONFIRMED_AVAILABLE) if actual_condition == "positive" else (status == CONFIRMED_UNAVAILABLE),
                "confidence": 10.0
            }
            evaluation_results.append(item_record)

        # Validate completeness across 20 temples
        val_summary = self.validator.validate_cross_temple_aggregation(
            category=property_key,
            condition=actual_condition,
            evaluation_results=evaluation_results,
            total_scope=len(temples)
        )

        # Select matching items based on condition
        if actual_condition == "negative":
            matches = val_summary["confirmed_unavailable"]
        else:
            matches = val_summary["confirmed_available"]

        # Deduplicate results by canonical temple name
        deduped_matches = self.deduplicate_results(matches)

        # Collect source files for matching records
        sources = sorted(list({m["source_file"] for m in deduped_matches if m.get("source_file")}))

        return {
            "property_key": property_key,
            "category": property_key,
            "condition": actual_condition,
            "total_temples_searched": len(temples),
            "total_scope_evaluated": len(temples),
            "matching_temples_count": len(deduped_matches),
            "matches": deduped_matches,
            "confirmed_available": val_summary["confirmed_available"],
            "confirmed_unavailable": val_summary["confirmed_unavailable"],
            "information_not_available": val_summary["information_not_available"],
            "sources": sources,
            "complete": val_summary["complete_evaluation"]
        }

    def deduplicate_results(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicates matched items preserving distinct canonical temple entries.
        """
        seen: Set[str] = set()
        deduped = []
        for m in matches:
            t = m["temple"]
            if t not in seen:
                seen.add(t)
                deduped.append(m)
        return deduped

    def format_response(self, aggregation_result: Dict[str, Any], query: str = "") -> str:
        """
        Formats deterministic, source-grounded response using ContextBuilder.
        """
        return self.context_builder.build_cross_temple_context(aggregation_result, query)
