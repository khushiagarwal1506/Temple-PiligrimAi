"""
PilgrimAI - Sequential Multi-Intent Pipeline (Phase H)

Executes multiple distinct user intents sequentially in deterministic order.
Preserves query context and explicit temple identity across decomposed sub-tasks.
Delegates factual execution strictly to registered specialist agents via BaseAgent.execute().
Aggregates all outcomes through ResponseAggregator.

Core Principles:
- 100% Deterministic (pure rule-based query decomposition and sequential dispatch).
- Zero LLM generation / classification.
- Zero direct database, vector index, or tool imports.
- Purely synchronous (zero threads, zero asynchronous execution).
- Graceful partial failure and future capability containment.
"""

import re
import time
import math
import logging
from typing import Dict, Any, List, Optional

from app.agents.base_agent import BaseAgent
from app.agents.state import AgentState, AgentResponse, AgentRouteDecision, ExecutionTrace
from app.agents.router import AgentRouter
from app.agents.response_aggregator import ResponseAggregator
from app.agents.temple_information_agent import TempleInformationAgent

logger = logging.getLogger(__name__)


class SequentialMultiIntentPipeline:
    """
    Deterministic multi-intent execution engine.
    Decomposes multi-intent queries into ordered tasks, sequentially invokes
    specialist agents, and synthesizes unified responses.
    """

    def __init__(
        self,
        router: Optional[AgentRouter] = None,
        aggregator: Optional[ResponseAggregator] = None,
        temple_info_agent: Optional[TempleInformationAgent] = None,
        agents: Optional[Dict[str, BaseAgent]] = None,
        memory_manager: Optional[Any] = None
    ):
        self.router = router or AgentRouter()
        self.aggregator = aggregator or ResponseAggregator()
        self.memory_manager = memory_manager

        if agents is not None:
            self.agents: Dict[str, BaseAgent] = dict(agents)
        else:
            default_agent = temple_info_agent or TempleInformationAgent()
            self.agents = {
                default_agent.name: default_agent
            }

    def decompose_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Deterministically decomposes a conjoined query into ordered sub-tasks,
        preserving detected temple context across clauses.
        """
        from app.agents.canonical_constraints import normalize_duration_expression
        clean_query = normalize_duration_expression(query.strip())
        detected_temples = self.router._detect_temples(clean_query)
        primary_temple = detected_temples[0] if detected_temples else None

        # Protect conjunctions connecting multiple detected temple names/aliases so multi-temple
        # itinerary / comparison phrases stay intact in a single clause
        from app.agents.coordinate_registry import _CANONICAL_TEMPLE_ALIASES
        all_aliases = set()
        for a_list in _CANONICAL_TEMPLE_ALIASES.values():
            all_aliases.update(a_list)
        if hasattr(self.router, "planner") and hasattr(self.router.planner, "temple_aliases"):
            for a_list in self.router.planner.temple_aliases.values():
                all_aliases.update(a_list)

        found_temple_keywords = []
        for kw in sorted(all_aliases, key=len, reverse=True):
            if re.search(rf"\b{re.escape(kw)}\b", clean_query, flags=re.IGNORECASE):
                found_temple_keywords.append(kw)

        protected_query = clean_query
        # Protect half-day durations e.g. "3 and half", "3 and a half", "three and a half"
        protected_query = re.sub(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+and\s+(?:a\s+)?half\b",
            r"\1 __CONJOINED_AND__ half",
            protected_query,
            flags=re.IGNORECASE
        )
        # Protect "between X and Y"
        protected_query = re.sub(
            r"(\bbetween\s+[A-Za-z0-9\s,.-]+?)\s+and\s+([A-Za-z0-9\s,.-]+)",
            r"\1 __CONJOINED_AND__ \2",
            protected_query,
            flags=re.IGNORECASE
        )
        # Protect traveller conjunctions e.g. "2 adults and 2 children and 1 senior citizen"
        trav_conj = re.compile(
            r"(\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s+(?:adults?|children|kids?|child|seniors?|senior\s+citizens?|elderly(?:\s+persons?|\s+people)?|adlts?|chld|chlds?|adulta)\b)\s+(?:and|&)\s+(\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s+(?:adults?|children|kids?|child|seniors?|senior\s+citizens?|elderly(?:\s+persons?|\s+people)?|adlts?|chld|chlds?|adulta)\b)",
            re.IGNORECASE
        )
        while trav_conj.search(protected_query):
            protected_query = trav_conj.sub(r"\1 __CONJOINED_AND__ \2", protected_query)

        if len(found_temple_keywords) > 1:
            for t1 in found_temple_keywords:
                for t2 in found_temple_keywords:
                    if t1.lower() != t2.lower():
                        # Protect direct temple-to-temple conjunctions (e.g. "Basara and Dharmapuri", "Birla Mandir & Karmanghat")
                        # but do not protect across separate action clauses (e.g. "timings of Yadadri and list all sevas at Dharmapuri")
                        p = rf"(\b{re.escape(t1)}\b(?:\s+temple)?)\s+(?:and|&|along\s+with|as\s+well\s+as)\s+((?:to\s+)?(?:the\s+)?\b{re.escape(t2)}\b)"
                        protected_query = re.sub(p, r"\1 __CONJOINED_AND__ \2", protected_query, flags=re.IGNORECASE)

        # Split on conjunctions, question marks, semicolons, or transition boundaries into other intents
        split_pattern = (
            r"(?:"
            r"\s+(?:and|also|as\s+well\s+as|along\s+with|&)\s+"
            r"|\s*[?;]\s*"
            r"|\s+(?=(?:with\s+(?:a\s+)?budget|under\s+(?:a\s+)?budget|(?<!with\s)(?<!with\sa\s)budget\s+(?:is|of|for)|(?:with|under|for|within)\s+(?:(?:rs\.?|inr|₹)\s*\d+|\d+\s*(?:rupees?|rs\.?|inr)))\b)"
            r"|\s+(?=(?:mention|check|tell\s+me|give|what\s+is|what\s+will|include|how\s+is|weather|forecast|temperature|today['’]?s)\s+(?:the\s+|today['’]?s\s+)?(?:weather|temperature|forecast|rain|climate)\b)"
            r"|\s+(?=(?:how\s+to\s+reach|how\s+to\s+get|directions?\s+to|route\s+to|navigation\s+to|distance\s+to)\b)"
            r")"
        )
        raw_clauses = [c.replace("__CONJOINED_AND__", "and").strip() for c in re.split(split_pattern, protected_query, flags=re.IGNORECASE) if c.strip()]

        if not raw_clauses:
            raw_clauses = [clean_query]

        AGENT_NAME_MAP = {
            "LiveWeatherAgent": "live_weather",
            "weather": "live_weather",
            "live_weather": "live_weather",
            "BudgetAgent": "budget",
            "budget": "budget",
            "ItineraryPlanningAgent": "itinerary_planning",
            "itinerary_planning": "itinerary_planning",
            "itinerary": "itinerary_planning",
            "LiveNavigationAgent": "live_navigation",
            "live_navigation": "live_navigation",
            "navigation": "live_navigation",
            "RecommendationAgent": "recommendation",
            "recommendation": "recommendation",
            "EmergencyAgent": "emergency",
            "emergency": "emergency",
            "TempleInformationAgent": "temple_information",
            "temple_information": "temple_information",
            "sevas": "temple_information",
            "seva": "temple_information",
            "accommodation": "temple_information",
            "accomodation": "temple_information",
            "rooms": "temple_information",
            "timings": "temple_information",
            "timing": "temple_information",
            "darshan": "temple_information",
            "prasadam": "temple_information",
            "food": "temple_information",
            "facilities": "temple_information",
            "history": "temple_information",
            "rules": "temple_information",
            "dress_code": "temple_information",
            "travel": "temple_information",
            "ComparisonAgent": "comparison",
            "comparison": "comparison",
            "CrossTempleAgent": "cross_temple_aggregation",
            "cross_temple_aggregation": "cross_temple_aggregation"
        }

        tasks = []
        task_counter = 1
        for i, clause in enumerate(raw_clauses):
            # Check if clause already mentions a registered temple
            clause_temples = self.router._detect_temples(clause)

            if not clause_temples and primary_temple:
                # Propagate primary temple context to the sub-query
                if len(detected_temples) > 1:
                    task_query = f"{clause} at {' and '.join(detected_temples)}"
                else:
                    task_query = f"{clause} at {primary_temple} temple"
            else:
                task_query = clause

            # Route individual sub-query to determine task intent and specialist
            route_decision = self.router.route(task_query)

            if route_decision.intent == "multi_intent":
                # Unpack compound sub_intents if clause wasn't split by delimiters
                sub_intents = route_decision.metadata.get("sub_intents", [])
                if sub_intents:
                    for si in sub_intents:
                        agent_key = AGENT_NAME_MAP.get(si, si)
                        tasks.append({
                            "task_id": task_counter,
                            "raw_clause": clause,
                            "query": task_query,
                            "intent": si,
                            "target_agent": agent_key,
                            "is_available": agent_key in self.agents,
                            "route_decision": route_decision
                        })
                        task_counter += 1
                    continue

            # Determine target agent name and availability
            target_agent = route_decision.selected_agent
            if target_agent == "none":
                target_agent = route_decision.metadata.get("target_agent", route_decision.intent)

            # Normalize through canonical agent registry map
            target_agent = AGENT_NAME_MAP.get(target_agent, target_agent)
            is_available = target_agent in self.agents

            tasks.append({
                "task_id": task_counter,
                "raw_clause": clause,
                "query": task_query,
                "intent": route_decision.intent,
                "target_agent": target_agent,
                "is_available": is_available,
                "route_decision": route_decision
            })
            task_counter += 1

        return tasks

    def execute(self, state: AgentState) -> AgentResponse:
        """
        Executes decomposed tasks in strictly sequential order and aggregates results.
        """
        t0 = time.perf_counter()

        if not state or not isinstance(state, AgentState):
            err_resp = AgentResponse(
                success=False,
                agent_name="multi_intent_pipeline",
                answer="Invalid agent state provided.",
                sources=[],
                error="Invalid state object",
                latency_ms=0.0
            )
            return err_resp

        query = state.user_query.strip() if state.user_query else ""
        if not query:
            empty_resp = AgentResponse(
                success=False,
                agent_name="multi_intent_pipeline",
                answer="Please provide a valid question.",
                sources=[],
                error="Empty query",
                latency_ms=0.0
            )
            state.final_response = empty_resp
            return empty_resp

        # Log pipeline start
        state.add_trace(
            agent_name="multi_intent_pipeline",
            step_name="pipeline_start",
            status="started",
            metadata={"query": query}
        )

        # -------------------------------------------------------------
        # 1. Ambiguous & Unsupported Pre-Checks
        # -------------------------------------------------------------
        existing_context = {}
        if self.memory_manager and state.session_id:
            existing_context = self.memory_manager.get_trip_context(state.session_id) or {}

        detected_temples = self.router._detect_temples(query)
        unknown_temple = self.router._detect_unregistered_temple(query)

        from app.agents.canonical_constraints import is_trip_followup, extract_canonical_constraints, validate_and_synchronize_response

        # Inherit previously established destinations ONLY if demonstrably a follow-up
        if not detected_temples and is_trip_followup(query, existing_context, []):
            detected_temples = list(existing_context.get("destinations", []))

        if unknown_temple:
            unsupported_text = (
                f"I don't have verified information for '{unknown_temple}' in the current PilgrimAI knowledge base. "
                f"PilgrimAI currently covers 20 major pilgrimage temples in Telangana (such as Yadadri, Basara, Bhadrachalam, Srisailam, Dharmapuri, etc.)."
            )
            unsupported_resp = AgentResponse(
                success=False,
                agent_name="multi_intent_pipeline",
                answer=unsupported_text,
                sources=[],
                error=f"Unregistered temple '{unknown_temple}'",
                metadata={"detected_unknown_temple": unknown_temple, "supported": False},
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )
            state.add_trace(
                agent_name="multi_intent_pipeline",
                step_name="unsupported",
                status="completed",
                latency_ms=unsupported_resp.latency_ms
            )
            state.final_response = unsupported_resp
            return unsupported_resp

        if not detected_temples and is_trip_followup(query, existing_context, []) and existing_context.get("temples"):
            detected_temples = list(existing_context["temples"])

        if not detected_temples:
            clarification_text = (
                "Which temple would you like information about? "
                "PilgrimAI currently covers 20 major temples in Telangana (e.g., Yadadri, Basara, Bhadrachalam, Srisailam, Dharmapuri, etc.)."
            )
            clarification_resp = AgentResponse(
                success=False,
                agent_name="multi_intent_pipeline",
                answer=clarification_text,
                sources=[],
                error="Temple-related query does not identify a specific supported temple.",
                metadata={"clarification_needed": True, "missing_temple": True},
                latency_ms=(time.perf_counter() - t0) * 1000.0
            )
            state.add_trace(
                agent_name="multi_intent_pipeline",
                step_name="clarification",
                status="completed",
                latency_ms=clarification_resp.latency_ms
            )
            state.final_response = clarification_resp
            return clarification_resp

        # -------------------------------------------------------------
        # 1.5 Extract Request-Scoped Canonical Constraints
        # -------------------------------------------------------------
        canonical = extract_canonical_constraints(
            query=query,
            session_trip_context=existing_context,
            detected_temples=detected_temples
        )
        state.structured_constraints = canonical.to_dict()

        logger.info(
            f"[MULTI-INTENT CONSTRAINTS] session_id={state.session_id} query='{query[:80]}' "
            f"budget={canonical.budget_limit} pax={canonical.pax} (adults={canonical.adults}, children={canonical.children}) "
            f"days={canonical.duration_days} nights={canonical.nights} destinations={canonical.destinations} "
            f"is_followup={canonical.is_followup} inherited={canonical.inherited_fields}"
        )

        # Update or replace persistent session trip context
        if self.memory_manager and state.session_id:
            if not canonical.is_followup:
                self.memory_manager.replace_trip_context(state.session_id, canonical.to_dict())
            else:
                self.memory_manager.update_trip_context(state.session_id, canonical.to_dict())

        # -------------------------------------------------------------
        # 2. Decompose Into Ordered Tasks
        # -------------------------------------------------------------
        tasks = self.decompose_query(query)
        task_responses: List[AgentResponse] = []
        ordered_task_results: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # 3. Synchronous Sequential Task Execution
        # -------------------------------------------------------------
        for task in tasks:
            task_id = task["task_id"]
            intent = task["intent"]
            task_query = task["query"]
            target_agent_name = task["target_agent"]

            state.add_trace(
                agent_name="multi_intent_pipeline",
                step_name="task_start",
                status="started",
                metadata={"task_id": task_id, "intent": intent, "query": task_query, "target_agent": target_agent_name}
            )

            if task["is_available"]:
                # If query is primarily an itinerary planning request, skip spurious comparison tasks
                is_itinerary_query = bool(re.search(r"\b(plan|itinerary|trip|tour|visit\s+plan)\b", query, re.IGNORECASE))
                if intent == "comparison" and is_itinerary_query:
                    continue

                # Execute registered specialist agent via BaseAgent.execute lifecycle
                specialist_agent = self.agents[target_agent_name]
                task_state = AgentState(
                    session_id=state.session_id,
                    user_query=task_query,
                    structured_constraints=dict(state.structured_constraints)
                )

                t_task_start = time.perf_counter()
                task_resp = specialist_agent.execute(task_state)
                task_latency_ms = (time.perf_counter() - t_task_start) * 1000.0

                # Propagate traces from sub-task into parent state
                for tr in task_state.execution_trace:
                    state.execution_trace.append(tr)

                trace_status = "completed" if task_resp.success else "failed"
                state.add_trace(
                    agent_name="multi_intent_pipeline",
                    step_name=f"task_{trace_status}",
                    status=trace_status,
                    latency_ms=task_latency_ms,
                    error=task_resp.error,
                    metadata={"task_id": task_id, "intent": intent}
                )

                task_responses.append(task_resp)
                # Propagate verified itinerary parameters to shared constraints for downstream tasks (e.g. budget)
                if task_resp.agent_name == "itinerary_planning" and task_resp.success:
                    meta = task_resp.metadata or {}
                    if "duration_days" in meta:
                        dur = meta["duration_days"]
                        state.structured_constraints["duration_days"] = dur
                        state.structured_constraints["days"] = dur
                        if "nights" in meta and meta["nights"] is not None:
                            state.structured_constraints["nights"] = int(meta["nights"])
                        elif dur % 1 != 0:
                            state.structured_constraints["nights"] = math.floor(dur)
                        else:
                            state.structured_constraints["nights"] = max(0, int(dur - 1))
                        state.structured_constraints["is_day_trip"] = (dur <= 1)
                        state.structured_constraints["days_assumed"] = False
                    if "temples" in meta and meta["temples"]:
                        state.structured_constraints["temples"] = meta["temples"]
                    if "pax" in meta:
                        state.structured_constraints["pax"] = meta["pax"]
                    if "rooms" in meta:
                        state.structured_constraints["rooms"] = meta["rooms"]
                    if "origin" in meta and not meta.get("origin_assumed", True):
                        state.structured_constraints["origin"] = meta["origin"]
                        state.structured_constraints["origin_assumed"] = False

                ordered_task_results.append({
                    "task_id": task_id,
                    "intent": intent,
                    "query": task_query,
                    "target_agent": target_agent_name,
                    "success": task_resp.success,
                    "response": task_resp
                })

            else:
                # Unsupported capability handling (e.g., comparison, itinerary planning)
                future_label = intent.replace("_", " ").title()
                unavail_msg = f"The requested capability ('{future_label}') is not yet available in the current agent build."
                unavail_resp = AgentResponse(
                    success=False,
                    agent_name="multi_intent_pipeline",
                    answer=unavail_msg,
                    sources=[],
                    error=f"Capability '{intent}' has no implemented specialist agent.",
                    metadata={"task_id": task_id, "intent": intent, "available": False},
                    latency_ms=0.0
                )

                state.add_trace(
                    agent_name="multi_intent_pipeline",
                    step_name="task_skipped",
                    status="skipped",
                    metadata={"task_id": task_id, "intent": intent, "reason": "unimplemented_capability"}
                )

                task_responses.append(unavail_resp)
                ordered_task_results.append({
                    "task_id": task_id,
                    "intent": intent,
                    "query": task_query,
                    "target_agent": target_agent_name,
                    "success": False,
                    "response": unavail_resp
                })

        # -------------------------------------------------------------
        # 4. Record Intermediate Results in State
        # -------------------------------------------------------------
        state.intermediate_results["multi_intent"] = {
            "task_count": len(tasks),
            "successful_tasks": sum(1 for t in ordered_task_results if t["success"]),
            "failed_tasks": sum(1 for t in ordered_task_results if not t["success"]),
            "tasks": ordered_task_results
        }

        # -------------------------------------------------------------
        # 5. Response Aggregation via ResponseAggregator
        # -------------------------------------------------------------
        successful_resps = [r for r in task_responses if r.success]
        failed_resps = [r for r in task_responses if not r.success]

        if not successful_resps:
            # All tasks failed or unavailable
            final_response = self.aggregator.aggregate(task_responses)
        else:
            # Aggregate successful specialist responses
            aggregated_success = self.aggregator.aggregate(successful_resps)

            # If there were unavailable future intents, attach a controlled notice
            unavail_notes = [
                r.answer for r in failed_resps
                if r.metadata.get("available") is False and r.answer
            ]

            if unavail_notes:
                combined_answer = aggregated_success.answer + "\n\nNote: " + " ".join(unavail_notes)
                merged_meta = dict(aggregated_success.metadata)
                merged_meta["partial_capability_notice"] = True
                merged_meta["unimplemented_intents"] = [
                    r.metadata.get("intent") for r in failed_resps if r.metadata.get("available") is False
                ]

                final_response = AgentResponse(
                    success=True,
                    agent_name="response_aggregator",
                    answer=combined_answer,
                    sources=aggregated_success.sources,
                    metadata=merged_meta,
                    latency_ms=aggregated_success.latency_ms
                )
            else:
                final_response = aggregated_success

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        final_response.metadata["pipeline_latency_ms"] = total_latency_ms

        final_response = validate_and_synchronize_response(canonical, final_response)

        state.add_trace(
            agent_name="multi_intent_pipeline",
            step_name="pipeline_complete",
            status="completed" if final_response.success else "failed",
            latency_ms=total_latency_ms
        )

        state.final_response = final_response
        return final_response
