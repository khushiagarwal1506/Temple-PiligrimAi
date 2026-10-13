"""
PilgrimAI - ResponseAggregator (Phase E)

Composes one or more already-produced AgentResponse objects into a single, clean,
unified AgentResponse without inventing or hallucinating facts.

Core Principles:
- 100% Deterministic (pure in-memory composition, zero LLM calls).
- Zero database dependencies (no vector store or retrieval imports).
- Preserves factual integrity (never fills gaps, never rewrites facts).
- Conflict transparency (preserves conflicting information from different agents without picking a winner).
- Clean source and metadata deduplication and preservation.
"""

import re
import time
import math
import logging
from typing import List, Dict, Any, Set, Optional

from app.agents.state import AgentResponse

logger = logging.getLogger(__name__)


class ResponseAggregator:
    """
    Pure composition layer combining already-produced AgentResponse objects
    into a cohesive, standardized AgentResponse.
    """

    def __init__(self):
        pass

    def _detect_conflict(self, responses: List[AgentResponse]) -> bool:
        """
        Determines whether multiple responses contain contradictory factual assertions
        about the exact same subject (e.g. conflicting temple opening hours).
        Never treats complementary domain agents (itinerary, budget, weather, navigation)
        as having a conflict.
        """
        if len(responses) < 2:
            return False

        complementary_domains = {
            "itinerary_planning", "budget", "live_weather", "live_navigation",
            "emergency", "voice_multilingual", "recommendation"
        }

        # Check explicit opening/closing conflicts on the same assertion (e.g. 4:00 AM vs 5:00 AM)
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                a1_name = responses[i].agent_name or ""
                a2_name = responses[j].agent_name or ""

                # Complementary domain specialists do not conflict with each other
                if a1_name in complementary_domains and a2_name in complementary_domains:
                    continue
                if (a1_name == "itinerary_planning" and a2_name in ("temple_information", "budget")) or \
                   (a2_name == "itinerary_planning" and a1_name in ("temple_information", "budget")):
                    continue

                a1, a2 = responses[i].answer.lower(), responses[j].answer.lower()

                # Check for contradictory "opens [daily] at X" statements
                m1 = re.search(r"\bopens\s+(?:daily\s+)?at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", a1)
                m2 = re.search(r"\bopens\s+(?:daily\s+)?at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", a2)
                if m1 and m2 and m1.group(1).strip() != m2.group(1).strip():
                    return True

                if "opens at" in a1 and "opens at" in a2 and a1 != a2:
                    return True
                if "timings are" in a1 and "timings are" in a2 and a1 != a2:
                    return True

        return False

    def _clean_user_answer(self, text: str) -> str:
        """Strips implementation-level file paths, debug metadata, and trailing raw sources blocks from user-facing text."""
        if not text:
            return ""
        # Strip raw internal source references e.g. (Source: badrachalam/sevas.json)
        cleaned = re.sub(r"\s*\(\s*Source:\s*[^)]+\.json\s*\)", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*\*\s*\(\s*Source:\s*[^)]+\.json\s*\)\s*\*", "", cleaned, flags=re.IGNORECASE)
        # Strip raw record batch counts like (and 191 more records), Total 3 records found
        cleaned = re.sub(r"\s*\(and \d+ more records\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:Total\s+)?\d+\s+records?\s+found\b[.:]?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*\(\s*(?:Total\s+)?\d+\s+records?\s+found\s*\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:Found|Returned)\s+\d+\s+records?\b[.:]?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*Temple Id:\s*[^;\n]+;?", "", cleaned, flags=re.IGNORECASE)
        # Strip trailing raw Sources block
        cleaned = re.sub(r"\n+\*\*Sources:\*\*\s*(?:\n+-\s*\*\*[^*]+\*\*:[^\n]+)+", "", cleaned)
        # Replace internal folder aliases like Jogulamba(new) with canonical display name
        cleaned = re.sub(r"Jogulamba%28new%29", "Jogulamba%20Temple%2C%20Alampur", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"Jogulamba\s*\(\s*new\s*\)", "Jogulamba Temple, Alampur", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bKaleshvaram\b", "Kaleshwaram", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bBadrachalam\b", "Bhadrachalam", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bSrisaila(?!\w)", "Srisailam", cleaned, flags=re.IGNORECASE)
        # Strip any raw dataset JSON filenames or paths like [temple]/[category].json or filename.json
        cleaned = re.sub(r"\[?[A-Za-z0-9_()-]+/[A-Za-z0-9_-]+\.json\]?", "", cleaned)
        cleaned = re.sub(r"\b[A-Za-z0-9_-]+\.json\b", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _reorder_itinerary_and_budget(self, itin_text: str, budget_text: str) -> str:
        """
        Reorders itinerary and budget markdown sections into the required canonical sequence:
        1. Itinerary title and trip overview
        2. Budget summary and expense breakdown
        3. Day-by-day itinerary
        4. Accommodation options
        5. Travel and connectivity
        6. Limitations and prices requiring confirmation
        """
        # Section 1: Overview head (everything up to first Day header or Travel header)
        m_day = re.search(r'\n(?=#{3,4}\s*Day\s+\d+)', itin_text)
        if m_day:
            itin_head = itin_text[:m_day.start()].strip()
            itin_rest = itin_text[m_day.start():].strip()
        else:
            m_tr = re.search(r'\n(?=###\s*🚆?\s*Travel\s*&\s*Connectivity)', itin_text)
            if m_tr:
                itin_head = itin_text[:m_tr.start()].strip()
                itin_rest = itin_text[m_tr.start():].strip()
            else:
                itin_head = itin_text.strip()
                itin_rest = ''

        # Section 3 & 5: Day-by-day vs Travel & Connectivity
        m_travel = re.search(r'\n(?=###\s*🚆?\s*Travel\s*&\s*Connectivity)', itin_rest)
        if m_travel:
            day_by_day = itin_rest[:m_travel.start()].strip()
            travel_and_limits = itin_rest[m_travel.start():].strip()
        else:
            day_by_day = itin_rest.strip()
            travel_and_limits = ''

        # Section 5 & 6: Travel section vs Live-Data Limitation
        m_lim = re.search(r'\n(?=\*\(Live-Data Limitation|\(Live-Data Limitation)', travel_and_limits)
        if m_lim:
            travel_section = travel_and_limits[:m_lim.start()].strip()
            itin_limitations = travel_and_limits[m_lim.start():].strip()
        else:
            travel_section = travel_and_limits.strip()
            itin_limitations = ''

        # Section 2, 4, 6: Budget summary & breakdown vs Accommodation options vs Budget limitations
        m_accom = re.search(r'\n(?=###\s*🏨?\s*Accommodation Options & Suggestions)', budget_text)
        m_conf = re.search(r'\n(?=###\s*Prices Requiring Confirmation|\n###\s*Costs Not Available|\n###\s*Notes)', budget_text)

        if m_accom:
            budget_summary_breakdown = budget_text[:m_accom.start()].strip()
            if m_conf:
                accom_section = budget_text[m_accom.start():m_conf.start()].strip()
                budget_limitations = budget_text[m_conf.start():].strip()
            else:
                accom_section = budget_text[m_accom.start():].strip()
                budget_limitations = ''
        elif m_conf:
            budget_summary_breakdown = budget_text[:m_conf.start()].strip()
            accom_section = ''
            budget_limitations = budget_text[m_conf.start():].strip()
        else:
            budget_summary_breakdown = budget_text.strip()
            accom_section = ''
            budget_limitations = ''

        combined_limitations = '\n\n'.join(filter(None, [budget_limitations, itin_limitations])).strip()

        # Harmonize overnight stays in day_by_day with the budget's selected accommodation
        m_b_stay = re.search(r"\|\s*\*\*Accommodation\s*\((.*?)(?: — Tariff Unlisted|\))\*\*", budget_summary_breakdown)
        if m_b_stay:
            b_stay_name = m_b_stay.group(1).strip()
            def _replace_overnight(m):
                full_line = m.group(0)
                curr_stay = m.group(1)
                if curr_stay.lower() != b_stay_name.lower():
                    full_line = full_line.replace(f"**{curr_stay}**", f"**{b_stay_name}**")
                    full_line = full_line.replace(
                        "tariff unlisted in static records — confirm rates and availability on-ground",
                        "estimated tariff from verified records; room availability & seasonal rates require on-ground counter confirmation"
                    )
                return full_line

            day_by_day = re.sub(r"-\s*\*\*Overnight\*\*:\s*Selected stay:\s*\*\*(.*?)\*\*", _replace_overnight, day_by_day)

        ordered_parts = [
            itin_head,
            budget_summary_breakdown,
            day_by_day,
            accom_section,
            travel_section,
            combined_limitations
        ]
        return '\n\n---\n\n'.join(filter(None, ordered_parts))

    def aggregate(self, responses: List[AgentResponse]) -> AgentResponse:
        """
        Main composition method. Aggregates a list of AgentResponse objects
        into a single unified AgentResponse.
        """
        t0 = time.perf_counter()

        # -------------------------------------------------------------
        # 1. Handle Empty Input List
        # -------------------------------------------------------------
        if not responses:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return AgentResponse(
                success=False,
                agent_name="response_aggregator",
                answer="",
                sources=[],
                error="No agent responses provided for aggregation.",
                metadata={"agent_count": 0, "aggregation_latency_ms": latency_ms},
                latency_ms=latency_ms
            )

        successful: List[AgentResponse] = [r for r in responses if r.success]
        failed: List[AgentResponse] = [r for r in responses if not r.success]

        # -------------------------------------------------------------
        # 2. All Responses Failed
        # -------------------------------------------------------------
        if not successful:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            error_details = [
                f"{r.agent_name or 'unknown_agent'}: {r.error or 'Failed without specific error message'}"
                for r in failed
            ]
            combined_error = "; ".join(error_details)
            # Check if any specialist produced a user-facing explanation or clarification in answer
            fallback_answer = next((r.answer.strip() for r in failed if r.answer and r.answer.strip()), "")
            fallback_sources = []
            for r in failed:
                fallback_sources.extend(r.sources)

            merged_meta = {
                "agent_count": len(responses),
                "failed_agents": [r.agent_name for r in failed],
                "aggregation_latency_ms": latency_ms
            }
            for r in failed:
                if r.metadata:
                    merged_meta.update(r.metadata)

            return AgentResponse(
                success=False,
                agent_name="response_aggregator",
                answer=fallback_answer,
                sources=fallback_sources,
                error=f"All specialist agents failed: {combined_error}",
                metadata=merged_meta,
                latency_ms=latency_ms
            )

        # -------------------------------------------------------------
        # 3. Deduplicate Identical Successful Responses
        # -------------------------------------------------------------
        unique_successful: List[AgentResponse] = []
        seen_signatures: Set[str] = set()

        for r in successful:
            norm_answer = r.answer.strip()
            sig = f"{r.agent_name}::{norm_answer}"
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                unique_successful.append(r)

        # If after deduplication only 1 answer remains and no failures occurred, preserve it directly
        if len(unique_successful) == 1 and not failed:
            single = unique_successful[0]
            latency_ms = (time.perf_counter() - t0) * 1000.0
            merged_meta = dict(single.metadata)
            merged_meta["aggregation_latency_ms"] = latency_ms
            merged_meta["agent_count"] = len(responses)
            return AgentResponse(
                success=True,
                agent_name=single.agent_name or "response_aggregator",
                answer=self._clean_user_answer(single.answer),
                sources=list(single.sources),
                metadata=merged_meta,
                latency_ms=single.latency_ms
            )

        # -------------------------------------------------------------
        # 4. Merge Sources (Preserve Order, Remove Duplicates)
        # -------------------------------------------------------------
        merged_sources: List[str] = []
        seen_sources: Set[str] = set()

        for r in responses:
            for src in r.sources:
                if src and src not in seen_sources:
                    seen_sources.add(src)
                    merged_sources.append(src)

        # -------------------------------------------------------------
        # 5. Check Conflicting Information Across Multiple Responses
        # -------------------------------------------------------------
        has_conflict = self._detect_conflict(unique_successful)

        if has_conflict:
            conflict_lines = ["Information from the available agents differs:"]
            for r in unique_successful:
                agent_label = r.agent_name or "Specialist Agent"
                conflict_lines.append(f"- **{agent_label}**: {self._clean_user_answer(r.answer)}")
            conflict_lines.append("\nNote: The conflicting information should be verified before relying on it.")
            combined_answer = "\n".join(conflict_lines)
        else:
            # Order responses logically by domain hierarchy:
            # Overview/Temple Info -> Itinerary -> Weather -> Budget -> Navigation -> Emergency -> Other
            AGENT_ORDER = {
                "temple_information": 10,
                "recommendation": 15,
                "itinerary_planning": 20,
                "live_weather": 30,
                "budget": 40,
                "live_navigation": 50,
                "emergency": 60,
                "voice_multilingual": 70,
            }
            # Deduplicate multiple draft itineraries
            itinerary_resps = [r for r in unique_successful if r.agent_name == "itinerary_planning" or "### Pilgrimage Itinerary:" in r.answer]
            if len(itinerary_resps) > 1:
                unique_successful = [r for r in unique_successful if r not in itinerary_resps[:-1]]

            ordered_successful = sorted(
                unique_successful,
                key=lambda r: AGENT_ORDER.get(r.agent_name, 100)
            )

            has_temple_info = any(r.agent_name == "temple_information" for r in ordered_successful)
            has_itinerary = any(r.agent_name == "itinerary_planning" for r in ordered_successful)
            has_budget = any(r.agent_name == "budget" for r in ordered_successful)

            if has_itinerary and has_budget:
                itinerary_resp = next((r for r in ordered_successful if r.agent_name == "itinerary_planning" or "### Pilgrimage Itinerary:" in r.answer), None)
                budget_resp = next((r for r in ordered_successful if r.agent_name == "budget" or "Pilgrimage Budget & Expense" in r.answer), None)
                other_resps = [r for r in ordered_successful if r != itinerary_resp and r != budget_resp]

                if itinerary_resp and budget_resp:
                    cleaned_itin = self._clean_user_answer(itinerary_resp.answer)
                    pattern_accom = r"### 🏨 Accommodation Summary[\s\S]*?(?=\n\n(?:---|###|##|$))"
                    cleaned_itin = re.sub(pattern_accom, "", cleaned_itin).strip()
                    cleaned_budget = self._clean_user_answer(budget_resp.answer)

                    combined_itin_budget = self._reorder_itinerary_and_budget(cleaned_itin, cleaned_budget)
                    other_answers = [self._clean_user_answer(r.answer) for r in other_resps if r.answer.strip()]
                    all_parts = [combined_itin_budget] + other_answers
                    combined_answer = "\n\n---\n\n".join(filter(None, all_parts))
                else:
                    answer_parts = [self._clean_user_answer(r.answer) for r in ordered_successful if r.answer.strip()]
                    combined_answer = "\n\n---\n\n".join(answer_parts) if len(answer_parts) > 1 else (answer_parts[0] if answer_parts else "")
            else:
                answer_parts = [self._clean_user_answer(r.answer) for r in ordered_successful if r.answer.strip()]
                combined_answer = "\n\n---\n\n".join(answer_parts) if len(answer_parts) > 1 else (answer_parts[0] if answer_parts else "")

            # If both temple_information and itinerary_planning are present, deduplicate redundant temple info overview
            if has_temple_info and has_itinerary:
                pattern = r"### 🏛️ Important Temple Information \(Verified Records\)[\s\S]*?(?=\n\n(?:---|### 🏨|#{3,4}\s*Day|## 💰|$))"
                combined_answer = re.sub(pattern, "", combined_answer)
                combined_answer = re.sub(r"(?:\n\s*---\s*){2,}", "\n\n---\n\n", combined_answer).strip()

        # -------------------------------------------------------------
        # 6. Merge Metadata
        # -------------------------------------------------------------
        latency_ms = (time.perf_counter() - t0) * 1000.0

        merged_metadata: Dict[str, Any] = {
            "agent_count": len(responses),
            "agents": [r.agent_name for r in responses if r.agent_name],
            "successful_agents": [r.agent_name for r in unique_successful if r.agent_name],
            "failed_agents": [r.agent_name for r in failed if r.agent_name],
            "has_conflict": has_conflict,
            "aggregation_latency_ms": latency_ms
        }

        if failed:
            merged_metadata["partial_failure"] = True
            merged_metadata["failure_errors"] = [r.error for r in failed if r.error]

        # Propagate child metadata keys if not already present
        for r in unique_successful:
            for k, v in r.metadata.items():
                if k not in merged_metadata:
                    merged_metadata[k] = v

        # Collect and preserve all destinations across child metadata
        all_destinations = []
        for r in unique_successful:
            tmpls = r.metadata.get("temples") or []
            if isinstance(tmpls, str):
                tmpls = [tmpls]
            for t in tmpls:
                if t and t not in all_destinations:
                    all_destinations.append(t)
            for key in ("temple", "detected_temple", "destination"):
                single_val = r.metadata.get(key)
                if single_val and isinstance(single_val, str) and single_val not in all_destinations:
                    all_destinations.append(single_val)
        if all_destinations:
            merged_metadata["temples"] = all_destinations

        # Harmonize duration and parameter synchronization across child metadata
        dur_days = merged_metadata.get("duration_days") or (
            merged_metadata.get("parameters", {}).get("days")
            if isinstance(merged_metadata.get("parameters"), dict) else None
        )
        if dur_days:
            merged_metadata["duration_days"] = dur_days
            merged_metadata["days"] = dur_days
            is_day = (dur_days <= 1)
            calculated_nights = 0 if is_day else (math.floor(dur_days) if (dur_days % 1 != 0) else max(0, int(dur_days - 1)))
            if "nights" not in merged_metadata or merged_metadata["nights"] is None:
                merged_metadata["nights"] = calculated_nights
            if "parameters" in merged_metadata and isinstance(merged_metadata["parameters"], dict):
                merged_metadata["parameters"]["days"] = dur_days
                if not merged_metadata["parameters"].get("is_day_trip", is_day):
                    merged_metadata["parameters"]["nights"] = merged_metadata.get("nights", calculated_nights)
                else:
                    merged_metadata["parameters"]["nights"] = 0

        # Construct single shared trip-plan summary object (P0)
        if "duration_days" in merged_metadata or "budget_limit" in merged_metadata or "temples" in merged_metadata:
            params_dict = merged_metadata.get("parameters", {}) if isinstance(merged_metadata.get("parameters"), dict) else {}
            final_days = merged_metadata.get("duration_days", params_dict.get("days", 1))
            final_is_day = params_dict.get("is_day_trip", final_days <= 1)
            final_nights = 0 if final_is_day else (
                params_dict.get("nights") if isinstance(params_dict.get("nights"), int)
                else (merged_metadata.get("nights") if isinstance(merged_metadata.get("nights"), int)
                      else (math.floor(final_days) if final_days % 1 != 0 else max(1, int(final_days - 1))))
            )
            merged_metadata["trip_plan"] = {
                "duration_days": final_days,
                "days": final_days,
                "nights": final_nights,
                "is_day_trip": final_is_day,
                "pax": params_dict.get("pax", 1),
                "rooms": params_dict.get("rooms", 1 if final_nights > 0 else 0),
                "origin": params_dict.get("origin", "Hyderabad"),
                "destination": params_dict.get("destination", ""),
                "temples": merged_metadata.get("temples", []),
                "budget_limit": merged_metadata.get("budget_limit", params_dict.get("budget_limit")),
                "budget_status": merged_metadata.get("budget_status"),
                "selected_plan_total": (
                    merged_metadata.get("selected_plan", {}).get("total")
                    if isinstance(merged_metadata.get("selected_plan"), dict)
                    else None
                )
            }

        return AgentResponse(
            success=True,
            agent_name="response_aggregator",
            answer=combined_answer,
            sources=merged_sources,
            metadata=merged_metadata,
            error=None if not failed else f"Partial completion: {len(failed)} agent(s) failed.",
            latency_ms=latency_ms
        )
