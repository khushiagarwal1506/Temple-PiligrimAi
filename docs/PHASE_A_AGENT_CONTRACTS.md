# PilgrimAI: Phase A — Foundational Agent Contracts

**Date:** September 17, 2026  
**Project:** PilgrimAI: A RAG-Based Multi-Agent Assistant for Smart Pilgrimage  
**Workspace:** `d:\pligrim ai new`  
**Phase:** Phase A: BaseAgent & Common Agent Contract  
**Status:** Certified & Passed (10/10 tests in 0.12s)  

---

## 1. Executive Summary

Phase A establishes the standardized, strongly-typed contracts and execution lifecycle for all future specialist agents in the PilgrimAI assistant.

These foundational contracts decouple agent orchestration from specific data retrieval implementations (Qdrant, BM25, Cross-Encoder, Ollama) and ensure that every specialist agent adheres to a predictable lifecycle with standardized timing, state propagation, error containment, and execution tracing.

---

## 2. Core Architecture & Components


### 2.1 `BaseAgent` (`backend/app/agents/base_agent.py`)
- **Purpose:** Abstract base class that all future specialist agents (`TempleInformationAgent`, `ComparisonAgent`, `CrossTempleAggregationAgent`, `ItineraryPlanningAgent`, `CompleteListRetriever`) will inherit from.
- **Contract:**
  - `name: str` — Canonical identifier of the agent.
  - `description: str` — Purpose and capability summary of the agent.
  - `_run(state: AgentState) -> AgentResponse` — Abstract method implemented by subclasses to perform domain logic.
  - `execute(state: AgentState) -> AgentResponse` — Concrete template method executing the standard lifecycle.

### 2.2 Execution Lifecycle
When `agent.execute(state)` is called:
1. **Active Agent Marking:** Sets `state.active_agent = self.name` (restoring previous active agent in `finally`).
2. **Timing Initialization:** Starts high-resolution clock (`time.perf_counter()`).
3. **Delegation:** Invokes subclass `_run(state)` within an isolated `try...except` block.
4. **Latency Measurement:** Computes elapsed time in milliseconds (`latency_ms`).
5. **Response Normalization:** Ensures `agent_name` and `latency_ms` are populated.
6. **Execution Trace Logging:** Appends an `ExecutionTrace` record to `state.execution_trace` with status `"completed"` or `"failed"`.
7. **Intermediate State Storage:** Caches the output in `state.intermediate_results[self.name]`.
8. **Standardized Error Containment:** Unhandled exceptions are caught, logged with stack trace, and converted into an `AgentResponse(success=False, error=...)` without crashing the coordinator.

### 2.3 `AgentState` (`backend/app/agents/state.py`)
- **Purpose:** Mutable in-memory state object passed through the coordinator and specialist agents for a given user turn.
- **Attributes:**
  - `session_id: str` — Session identifier for conversational continuity.
  - `user_query: str` — The raw or sanitized user input string.
  - `detected_intents: List[str]` — Intents detected during query analysis.
  - `active_agent: Optional[str]` — Name of the agent currently executing.
  - `intermediate_results: Dict[str, Any]` — Structured map of outputs from each specialist agent.
  - `final_response: Optional[AgentResponse]` — Unified response produced by the aggregator.
  - `execution_trace: List[ExecutionTrace]` — Chronological trace log of all pipeline steps.
  - `add_trace(...)` — Helper method to append structured trace entries.

### 2.4 `ExecutionTrace` (`backend/app/agents/state.py`)
- **Purpose:** In-memory telemetry unit recording individual agent and pipeline steps.
- **Attributes:**
  - `agent_name: str` — Executing agent.
  - `step_name: str` — Step name (e.g. `"execute"`, `"retrieve"`, `"format"`).
  - `status: str` — `"started"`, `"completed"`, or `"failed"`.
  - `latency_ms: Optional[float]` — Elapsed execution time in milliseconds.
  - `error: Optional[str]` — Error message if the step encountered an issue.
  - `metadata: Dict[str, Any]` — Custom step metadata for debugging.

### 2.5 `AgentRouteDecision` (`backend/app/agents/state.py`)
- **Purpose:** Data contract emitted by future `AgentRouter` to instruct the `MasterAgentCoordinator`.
- **Attributes:**
  - `selected_agent: str` — Chosen target agent.
  - `intent: str` — Classified user intent category.
  - `confidence: float` — Confidence score (0.0 to 1.0).
  - `reason: str` — Human-readable explanation of why this route was selected.
  - `metadata: Dict[str, Any]` — Context parameters extracted during routing.

### 2.6 `AgentResponse` (`backend/app/agents/state.py`)
- **Purpose:** Standardized output format returned by every agent.
- **Attributes:**
  - `success: bool` — True if task succeeded without unhandled errors.
  - `agent_name: str` — Originating agent.
  - `answer: str` — Formatted natural language or Markdown response text.
  - `sources: List[str]` — Source files or citations used.
  - `metadata: Dict[str, Any]` — Domain-specific metrics or structured data.
  - `error: Optional[str]` — Error details when `success=False`.
  - `latency_ms: Optional[float]` — Measured execution duration.

---

## 3. Dependency & Layering Rules

```text
┌────────────────────────────────────────────────────────┐
│                      Agent Layer                       │
│  BaseAgent, AgentState, AgentRouteDecision, Trace      │
└──────────────────────────┬─────────────────────────────┘
                           │ (Phase B Facade)
                           ▼
┌────────────────────────────────────────────────────────┐
│                        RAGTool                         │
└──────────────────────────┬─────────────────────────────┘
                           │ (Read-Only)
                           ▼
┌────────────────────────────────────────────────────────┐
│             FROZEN RAG / Storage Foundation            │
│  PilgrimRAGService, Qdrant, BM25, KnowledgeRegistry   │
└────────────────────────────────────────────────────────┘
```

- **Strict Decoupling:** `BaseAgent` and `AgentState` contain **zero** imports of:
  - `qdrant_client`
  - `rank_bm25`
  - `sentence_transformers`
  - `LocalQdrantManager`
  - `LocalBM25Manager`
  - Raw JSON files from temple directories.
- All future database or index access will pass strictly through `RAGTool` (Phase B).

---

## 4. Verification and Test Results

### 4.1 Phase A Unit Tests (`tests/test_agent_contracts.py`)
- Total tests: **10**
- Passing: **10 (100%)**
- Duration: **0.12 seconds**

| Test Name | Verification Focus | Result |
| :--- | :--- | :--- |
| `test_1_base_agent_is_abstract` | Confirms `BaseAgent` cannot be directly instantiated | **PASS** |
| `test_2_dummy_agent_inheritance` | Confirms concrete subclass can inherit from `BaseAgent` | **PASS** |
| `test_3_and_4_successful_execution_and_response` | Confirms `execute()` lifecycle produces valid `AgentResponse` | **PASS** |
| `test_5_execution_trace_recorded` | Confirms execution trace is appended to `AgentState` | **PASS** |
| `test_6_execution_timing_recorded` | Confirms latency is measured and recorded in milliseconds | **PASS** |
| `test_7_agent_error_captured_consistently` | Confirms exceptions are caught and formatted into `AgentResponse(success=False)` | **PASS** |
| `test_8_agent_state_creation` | Confirms `AgentState` defaults and custom properties | **PASS** |
| `test_9_agent_route_decision_creation` | Confirms `AgentRouteDecision` instantiation and serialization | **PASS** |
| `test_10_agent_response_creation` | Confirms `AgentResponse` fields and dictionary dumping | **PASS** |
| `test_11_active_agent_lifecycle_transition` | Confirms `active_agent` transition and restoration during execution | **PASS** |

### 4.2 RAG Baseline Integrity Check (`tests/test_query_planner.py`)
- Total tests: **40**
- Passing: **40 (100%)**
- Duration: **0.13 seconds**
- **Outcome:** RAG foundation remains completely untouched and functional.

---

## 5. File Modification Audit

```text
Files Created:
- backend/app/agents/base_agent.py
- backend/app/agents/state.py
- tests/test_agent_contracts.py
- docs/PHASE_A_AGENT_CONTRACTS.md

Protected Files Modified:
- NONE (0 files changed in backend/app/rag/, qdrant_db/, bm25_index/, temple JSONs, .env, frontend)
```
