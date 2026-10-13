"""
Phase F - MasterAgentCoordinator Certification Tests

Verifies end-to-end multi-agent orchestration across Phase A-E components:
1. Coordinator initializes with router, aggregator, and agent registry.
2. Real Yadadri factual query executes through Coordinator -> Router -> Specialist -> Aggregator.
3. Real Dharmapuri factual query executes successfully.
4. Ambiguous queries trigger clarification without RAG execution or random temple guessing.
5. Explicit unsupported temples (e.g., Golden Temple) produce clean unsupported response.
6. Future comparison intent is caught and rejected gracefully without executing nonexistent agent.
7. Future itinerary intent is caught and rejected gracefully without executing nonexistent agent.
8. AgentState is updated with final_response, intermediate_results, and detected_intents.
9. Execution traces from coordinator and specialist agents are chronologically recorded.
10. Specialist failures are preserved without fabricating replacement answers.
11. Aggregator integration verifies all responses pass through ResponseAggregator.aggregate().
12. Determinism check verifies identical routing, state, and structure across runs.
13. Strict dependency boundary audit confirms zero raw database, reranker, or LLM imports.

Integration tests against live certified Phase A-E stack — zero data mutation.
"""

import sys
from pathlib import Path
import pytest

# Ensure backend is in sys.path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agents.base_agent import BaseAgent
from app.agents.state import AgentState, AgentResponse
from app.agents.router import AgentRouter
from app.agents.response_aggregator import ResponseAggregator
from app.agents.temple_information_agent import TempleInformationAgent
from app.agents.coordinator import MasterAgentCoordinator


class DummyFailingAgent(BaseAgent):
    """Test double simulating a specialist agent failure."""

    def __init__(self, name="temple_information", description="Failing agent double"):
        super().__init__(name=name, description=description)

    def _run(self, state: AgentState) -> AgentResponse:
        raise RuntimeError("Controlled specialist agent execution failure")


@pytest.fixture(scope="module")
def coordinator():
    return MasterAgentCoordinator()


def test_1_coordinator_initialization(coordinator):
    """Verify MasterAgentCoordinator initializes router, aggregator, and active agents registry."""
    assert coordinator is not None
    assert isinstance(coordinator.router, AgentRouter)
    assert isinstance(coordinator.aggregator, ResponseAggregator)
    assert "temple_information" in coordinator.agents
    assert isinstance(coordinator.agents["temple_information"], TempleInformationAgent)


def test_2_yadadri_end_to_end_flow(coordinator):
    """Verify full end-to-end execution: Coordinator -> Router -> TempleInformationAgent -> Aggregator."""
    state = AgentState(user_query="What are the timings of Yadadri temple?")
    res = coordinator.process(state)

    assert isinstance(res, AgentResponse)
    assert res.success is True
    assert len(res.answer) > 20
    assert "4:00" in res.answer or "timings" in res.answer.lower() or "open" in res.answer.lower()
    assert len(res.sources) > 0
    assert any("yadadri" in s.lower() for s in res.sources)
    assert state.final_response == res


def test_3_dharmapuri_flow(coordinator):
    """Verify factual query for Dharmapuri executes cleanly through the coordinator."""
    state = AgentState(user_query="What sevas are available at Dharmapuri temple?")
    res = coordinator.process(state)

    assert isinstance(res, AgentResponse)
    assert res.success is True
    assert len(res.answer) > 20
    assert len(res.sources) > 0
    assert any("dharmapuri" in s.lower() for s in res.sources)


def test_4_ambiguous_query(coordinator):
    """Verify ambiguous query lacking temple name returns clarification without querying RAG."""
    state = AgentState(user_query="What are the temple timings?")
    res = coordinator.process(state)

    assert res.success is False
    assert "which temple" in res.answer.lower()
    assert len(res.sources) == 0
    assert "clarification" in state.detected_intents


def test_5_unsupported_temple(coordinator):
    """Verify explicitly named non-registered temple returns controlled unsupported response."""
    state = AgentState(user_query="What are the timings of Golden Temple in Amritsar?")
    res = coordinator.process(state)

    assert res.success is False
    assert "don't have" in res.answer.lower() or "knowledge base" in res.answer.lower()
    assert len(res.sources) == 0
    assert "unsupported" in state.detected_intents


def test_6_unsupported_future_comparison_intent(coordinator):
    """Verify future comparison intent is rejected safely without calling nonexistent agent."""
    state = AgentState(user_query="Compare Yadadri and Dharmapuri temples.")
    res = coordinator.process(state)

    assert res.success is False
    assert "not yet available" in res.answer.lower()
    assert "comparison" in state.detected_intents


def test_7_unsupported_future_itinerary_intent(coordinator):
    """Verify future itinerary intent is rejected safely without calling nonexistent agent."""
    state = AgentState(user_query="Plan a trip to Yadadri temple.")
    res = coordinator.process(state)

    assert res.success is False
    assert "not yet available" in res.answer.lower()
    assert "itinerary_planning" in state.detected_intents


def test_8_state_updates(coordinator):
    """Verify AgentState is properly populated with intermediate results and final response."""
    state = AgentState(user_query="What are the timings of Yadadri temple?")
    res = coordinator.process(state)

    assert state.final_response == res
    assert "route_decision" in state.intermediate_results
    assert "temple_information" in state.intermediate_results
    assert state.detected_intents == ["temple_information"]


def test_9_execution_trace(coordinator):
    """Verify execution traces from both coordinator and specialist are recorded chronologically."""
    state = AgentState(user_query="What are the timings of Yadadri temple?")
    coordinator.process(state)

    traces = state.execution_trace
    assert len(traces) >= 2

    agent_names = [t.agent_name for t in traces]
    assert "coordinator" in agent_names
    assert "temple_information" in agent_names

    # Coordinator route step followed by dispatch
    step_names = [t.step_name for t in traces]
    assert "route" in step_names
    assert "execute" in step_names


def test_10_specialist_failure():
    """Verify specialist failure is handled safely without fabricating answers."""
    failing_agent = DummyFailingAgent()
    coord = MasterAgentCoordinator(agents={"temple_information": failing_agent})

    state = AgentState(user_query="What are the timings of Yadadri temple?")
    res = coord.process(state)

    assert res.success is False
    assert res.answer == "" or "could not retrieve details" in res.answer.lower()
    assert "controlled specialist agent execution failure" in res.error.lower()


def test_11_aggregator_integration(coordinator):
    """Verify response passes through ResponseAggregator and metadata is attached."""
    state = AgentState(user_query="What are the timings of Yadadri temple?")
    res = coordinator.process(state)

    # Aggregator metadata fields must be present
    assert "agent_count" in res.metadata
    assert "aggregation_latency_ms" in res.metadata
    assert "coordinator_latency_ms" in res.metadata


def test_12_deterministic_repeated_execution(coordinator):
    """Verify repeated executions produce deterministic routing and trace structures."""
    q = "What are the timings of Yadadri temple?"

    states = [AgentState(user_query=q) for _ in range(3)]
    responses = [coordinator.process(s) for s in states]

    # Verify routing and structural determinism
    for s in states:
        assert s.detected_intents == ["temple_information"]
        assert s.intermediate_results["route_decision"].selected_agent == "temple_information"
        assert len(s.execution_trace) == len(states[0].execution_trace)

    for r in responses:
        assert r.success is True
        assert len(r.sources) > 0


def test_13_dependency_boundary():
    """Verify coordinator.py contains zero direct imports of databases, rerankers, or models."""
    file_path = root_dir / "backend" / "app" / "agents" / "coordinator.py"
    content = file_path.read_text(encoding="utf-8")

    forbidden = [
        "qdrant_client",
        "rank_bm25",
        "sentence_transformers",
        "LocalQdrantManager",
        "LocalBM25Manager",
        "PilgrimRAGService",
        "KnowledgeRegistry",
        "RAGTool",
        "Ollama"
    ]

    for item in forbidden:
        assert item not in content, f"Forbidden import '{item}' detected in coordinator.py!"
