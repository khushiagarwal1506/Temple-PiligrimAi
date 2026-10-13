import json
import re
import sys
from pathlib import Path
import pytest

root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agents.state import AgentState
from app.agents.budget_agent import BudgetAgent, extract_budget_parameters
from app.agents.itinerary_planning_agent import ItineraryPlanningAgent
from app.agents.router import AgentRouter
from app.agents.multi_intent_pipeline import SequentialMultiIntentPipeline
from app.agents.response_aggregator import ResponseAggregator


class MockDiskRAGTool:
    """Mock RAG tool that directly loads real JSON datasets from the filesystem."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root

    def get_temple_category_data(self, temple: str, category: str):
        t_dir = self.workspace_root / temple.lower().strip()
        fpath = t_dir / f"{category}.json"
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def get_complete_parent_text(self, temple: str, category: str):
        data = self.get_temple_category_data(temple, category)
        if data:
            return json.dumps(data)
        return None

    def get_registered_temples(self):
        return [
            "Ahobilam", "Badrachalam", "Balkampet", "Basara", "Birla Mandir",
            "Chilkur", "Dharmapuri", "Edupayal", "Jogulamba(new)", "Kaleshvaram",
            "Karmanghat", "Medaram", "Ramappa", "Srisaila", "Thousand Pillar",
            "Ttd", "Ujjaini", "Veera Satyanarayana", "Vemulavada", "Yadadri"
        ]

    def get_registered_categories(self):
        return [
            "accommodation", "contacts", "crowd", "donations", "elderly_ppl",
            "emergency_nos", "facilities", "faq", "festivals", "gallery",
            "maps", "nearby_attractions", "prasadam", "restaurants", "river_bank",
            "rules", "sevas", "sub_temples", "temple", "timings", "travel"
        ]


@pytest.fixture
def mock_rag():
    return MockDiskRAGTool(root_dir)


@pytest.fixture
def budget_agent(mock_rag):
    return BudgetAgent(rag_tool=mock_rag)


@pytest.fixture
def itinerary_agent(mock_rag):
    return ItineraryPlanningAgent(rag_tool=mock_rag)


@pytest.fixture
def multi_intent_pipeline(budget_agent, itinerary_agent):
    router = AgentRouter()
    router.register_agent("itinerary_planning")
    router.register_agent("budget")
    agents = {
        "itinerary_planning": itinerary_agent,
        "budget": budget_agent
    }
    return SequentialMultiIntentPipeline(
        router=router,
        agents=agents,
        aggregator=ResponseAggregator()
    )


def test_chilkur_parameter_extraction_with_budget():
    """Verify parameters are extracted accurately for Chilkur 3-day query with budget 3400."""
    q = "plan a 3 day trip to chilkur balaji with a budget of 3400"
    params = extract_budget_parameters(q)
    assert params["days"] == 3
    assert params["nights"] == 2
    assert params["budget_limit"] == 3400.0
    assert params["origin_assumed"] is True
    assert params["origin"] == "Hyderabad"


def test_chilkur_unknown_origin_excludes_transit_from_baseline(budget_agent):
    """Unknown origin must NOT charge local transit or assume Hyderabad starting location."""
    q = "plan a 3 day trip to chilkur balaji with a budget of 3400"
    state = AgentState(
        session_id="test_origin",
        user_query=q,
        structured_constraints={
            "days": 3,
            "nights": 2,
            "pax": 1,
            "origin_assumed": True,
            "origin": "Hyderabad",
            "budget_limit": 3400.0
        }
    )
    resp = budget_agent.execute(state)
    assert resp.success is True
    ans = resp.answer

    # Transit should be labeled as preliminary unstated / pending starting location
    assert "Preliminary Non-Transport Estimate" in ans
    assert "Intercity Transit (Starting Location Unspecified)" in ans
    # Must NOT charge ₹80 local transit in baseline total
    assert "Round-Trip Local Transit" not in ans


def test_chilkur_budget_aware_accommodation_avoids_luxury_resort(budget_agent):
    """Budget agent must NOT select Golkonda Resorts (₹17,000) when budget is ₹3,400."""
    q = "plan a 3 day trip to chilkur balaji with a budget of 3400"
    state = AgentState(
        session_id="test_budget_accom",
        user_query=q,
        structured_constraints={
            "days": 3,
            "nights": 2,
            "pax": 1,
            "origin_assumed": True,
            "budget_limit": 3400.0
        }
    )
    resp = budget_agent.execute(state)
    assert resp.success is True
    ans = resp.answer

    # Golkonda Resorts (₹17,000) must NOT be the selected baseline accommodation
    assert "Accommodation (Golkonda Resorts & Spa)" not in ans
    # Selected baseline must be budget lodging with unlisted tariff
    assert "Local Highway Lodges" in ans

    # Quantified baseline total must be around ₹750 (Meals ₹600 + Local Transit ₹150 + Darshan ₹0)
    assert "₹750" in ans
    # Remaining safety buffer must be ₹2,650 (₹3,400 - ₹750)
    assert "₹2,650" in ans
    # Golkonda Resorts must be labeled as exceeding budget
    assert "Exceeds ₹3,400 Budget" in ans


def test_chilkur_itinerary_no_hallucinated_devasthanam_stay(itinerary_agent):
    """Itinerary must NOT invent a Devasthanam Rest House for Chilkur Balaji Temple."""
    q = "plan a 3 day trip to chilkur balaji with a budget of 3400"
    state = AgentState(
        session_id="test_itin_stay",
        user_query=q,
        structured_constraints={
            "duration_days": 3.0,
            "days": 3,
            "nights": 2,
            "budget_limit": 3400.0
        }
    )
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    ans = resp.answer

    # Must NOT claim non-existent Devasthanam Rest House / Choultry at Chilkur
    assert "Devasthanam Rest House / Choultry (Chilkur Balaji Temple)" not in ans
    # Must use verified lodging (e.g. Local Highway Lodges)
    assert "Local Highway Lodges" in ans


def test_chilkur_itinerary_day2_attractions_and_dining_rotation(itinerary_agent):
    """Day 2 must explore verified nearby attractions and rotate restaurants across days."""
    q = "plan a 3 day trip to chilkur balaji with a budget of 3400"
    state = AgentState(
        session_id="test_itin_attractions",
        user_query=q,
        structured_constraints={
            "duration_days": 3.0,
            "days": 3,
            "nights": 2,
            "budget_limit": 3400.0
        }
    )
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    ans = resp.answer

    # Day 2 should visit Kamaldham Mandir / Lotus Temple or Osman Sagar
    assert "Kamaldham Mandir" in ans or "Lotus Temple" in ans
    assert "Osman Sagar" in ans

    # Day 3 should have concluding darshan and explore Mrugavani National Park
    assert "Mrugavani National Park" in ans

    # Dining should rotate (Iraa Restaurant and Balaji Chilkur Family Dhaba)
    assert "Iraa Restaurant" in ans
    assert "Balaji Chilkur Family Dhaba" in ans


def test_chilkur_end_to_end_pipeline_consistency(multi_intent_pipeline):
    """End-to-end multi-intent pipeline generates consistent budget and itinerary."""
    q = "plan a 3 day trip to chilkur balaji with a budget of 3400"
    state = AgentState(
        session_id="test_pipeline_e2e",
        user_query=q
    )
    resp = multi_intent_pipeline.execute(state)
    assert resp.success is True
    ans = resp.answer

    # Budget & Itinerary consistency checks
    assert "Devasthanam Rest House / Choultry (Chilkur Balaji Temple)" not in ans
    assert "Accommodation (Golkonda Resorts & Spa)" not in ans
    assert "Local Highway Lodges" in ans
    assert "Preliminary Non-Transport Estimate" in ans
    assert "₹750" in ans
    assert "₹2,650" in ans
