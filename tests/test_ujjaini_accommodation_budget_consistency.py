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
from app.agents.budget_agent import (
    BudgetAgent,
    extract_budget_parameters,
    parse_price_text,
    normalize_price_display_text,
    classify_and_enrich_accommodation,
)
from app.agents.itinerary_planning_agent import ItineraryPlanningAgent
from app.agents.router import AgentRouter
from app.agents.multi_intent_pipeline import SequentialMultiIntentPipeline
from app.agents.response_aggregator import ResponseAggregator


class MockDiskRAGTool:
    """Loads authoritative JSON datasets directly from filesystem."""

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


def test_minerva_grand_price_parsing_and_normalization():
    """Verify Minerva Grand tariff typo '₹4,00 to ₹5000' is normalized without altering dataset."""
    raw = "₹4,00 to ₹5000 per night"
    normalized = normalize_price_display_text(raw)
    assert normalized == "₹4,000 to ₹5,000 per night"

    min_v, max_v, unit, is_free, is_range, is_quant, notes = parse_price_text(raw, category="accommodation")
    assert min_v == 4000.0
    assert max_v == 5000.0
    assert is_range is True
    assert is_quant is True
    assert "₹4,000 to ₹5,000" in notes

    # Verify enrichment directly from raw Ujjaini record
    minerva_raw = {
        "name": "Minerva Grand Hotel - Secunderabad",
        "type": "3-star hotel",
        "distance": "600 metres",
        "price": "₹4,00 to ₹5000 per night",
        "description": "Rating: 4.1. Automated Valet Parking."
    }
    enriched = classify_and_enrich_accommodation(minerva_raw, temple_canonical="ujjaini")
    assert enriched["price"] == "₹4,000 to ₹5,000 per night"
    assert "₹4,00 to" not in enriched["price"]


def test_ujjaini_accommodation_consistency_between_itinerary_and_budget(budget_agent, itinerary_agent):
    """Budget and Itinerary agents must both select Hotel Maya Deluxe for 3.5-day Ujjaini trip."""
    q = "plan a 3.5 day trip ujjaini for 2 people with 3000 rupees"
    params = extract_budget_parameters(q)
    assert params["days"] == 3.5
    assert params["nights"] == 3
    assert params["pax"] == 2
    assert params["budget_limit"] == 3000.0

    b_state = AgentState(
        session_id="test_ujjaini_b",
        user_query=q,
        structured_constraints=dict(params)
    )
    b_resp = budget_agent.execute(b_state)
    assert b_resp.success is True
    assert "Hotel Maya Deluxe" in b_resp.answer
    assert "Minerva Grand" not in b_resp.metadata.get("selected_plan", {}).get("line_items", [{}])[2].get("item_name", "")

    i_state = AgentState(
        session_id="test_ujjaini_i",
        user_query=q,
        structured_constraints=dict(params)
    )
    i_resp = itinerary_agent.execute(i_state)
    assert i_resp.success is True
    # Overnight entries for Days 1, 2, and 3 must select Hotel Maya Deluxe
    assert "Hotel Maya Deluxe" in i_resp.answer
    # Minerva Grand should NOT be the overnight stay
    assert "Selected stay: **Minerva Grand Hotel" not in i_resp.answer
    # Tariff should NOT be falsely claimed as unlisted
    assert "tariff unlisted in static records" not in i_resp.answer
    assert "₹900 to ₹1,500" in i_resp.answer


def test_ujjaini_multi_intent_pipeline_integrated_flow(multi_intent_pipeline):
    """Integrated pipeline verifies single consistent accommodation, OVER_BUDGET status, and clean ordering."""
    q = "plan a 3.5 day trip ujjaini for 2 people with 3000 rupees"
    state = AgentState(user_query=q)
    resp = multi_intent_pipeline.execute(state)
    assert resp.success is True

    # 1. Accommodation Consistency
    ans = resp.answer
    assert "Hotel Maya Deluxe" in ans
    assert "Selected stay: **Minerva Grand Hotel" not in ans
    # Check that each overnight line features Hotel Maya Deluxe
    overnights = [line for line in ans.split("\n") if "- **Overnight**:" in line]
    assert len(overnights) == 3
    for ov in overnights:
        assert "Hotel Maya Deluxe" in ov
        assert "Minerva Grand" not in ov

    # 2. Budget Accuracy
    meta = resp.metadata
    assert meta["budget_status"] == "OVER_BUDGET"
    assert meta["shortfall"] == 1450.0
    assert meta["selected_plan"]["total"] == 4450.0
    assert meta["selected_plan"]["budget_limit"] == 3000.0

    # 3. Accommodation options list is kept and Minerva price is normalized
    assert "### 🏨 Accommodation Options & Suggestions" in ans
    assert "₹4,000 to ₹5,000 per night" in ans
    assert "₹4,00 to" not in ans


def test_budget_accuracy_distinguishes_known_estimates_and_pending(budget_agent):
    """Budget must clearly distinguish known costs, estimates, and pending intercity transit."""
    q = "plan a 3.5 day trip ujjaini for 2 people with 3000 rupees"
    state = AgentState(user_query=q)
    resp = budget_agent.execute(state)
    assert resp.success is True

    items = resp.metadata["line_items"]
    # 1. General darshan is known complimentary
    darshan = next(it for it in items if "Darshan" in it["item_name"])
    assert darshan["is_free"] is True
    assert darshan["subtotal_min"] == 0.0

    # 2. Intercity transit is pending starting location (unspecified)
    transit = next(it for it in items if it["category"] == "travel")
    assert transit["provenance"] == "UNAVAILABLE"
    assert "Starting Location Unspecified" in transit["item_name"]

    # 3. Lodging, meals, and local transit are defensible estimates
    lodging = next(it for it in items if it["category"] == "accommodation")
    assert lodging["subtotal_min"] == 2700.0
    assert lodging["subtotal_max"] == 4500.0

    meals = next(it for it in items if it["category"] == "meals")
    assert meals["subtotal_min"] == 1400.0
    assert meals["provenance"] == "ESTIMATE"

    local = next(it for it in items if it["category"] == "local_transit")
    assert local["subtotal_min"] == 350.0
    assert local["provenance"] == "ESTIMATE"
