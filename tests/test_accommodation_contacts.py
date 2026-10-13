import sys
from pathlib import Path
import pytest

root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agents.budget_agent import (
    BudgetAgent,
    classify_and_enrich_accommodation
)
from app.agents.state import AgentState

def test_classify_and_enrich_temple_managed_with_phone():
    # 1. Temple-managed accommodation with verified phone
    raw_item = {
        "name": "Hemalapuri Sadan",
        "type": "Official Temple-managed Guest House",
        "distance": "~1.1 km",
        "price": "₹300 - ₹1,200 / night",
        "description": "Official temple-managed guest house located close to the main complex. Rooms can be booked by directly contacting the management desk at +91 79899 93291."
    }
    enriched = classify_and_enrich_accommodation(raw_item, temple_canonical="jogulamba(new)")
    assert enriched["management_type"] == "TEMPLE_MANAGED"
    assert enriched["phone"] == "+91 79899 93291"
    assert "Contact temple administration" in enriched["contact_label"]
    assert "+91 79899 93291" in enriched["contact_label"]
    assert enriched["contact_source"] == "Verified temple records"
    assert "Contact devasthanam" not in enriched["price"]


def test_classify_and_enrich_temple_managed_without_phone():
    # 2. Temple-managed accommodation without dedicated phone
    raw_item = {
        "name": "Devasthanam Guest Houses & Choultries",
        "type": "Temple Rest House",
        "distance": "~ 450 m",
        "price": "₹300 - 1200",
        "description": "Maintained directly by the temple administration, offering affordable budget rooms and dormitory halls."
    }
    enriched = classify_and_enrich_accommodation(raw_item, temple_canonical="kaleshvaram")
    assert enriched["management_type"] == "TEMPLE_MANAGED"
    assert enriched["phone"] is None
    assert enriched["contact_label"] == "Contact temple administration"
    assert enriched["contact_source"] == "Temple administrative desk"
    assert "Contact devasthanam" not in enriched["price"]


def test_classify_and_enrich_telangana_tourism_with_phone():
    # 3. Telangana Tourism hotel with verified phone
    raw_item = {
        "name": "Haritha Hotel Bhadrachalam (Telangana Tourism)",
        "type": "State Tourism Hotel",
        "distance": "~1 km",
        "price": "₹1,500 - ₹2,500/night",
        "description": "Managed by TGTDC. Direct mobile: +91 98488 23073. Bookable via the Telangana Tourism Portal."
    }
    enriched = classify_and_enrich_accommodation(raw_item, temple_canonical="badrachalam")
    assert enriched["management_type"] == "GOVERNMENT_TOURISM"
    assert enriched["phone"] == "+91 98488 23073"
    assert enriched["contact_label"] == "Telangana Tourism: +91 98488 23073"
    assert enriched["contact_source"] == "Verified tourism department records"
    assert "Contact devasthanam" not in enriched["contact_label"]


def test_classify_and_enrich_private_lodge_no_contact():
    # 4. Private lodge with NO verified contact details
    raw_item = {
        "name": "Shilpa Lodge",
        "type": "Budget Town Lodge",
        "distance": "1.5 km",
        "best_for": "Budget stay near the bus stand"
    }
    enriched = classify_and_enrich_accommodation(raw_item, temple_canonical="thousand pillar")
    assert enriched["management_type"] == "PRIVATE"
    assert enriched["phone"] is None
    assert enriched["booking_url"] is None
    assert enriched["contact_label"] == "Contact details unavailable"
    assert enriched["contact_source"] == "Unavailable"
    # Price should NOT be "Contact devasthanam"
    assert "Contact devasthanam" not in enriched["price"]
    assert "Tariff" in enriched["price"] or "Unlisted" in enriched["price"]


def test_classify_and_enrich_private_hotel_with_url():
    # 5. Private or Board accommodation with URL
    raw_item = {
        "name": "YTDA Divya Sannidhi Cottages",
        "type": "Premium, modern villa-style cottages",
        "distance": "2.7 to 3.5 km",
        "price": "₹3,000 per night",
        "description": "Bookable online via the official YTDA portal (ytda.in / ytdabooking.in)."
    }
    enriched = classify_and_enrich_accommodation(raw_item, temple_canonical="yadadri")
    assert enriched["booking_url"] is not None
    assert "ytdabooking" in enriched["booking_url"]
    assert any("ytda.in" in u for u in enriched["all_urls"])


def test_classify_and_enrich_availability_status():
    # 6. Availability status preservation (e.g. peak Jathara closed status)
    raw_item = {
        "name": "Haritha Tourism Suite (1 BHK)",
        "type": "State Tourism (Telangana Tourism)",
        "price": "₹1,000 / Day",
        "status": "Bookings Closed during peak Jathara",
        "description": "1 BHK Suite accommodation managed by Telangana Tourism."
    }
    enriched = classify_and_enrich_accommodation(raw_item, temple_canonical="medaram")
    assert enriched["availability_status"] == "Bookings Closed during peak Jathara"


def test_budget_agent_accommodation_output_no_devasthanam_leakage():
    # 7. End-to-end BudgetAgent execution for Thousand Pillar (only private lodges exist)
    agent = BudgetAgent()
    state = AgentState(
        user_query="Plan a 2-day visit to Thousand Pillar Temple from Hyderabad with a budget of ₹5,000."
    )
    resp = agent.process(state)
    assert resp.success is True
    meta = resp.metadata
    accom_opts = meta.get("accommodation_options", {})
    assert len(accom_opts) > 0

    # Ensure none of the private stays at Thousand Pillar have "Contact devasthanam"
    for t_name, opts in accom_opts.items():
        for o in opts:
            assert o["management_type"] != "TEMPLE_MANAGED" or "thousand" not in t_name.lower()
            assert "Contact devasthanam" not in o.get("price", "")
            if o["management_type"] == "PRIVATE":
                assert "Contact devasthanam" not in o.get("contact_label", "")

    # Check answer markdown
    assert "Contact devasthanam" not in resp.answer


def test_over_budget_itinerary_preserved():
    # 8. Over-budget scenario
    agent = BudgetAgent()
    state = AgentState(
        user_query="Plan a 3-day trip to Bhadrachalam and Yadadri with budget of ₹1,000."
    )
    resp = agent.process(state)
    assert resp.success is True
    assert resp.metadata.get("budget_status") == "OVER_BUDGET"
    assert "OVER_BUDGET" in resp.metadata.get("budget_status")
    # Verify shortfall / excess is noted
    assert resp.metadata.get("parameters", {}).get("nights") == 2
