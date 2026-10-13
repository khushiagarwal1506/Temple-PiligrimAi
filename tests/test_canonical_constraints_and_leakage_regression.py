import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.agents.canonical_constraints import (
    extract_canonical_constraints,
    parse_budget_value,
    is_trip_followup
)

import requests

client = TestClient(app)

def post_chat(payload):
    try:
        r = requests.post("http://127.0.0.1:8000/chat", json=payload, timeout=60)
        if r.status_code == 200:
            return r
    except Exception:
        pass
    return client.post("/chat", json=payload)


def test_parse_budget_value_robustness():
    """Verify deterministic budget value parser across digits, k-suffixes, and words."""
    assert parse_budget_value("2k") == 2000.0
    assert parse_budget_value("2.5k") == 2500.0
    assert parse_budget_value("3.8k") == 3800.0
    assert parse_budget_value("10k") == 10000.0
    assert parse_budget_value("2 thousand") == 2000.0
    assert parse_budget_value("3800") == 3800.0
    assert parse_budget_value("8,000") == 8000.0
    assert parse_budget_value("eight thousand") == 8000.0
    assert parse_budget_value("two thousand") == 2000.0


def test_adversarial_1_leakage_3800_followed_by_2000():
    """Requirement D.1: ₹3,800 request followed by ₹2,000 request: second response must use ₹2,000."""
    session_id = "test_adv_sess_3800_2000"

    # Request 1
    r1 = post_chat({
        "question": "plan a 3 day trip to birla mandir for 2 adults and 1 child with a budget of 3800",
        "session_id": session_id
    })
    assert r1.status_code == 200
    meta1 = r1.json().get("metadata", {})
    assert meta1.get("budget_limit") == 3800.0

    # Request 2
    r2 = post_chat({
        "question": "plan a 6 day trip to edupayala, rammapa and thousand pillar for 4 people with a budget of 2k",
        "session_id": session_id
    })
    assert r2.status_code == 200
    data2 = r2.json()
    meta2 = data2.get("metadata", {})

    # Second response MUST use ₹2,000 and NOT inherit ₹3,800
    assert meta2.get("budget_limit") == 2000.0
    assert meta2.get("pax") == 4
    assert meta2.get("duration_days") == 6.0
    assert "- **Budget limit**: ₹2,000" in data2.get("answer", "")
    assert "₹3,800" not in data2.get("answer", "")


def test_adversarial_2_leakage_2000_followed_by_8000():
    """Requirement D.2: ₹2,000 request followed by ₹8,000 request: second response must use ₹8,000."""
    session_id = "test_adv_sess_2000_8000"

    r1 = post_chat({
        "question": "plan a 2 day trip to yadadri for 2 people with a budget of 2k",
        "session_id": session_id
    })
    assert r1.status_code == 200
    assert r1.json()["metadata"]["budget_limit"] == 2000.0

    r2 = post_chat({
        "question": "plan a 4.5 day trip to ttd and dharmapuri for 2 people with 8000 rupees",
        "session_id": session_id
    })
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["metadata"]["budget_limit"] == 8000.0
    assert "- **Budget limit**: ₹8,000" in data2["answer"]


def test_adversarial_3_same_destinations_updated_budget():
    """Requirement D.3: ₹8,000 request followed by ₹3,800 for the same destinations and duration: second response must use ₹3,800."""
    session_id = "test_adv_sess_same_dest_update_budget"

    r1 = post_chat({
        "question": "plan a 3 day trip to birla mandir for 2 people with a budget of 8000",
        "session_id": session_id
    })
    assert r1.status_code == 200
    assert r1.json()["metadata"]["budget_limit"] == 8000.0

    r2 = post_chat({
        "question": "plan a 3 day trip to birla mandir for 2 people with a budget of 3800",
        "session_id": session_id
    })
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["metadata"]["budget_limit"] == 3800.0
    assert "- **Budget limit**: ₹3,800" in data2["answer"]


def test_adversarial_4_party_size_two_to_four():
    """Requirement D.4: Two-person request followed by a four-person request: party size must update to four."""
    session_id = "test_adv_sess_pax_update"

    r1 = post_chat({
        "question": "plan a 3 day trip to yadadri for 2 people with 5000 rupees",
        "session_id": session_id
    })
    assert r1.status_code == 200
    assert r1.json()["metadata"]["pax"] == 2

    r2 = post_chat({
        "question": "plan a 3 day trip to yadadri for 4 people with 5000 rupees",
        "session_id": session_id
    })
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["metadata"]["pax"] == 4
    assert "- **Travellers**: 4 pilgrim(s)" in data2["answer"]


def test_adversarial_5_adults_and_children_preserved():
    """Requirement D.5: Two adults plus one child: total travellers must be three, with adult/child counts preserved separately."""
    session_id = "test_adv_sess_adults_children"

    r = post_chat({
        "question": "plan a 3 day trip to birla mandir for 2 adults and 1 child with a budget of 3800",
        "session_id": session_id
    })
    assert r.status_code == 200
    data = r.json()
    meta = data.get("metadata", {})
    sc = meta.get("structured_constraints", {})

    assert sc.get("pax") == 3
    assert sc.get("adults") == 2
    assert sc.get("children") == 1
    assert meta.get("pax") == 3
    assert meta.get("adults") == 2
    assert meta.get("children") == 1
    assert "3 pilgrim(s)" in data.get("answer", "")
    assert "2 adults" in data.get("answer", "")
    assert "1 child" in data.get("answer", "")


def test_adversarial_6_duration_six_to_4_point_5():
    """Requirement D.6: Six-day request followed by a 4.5-day request: duration and nights must be recalculated."""
    session_id = "test_adv_sess_duration_update"

    r1 = post_chat({
        "question": "plan a 6 day trip to dharmapuri for 2 people with 5000 rupees",
        "session_id": session_id
    })
    assert r1.status_code == 200
    assert r1.json()["metadata"]["duration_days"] == 6.0
    assert r1.json()["metadata"]["nights"] == 5

    r2 = post_chat({
        "question": "plan a 4.5 day trip to dharmapuri for 2 people with 5000 rupees",
        "session_id": session_id
    })
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["metadata"]["duration_days"] == 4.5
    assert data2["metadata"]["nights"] == 4


def test_adversarial_7_new_destination_does_not_reuse_previous():
    """Requirement D.7: A new destination request must not reuse previous destination selections."""
    session_id = "test_adv_sess_new_dest"

    r1 = post_chat({
        "question": "plan a 3 day trip to birla mandir for 2 people with 3800 rupees",
        "session_id": session_id
    })
    assert r1.status_code == 200

    r2 = post_chat({
        "question": "plan a 6 day trip to edupayala, rammapa and thousand pillar for 4 people with a budget of 2k",
        "session_id": session_id
    })
    assert r2.status_code == 200
    data2 = r2.json()
    destinations = data2["metadata"].get("destinations", [])
    assert "Birla Mandir" not in destinations
    assert any("Edupayal" in d for d in destinations)
    assert any("Ramappa" in d for d in destinations)


def test_adversarial_8_genuine_followup_increases_budget():
    """Requirement D.8: A genuine follow-up such as 'increase the budget to ₹5,000' must retain the relevant trip while changing only the budget."""
    session_id = "test_adv_sess_genuine_followup"

    r1 = post_chat({
        "question": "plan a 3 day trip to birla mandir for 2 adults and 1 child with a budget of 3800",
        "session_id": session_id
    })
    assert r1.status_code == 200
    assert r1.json()["metadata"]["budget_limit"] == 3800.0

    r2 = post_chat({
        "question": "increase the budget to 5000",
        "session_id": session_id
    })
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["metadata"]["budget_limit"] == 5000.0
    assert data2["metadata"]["pax"] == 3
    assert data2["metadata"]["duration_days"] == 3.0
    assert "- **Budget limit**: ₹5,000" in data2["answer"]


def test_adversarial_9_unrelated_new_request_does_not_inherit():
    """Requirement D.9: An unrelated new request must not inherit previous trip parameters."""
    session_id = "test_adv_sess_unrelated"

    r1 = post_chat({
        "question": "plan a 3 day trip to birla mandir for 2 people with a budget of 3800",
        "session_id": session_id
    })
    assert r1.status_code == 200

    r2 = post_chat({
        "question": "what are the temple timings of basara?",
        "session_id": session_id
    })
    assert r2.status_code == 200
    data2 = r2.json()
    # Should be answered by temple_information, not inherit 3800 budget table
    assert data2.get("agent_name") in ("temple_information", "coordinator")
    assert "Budget limit" not in data2.get("answer", "")


def test_adversarial_11_response_table_itinerary_synchronization():
    """Requirement D.11: The final structured response, budget table, itinerary, accommodation selection, and display must agree with the canonical request."""
    session_id = "test_adv_sess_sync_check"

    r = post_chat({
        "question": "plan a 6 day trip to edupayala, rammapa and thousand pillar for 4 people with a budget of 2k",
        "session_id": session_id
    })
    assert r.status_code == 200
    data = r.json()
    meta = data["metadata"]
    sc = meta.get("structured_constraints", {})

    assert sc.get("budget_limit") == 2000.0
    assert sc.get("pax") == 4
    assert sc.get("duration_days") == 6.0
    assert sc.get("nights") == 5

    # Check that answer table displays 4 travellers and 6.0 days
    assert "4 pilgrim(s) × 6.0 day(s)" in data["answer"]
    assert "- **Budget limit**: ₹2,000" in data["answer"]


def test_adversarial_12_cross_session_and_alternating_isolation():
    """Requirement D.12: Run alternating requests in the same session and separate sessions to detect cross-request and cross-session leakage."""
    sess_a = "test_isolation_sess_A"
    sess_b = "test_isolation_sess_B"

    # Turn 1: Sess A (3800)
    ra1 = post_chat({
        "question": "plan a 3 day trip to birla mandir for 2 people with a budget of 3800",
        "session_id": sess_a
    })
    assert ra1.json()["metadata"]["budget_limit"] == 3800.0

    # Turn 1: Sess B (2000)
    rb1 = post_chat({
        "question": "plan a 6 day trip to edupayala for 4 people with a budget of 2k",
        "session_id": sess_b
    })
    assert rb1.json()["metadata"]["budget_limit"] == 2000.0

    # Turn 2: Sess A (new trip with 5000)
    ra2 = post_chat({
        "question": "plan a 2 day trip to yadadri for 2 people with a budget of 5000",
        "session_id": sess_a
    })
    assert ra2.json()["metadata"]["budget_limit"] == 5000.0
    assert "- **Budget limit**: ₹5,000" in ra2.json()["answer"]

    # Turn 2: Sess B (new trip with 8000)
    rb2 = post_chat({
        "question": "plan a 4.5 day trip to ttd and dharmapuri for 2 people with 8000 rupees",
        "session_id": sess_b
    })
    assert rb2.json()["metadata"]["budget_limit"] == 8000.0
    assert "- **Budget limit**: ₹8,000" in rb2.json()["answer"]


def test_section_e_ttd_dharmapuri_baseline_and_integrity():
    """Requirement E: Verify TTD + Dharmapuri 4.5 day trip retains ₹8,000 cap, 2 people, ₹4,750 baseline, and coherence."""
    session_id = "test_section_e_ttd_dharmapuri"

    r = post_chat({
        "question": "plan a 4.5 day trip to ttd and dharmapuri for 2 people with 8000 rupees",
        "session_id": session_id
    })
    assert r.status_code == 200
    data = r.json()
    meta = data["metadata"]
    ans = data["answer"]

    # 1. Budget cap remains ₹8,000
    assert meta["budget_limit"] == 8000.0
    assert "- **Budget limit**: ₹8,000" in ans

    # 2. Party size remains two
    assert meta["pax"] == 2
    assert "- **Travellers**: 2 pilgrim(s)" in ans

    # 3. Duration remains 4.5 days and accommodation nights remain synchronized (4 nights)
    assert meta["duration_days"] == 4.5
    assert meta["nights"] == 4

    # 4. Partial baseline ₹4,750 and remainder before transit ₹3,250
    assert "₹4,750" in ans
    assert "₹3,250" in ans

    # 5. Arrival window ~10:30 PM to ~02:30 AM
    assert "10:30 PM" in ans
    assert "02:30 AM" in ans

    # 6. Excursions simplified (Gudem Gutta present, Kawal Tiger Reserve omitted from mandatory schedule)
    assert "Gudem Gutta" in ans

    # 7. Accommodation catalogue pruned
    assert ans.count("Private Hotel") + ans.count("Temple Guest House") + ans.count("Devasthanam") < 15


def test_issue_p_missing_origin_intercity_transit_pending():
    """ISSUE P: Missing origin leaves intercity transport pending instead of silently assuming confirmed travel."""
    session_id = "test_issue_p_origin"
    r = post_chat({
        "question": "plan a 3 day trip to yadadri for 2 people with a budget of 5000",
        "session_id": session_id
    })
    assert r.status_code == 200
    data = r.json()
    ans = data["answer"]
    meta = data["metadata"]

    assert meta.get("origin_assumed", False) is True
    # Table must reflect starting location pending confirmation
    assert "Pending starting location" in ans or "Starting Location Unspecified" in ans
    assert "| Price Type |" not in ans


def test_issue_4_generalized_hotel_and_restaurant_retrieval():
    """ISSUE 4: Multi-result verified hotel/restaurant retrieval and multi-destination preservation."""
    # 1. Chilkur Balaji hotels: multiple verified hotels
    r_hotels = post_chat({
        "question": "hotels near chilkur balaji",
        "session_id": "test_issue_4_hotels"
    })
    assert r_hotels.status_code == 200
    ans_hotels = r_hotels.json()["answer"]
    assert "Golkonda Resorts" in ans_hotels
    assert "Mrugavani Resort" in ans_hotels
    assert "Local Highway Lodges" in ans_hotels

    # 2. Chilkur Balaji restaurants: multiple verified restaurants
    r_rest = post_chat({
        "question": "restaurants near chilkur balaji",
        "session_id": "test_issue_4_rest"
    })
    assert r_rest.status_code == 200
    ans_rest = r_rest.json()["answer"]
    assert "Balaji Chilkur Family Dhaba" in ans_rest
    assert "Iraa Restaurant" in ans_rest
    assert "Local Tiffin Stalls" in ans_rest

    # 3. Multi-destination hotel preservation (Ujjaini + Birla Mandir)
    r_multi = post_chat({
        "question": "hotels for ujjaini mahakali and birla mandir",
        "session_id": "test_issue_4_multi"
    })
    assert r_multi.status_code == 200
    ans_multi = r_multi.json()["answer"]
    assert "Minerva Grand" in ans_multi or "Grand Continent" in ans_multi
    assert "The Superb Hotels" in ans_multi or "Amrutha Castle" in ans_multi


def test_issue_6_expense_table_three_columns_and_day_trip_not_required():
    """ISSUE 6: 3-column expense table without Price Type, and day trip displays 'Not required' for accommodation."""
    session_id = "test_issue_6_table"
    r = post_chat({
        "question": "plan a 1 day trip to yadadri for 2 people with a budget of 2000",
        "session_id": session_id
    })
    assert r.status_code == 200
    ans = r.json()["answer"]

    # Exactly 3 columns in markdown header
    assert "| Expense Category | Selected Option & Basis | Estimated Amount |" in ans
    assert "| Price Type |" not in ans
    assert "| Listed price |" not in ans

    # Day trip accommodation must display 'Not required', NOT 'FREE'
    assert "Day trip — no overnight stay required | **Not required** |" in ans
    assert "no overnight stay required | **FREE** |" not in ans


def test_additional_issue_location_aware_one_day_recommendation():
    """ADDITIONAL ISSUE: Location-aware 1-day temple recommendations."""
    # 1. State-only request asks for starting city and explains round-trip feasibility
    r_state = post_chat({
        "question": "i live in karnataka i have only i day holiday which temple can i visit in 1 day",
        "session_id": "test_addl_state_only"
    })
    assert r_state.status_code == 200
    ans_state = r_state.json()["answer"]
    assert "Which city or district in Karnataka will you be starting from?" in ans_state
    assert "20 certified pilgrimage devasthanams in Telangana" in ans_state
    assert "Jogulamba" in ans_state

    # 2. City-specific request evaluates feasibility and recommends best fit
    r_city = post_chat({
        "question": "i live in raichur karnataka i have only 1 day holiday which temple can i visit in 1 day",
        "session_id": "test_addl_raichur"
    })
    assert r_city.status_code == 200
    ans_city = r_city.json()["answer"]
    assert "Jogulamba Temple, Alampur" in ans_city
    assert "Best Fit" in ans_city
    assert "Round-Trip Feasibility" in ans_city

