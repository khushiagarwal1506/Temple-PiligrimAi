import sys
from pathlib import Path
import pytest
import re

root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
from app.agents.itinerary_planning_agent import ItineraryPlanningAgent
from app.agents.budget_agent import BudgetAgent
from app.agents.base_agent import AgentState
from app.agents.tools.rag_tool import RAGTool


@pytest.fixture
def rag_tool():
    return RAGTool()


@pytest.fixture
def itinerary_agent(rag_tool):
    return ItineraryPlanningAgent(rag_tool=rag_tool)


@pytest.fixture
def budget_agent(rag_tool):
    return BudgetAgent(rag_tool=rag_tool)


def test_clean_text_preserves_meter_unit(itinerary_agent):
    """Ensure _clean_text does not strip 'm' unit from distance strings."""
    assert itinerary_agent._clean_text("~ 900 m") == "~ 900 m"
    assert itinerary_agent._clean_text("350 m") == "350 m"
    assert itinerary_agent._clean_text("2 km") == "2 km"
    # Still strips stray dangling single letters from OCR
    assert itinerary_agent._clean_text("Some random description q") == "Some random description"


def test_parse_restaurants_preserves_distance_unit(itinerary_agent):
    """Ensure _parse_restaurants does not strip distance units."""
    raw_data = {
        "restaurants": [
            {
                "name": "Sri Hanuman tiffins and meals",
                "distance": "~ 900 m",
                "pricing": "Affordable / Budget-friendly",
                "type": "Private Veg Hotel"
            }
        ]
    }
    parsed = itinerary_agent._parse_restaurants(raw_data, "dharmapuri")
    assert len(parsed["all"]) == 1
    item = parsed["all"][0]
    assert item["name"] == "Sri Hanuman tiffins and meals"
    assert item["distance"] == "~ 900 m"
    assert "900 m" in item["distance"]


def test_ttd_dharmapuri_itinerary_coherence_and_route_verification(itinerary_agent):
    """
    Verify route duration disclosure and 4.5-day schedule coherence
    for Tirumala and Dharmapuri.
    """
    state = AgentState(
        user_query="plan a 4.5 day trip to ttd and dharmapuri for 2 people with 8000 rupees",
        route_params={
            "temples": ["ttd", "dharmapuri"],
            "days": 4.5,
            "pax": 2,
            "budget_limit": 8000,
            "is_budget_request": True,
            "is_itinerary_request": True,
            "origin": "Hyderabad",
            "origin_assumed": True
        }
    )
    resp = itinerary_agent.execute(state)
    assert resp.success is True
    ans = resp.answer

    # 1. Route verification: No unsupported 6-8 hours for ~775 km
    assert "~6–8 hrs connecting transit" not in ans
    assert "unverified road/rail estimate: ~14–18 hrs" in ans
    assert "live GPS routing unverified" in ans

    # 2. Restaurant parsing: Distinguish price vs distance
    assert "(~ 900," not in ans
    assert "(~ 900 m" in ans
    assert "Pricing: Affordable / Budget-friendly" in ans

    # 3. Check-out breakfast at Tirumala, not Dharmapuri
    d3_pos = ans.find("#### Day 3:")
    d4_pos = ans.find("#### Day 4:")
    d3_text = ans[d3_pos:d4_pos]
    assert "complete hotel check-out in Tirumala Venkateswara Temple" in d3_text
    assert "TTD Subsidized Canteens" in d3_text
    # Dharmapuri breakfast must NOT be in the Tirumala check-out morning
    assert "Breakfast at **Sri Hanuman tiffins and meals**" not in d3_text

    # 4. Itinerary coherence: 5 calendar days, 4 nights
    assert "#### Day 1: Tirumala Venkateswara Temple" in ans
    assert "#### Day 2: Tirumala Venkateswara Temple" in ans
    assert "#### Day 3: Dharmapuri Lakshmi Narasimha Swamy Temple" in ans
    assert "#### Day 4: Dharmapuri Lakshmi Narasimha Swamy Temple" in ans
    assert "#### Day 5: Dharmapuri Lakshmi Narasimha Swamy Temple" in ans

    # 5. Overnights match 2 nights each (total 4 nights)
    assert ans.count("Selected stay: **Ampro Rest House") == 2
    assert ans.count("Selected stay: **Community Satralu") == 2
    assert "room availability & tariff confirmation pending" in ans
    assert "not pre-booked" in ans


def test_ttd_dharmapuri_budget_transparency(budget_agent):
    """
    Verify budget remainder labeling, pending intercity transit,
    and distinct darshan rows per temple.
    """
    state = AgentState(
        user_query="plan a 4.5 day trip to ttd and dharmapuri for 2 people with 8000 rupees",
        route_params={
            "temples": ["ttd", "dharmapuri"],
            "days": 4.5,
            "pax": 2,
            "budget_limit": 8000,
            "is_budget_request": True,
            "is_itinerary_request": True,
            "origin": "Hyderabad",
            "origin_assumed": True
        }
    )
    resp = budget_agent.execute(state)
    assert resp.success is True
    ans = resp.answer

    # 1. Budget transparency: Arithmetic remainder before intercity transport
    assert "Arithmetic remainder before intercity transport & unconfirmed costs" in ans
    assert "NOT guaranteed spendable money" in ans
    assert "Preliminary Non-Transport" in ans

    # 2. Intercity transit is pending, not zero or free
    assert "| **Intercity Transit (Starting Location Unspecified)** | Starting location unspecified | **Pending starting location** | Pending confirmation |" in ans

    # 3. Darshan rows identify each temple without duplication
    assert "| **Tirumala Venkateswara Temple — General Darshan** | 2 pilgrim(s) | **FREE** | Listed price (Complimentary) |" in ans
    assert "| **Dharmapuri Lakshmi Narasimha Swamy Temple — General Darshan** | 2 pilgrim(s) | **FREE** | Listed price (Complimentary) |" in ans

    # 4. Accommodation calculation: 2 nights each for 4 nights total
    assert "1 room(s) × 2 night(s) in Tirumala Venkateswara Temple" in ans
    assert "1 room(s) × 2 night(s) in Dharmapuri Lakshmi Narasimha Swamy Temple" in ans
    assert "₹1,500" in ans  # 2 * 750
    assert "₹1,000" in ans  # 2 * 500

    # 5. Room availability confirmation pending
    assert "Room availability & tariff confirmation at Tirumala Venkateswara Temple (1 room(s) for 2 night(s))" in ans
    assert "Room availability & tariff confirmation at Dharmapuri Lakshmi Narasimha Swamy Temple (1 room(s) for 2 night(s))" in ans


def test_transit_arrival_window_and_night_coherence(itinerary_agent):
    """
    1. A 14–18-hour journey cannot automatically promise an evening arrival after a morning departure.
    Calculate the arrival window using the departure time and estimated travel duration,
    disclose late-night/past-midnight arrival, and synchronize accommodation nights.
    """
    state = AgentState(
        user_query="plan a 4.5 day trip to ttd and dharmapuri for 2 people with 8000 rupees",
        route_params={
            "temples": ["ttd", "dharmapuri"],
            "days": 4.5,
            "pax": 2,
            "budget_limit": 8000,
            "is_budget_request": True,
            "is_itinerary_request": True,
            "origin": "Hyderabad",
            "origin_assumed": True
        }
    )
    resp = itinerary_agent.execute(state)
    ans = resp.answer

    d3_pos = ans.find("#### Day 3:")
    d4_pos = ans.find("#### Day 4:")
    d3_text = ans[d3_pos:d4_pos]

    # Theme must not claim evening arrival
    assert "Evening Arrival" not in d3_text
    assert "Late-Night / Overnight Arrival" in d3_text or "Late-Night" in d3_text

    # Evening/Night window must state arrival window spans late night to past midnight
    assert "arrival window in Dharmapuri Lakshmi Narasimha Swamy Temple area spans late night (~10:30 PM) to past midnight (~02:30 AM into Day 4" in d3_text
    assert "Evening arrival cannot be assured" in d3_text
    assert "live GPS routing and delays unverified" in d3_text
    assert "covers late-night / past-midnight transit arrival" in d3_text


def test_unspecified_origin_return_journey_claims(itinerary_agent):
    """
    2. Missing origin prevents a definitive return-arrival claim and return-fare calculation.
    Day 5 must say that the return journey begins and that the final arrival time and fare are pending.
    Must not claim that traveler reaches starting city or that trip is completed on Day 5 without origin.
    """
    state = AgentState(
        user_query="plan a 4.5 day trip to ttd and dharmapuri for 2 people with 8000 rupees",
        route_params={
            "temples": ["ttd", "dharmapuri"],
            "days": 4.5,
            "pax": 2,
            "budget_limit": 8000,
            "is_budget_request": True,
            "is_itinerary_request": True,
            "origin": "Hyderabad",
            "origin_assumed": True
        }
    )
    resp = itinerary_agent.execute(state)
    ans = resp.answer

    d5_pos = ans.find("#### Day 5:")
    d5_text = ans[d5_pos:]

    # Must NOT claim trip is completed or destination reached without origin info
    assert "trip completed" not in d5_text.lower()
    assert "back to your starting destination (trip completed" not in d5_text

    # Must disclose return travel commences, arrival time and fares pending
    assert "Commence scheduled return journey by afternoon" in d5_text
    assert "departure origin is unspecified" in d5_text
    assert "final arrival time, connecting transit duration, and return fares remain pending" in d5_text


def test_day_4_infeasible_excursions_simplified(itinerary_agent):
    """
    3. An infeasible sightseeing schedule is rejected or simplified.
    Kawal Tiger Reserve and Kondagattu must NOT both be scheduled as mandatory visits
    on Day 4 alongside core temple rituals. Feasible local visit is preferred.
    """
    state = AgentState(
        user_query="plan a 4.5 day trip to ttd and dharmapuri for 2 people with 8000 rupees",
        route_params={
            "temples": ["ttd", "dharmapuri"],
            "days": 4.5,
            "pax": 2,
            "budget_limit": 8000,
            "is_budget_request": True,
            "is_itinerary_request": True,
            "origin": "Hyderabad",
            "origin_assumed": True
        }
    )
    resp = itinerary_agent.execute(state)
    ans = resp.answer

    d4_pos = ans.find("#### Day 4:")
    d5_pos = ans.find("#### Day 5:")
    d4_text = ans[d4_pos:d5_pos]

    # Must NOT schedule Kawal Tiger Reserve as morning exploration
    assert "Explore verified nearby landmark: **Kawal Tiger Reserve**" not in d4_text

    # Must feature nearby riverside shrine Gudem Gutta (~11.7 km)
    assert "Sri Satyanarayana Swamy Temple Gudem Gutta" in d4_text

    # Must NOT schedule Kondagattu as mandatory afternoon visit
    assert "In the afternoon, visit **Sri Anjaneya Swamy Temple | Kondagattu**" not in d4_text


def test_accommodation_classification_and_free_room_integrity(budget_agent):
    """
    4. Conflicting accommodation categories and unsupported free-room claims are flagged.
    Never label a property both '100% Free' and 'Private Hotel'.
    Distinguish free dormitory/waiting hall from private overnight rooms.
    """
    from app.agents.budget_agent import classify_and_enrich_accommodation

    # 1. PAC 1, 2, 3 & 4 in TTD must NOT be Private Hotel
    pac_item = {
        "name": "Pilgrim Amenities Complexes (PAC 1, 2, 3 & 4)",
        "type": "Free Transit Dormitories",
        "location": "Tirumala Hilltop",
        "price": "100% Free",
        "description": "Massive open halls with free lockers, 24/7 hot water, and canteens."
    }
    enriched_pac = classify_and_enrich_accommodation(pac_item, temple_canonical="Tirumala Venkateswara Temple")
    assert enriched_pac["property_type"] != "Private Hotel"
    assert enriched_pac["management_type"] == "TEMPLE_MANAGED"
    assert "Open Dormitory Hall" in enriched_pac["property_type"] or "Free Transit" in enriched_pac["property_type"]
    assert "does not provide private double room" in enriched_pac["price"]

    # 2. Choultries in TTD must be temple managed, not private hotels
    choultry_item = {
        "name": "Sri Venkateswara Dharmasala & Choultries (1st, 2nd NC & 3rd NC)",
        "type": "City Budget Choultries",
        "location": "Near Railway Station, Tirupati",
        "price": "Free / ₹50 – ₹100/night",
        "description": "Historical choultries with free locker facilities."
    }
    enriched_choultry = classify_and_enrich_accommodation(choultry_item, temple_canonical="ttd")
    assert enriched_choultry["property_type"] != "Private Hotel"
    assert enriched_choultry["management_type"] == "TEMPLE_MANAGED"

    # 3. Universal invariant: No property can be Private Hotel and 100% Free
    generic_free = {
        "name": "Random Waiting Hall",
        "type": "Waiting Hall",
        "price": "100% Free",
        "description": "Open hall for devotees"
    }
    enriched_gen = classify_and_enrich_accommodation(generic_free, temple_canonical="random")
    assert enriched_gen["property_type"] != "Private Hotel"


def test_unverified_and_low_tariff_caveats(budget_agent):
    """
    5. Unverified tariffs are not silently converted into confirmed prices.
    Unusually low tariffs (₹50/night) must carry offline CRO queue token caveats.
    Unlisted prices must show 'Tariff/eligibility requires confirmation'.
    """
    from app.agents.budget_agent import classify_and_enrich_accommodation

    cottage_50 = {
        "name": "Hill View Cottages (Ordinary)",
        "type": "Budget Cottages",
        "location": "Tirumala Hilltop",
        "price": "₹50/night",
        "description": "Ordinary non-AC budget cottages in the Hill View zone."
    }
    enriched_50 = classify_and_enrich_accommodation(cottage_50, temple_canonical="ttd")
    assert "Offline devasthanam CRO counter quota" in enriched_50["price"]
    assert "availability unconfirmed" in enriched_50["price"]

    unlisted_item = {
        "name": "Local Lodge",
        "type": "Lodge",
        "price": "",
        "description": "Local lodge near temple"
    }
    enriched_unlisted = classify_and_enrich_accommodation(unlisted_item, temple_canonical="somewhere")
    assert enriched_unlisted["price"] == "Tariff/eligibility requires confirmation"


def test_accommodation_budget_itinerary_synchronization(itinerary_agent, budget_agent):
    """
    6. Selected accommodation, budget totals, itinerary nights, and trip duration remain synchronized.
    4.5 days requires 4 nights (2 nights TTD + 2 nights Dharmapuri).
    """
    state = AgentState(
        user_query="plan a 4.5 day trip to ttd and dharmapuri for 2 people with 8000 rupees",
        route_params={
            "temples": ["ttd", "dharmapuri"],
            "days": 4.5,
            "pax": 2,
            "budget_limit": 8000,
            "is_budget_request": True,
            "is_itinerary_request": True,
            "origin": "Hyderabad",
            "origin_assumed": True
        }
    )
    b_resp = budget_agent.execute(state)
    i_resp = itinerary_agent.execute(state)

    b_ans = b_resp.answer
    i_ans = i_resp.answer

    # Budget table: 2 nights at TTD, 2 nights at Dharmapuri
    assert "1 room(s) × 2 night(s) in Tirumala Venkateswara Temple" in b_ans
    assert "1 room(s) × 2 night(s) in Dharmapuri Lakshmi Narasimha Swamy Temple" in b_ans

    # Itinerary days: Day 1-2 overnight at TTD, Day 3-4 overnight at Dharmapuri
    assert i_ans.count("Selected stay: **Ampro Rest House") == 2
    assert i_ans.count("Selected stay: **Community Satralu") == 2

    # Metadata nights
    assert b_resp.metadata["parameters"]["nights"] == 4
    assert i_resp.metadata["nights"] == 4


def test_accommodation_catalogue_deduplication_and_limit(budget_agent):
    """
    7. The final response does not repeat the full accommodation catalogue unnecessarily.
    Displays at most 3-5 relevant, validated alternatives per destination.
    """
    state = AgentState(
        user_query="plan a 4.5 day trip to ttd and dharmapuri for 2 people with 8000 rupees",
        route_params={
            "temples": ["ttd", "dharmapuri"],
            "days": 4.5,
            "pax": 2,
            "budget_limit": 8000,
            "is_budget_request": True,
            "is_itinerary_request": True,
            "origin": "Hyderabad",
            "origin_assumed": True
        }
    )
    resp = budget_agent.execute(state)
    ans = resp.answer

    # Accommodation options section
    assert "### 🏨 Accommodation Options & Suggestions" in ans
    ttd_pos = ans.find("#### Tirumala Venkateswara Temple")
    dharm_pos = ans.find("#### Dharmapuri Lakshmi Narasimha Swamy Temple")
    seva_pos = ans.find("<details>")

    ttd_accom_text = ans[ttd_pos:dharm_pos]
    ttd_bullet_count = ttd_accom_text.count("- **")

    # Full catalogue has 48 items; filtered response must have between 3 and 5 items
    assert 3 <= ttd_bullet_count <= 5

    # Dharmapuri options
    dharm_accom_text = ans[dharm_pos:seva_pos] if seva_pos != -1 else ans[dharm_pos:]
    dharm_bullet_count = dharm_accom_text.count("- **")
    assert 1 <= dharm_bullet_count <= 5

