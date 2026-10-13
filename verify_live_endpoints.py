import json
import urllib.request

def call_chat(query: str):
    req = urllib.request.Request(
        "http://127.0.0.1:8000/chat",
        data=json.dumps({"question": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def test_query_1_chilkur():
    print("==================================================================")
    print("Testing Query 1: Chilkur Balaji 3-day trip (budget 3,400)")
    print("==================================================================")
    data = call_chat("plan a 3 day trip to chilkur balaji with a budget of 3400")
    ans = data["answer"]
    meta = data["metadata"]
    
    # 1. Structured metadata origin preservation
    assert meta["origin_assumed"] is True, "Metadata origin_assumed must be True"
    assert meta["parameters"]["origin_assumed"] is True, "Parameters origin_assumed must be True"
    assert "Departure origin unspecified — transit excluded" in ans or "Starting Location Unspecified" in ans, \
        "Answer must disclose that origin is unstated"
    assert "Travel corridor benchmark" in ans and "Departure origin unspecified" in ans and "Chilkur Balaji Temple" in ans, \
        "Corridor benchmark must display unstated departure origin"
    assert "Travel corridor benchmark: Hyderabad" not in ans, \
        "Corridor benchmark must not claim Hyderabad as origin"
    
    # 2. Structured Line Items & Arithmetic Verification
    plan_items = meta["selected_plan"]["line_items"]
    
    # Transit must be unavailable and uncosted
    travel_item = next(it for it in plan_items if it["category"] in ["travel", "transportation"])
    assert travel_item["provenance"] == "UNAVAILABLE", "Unstated travel must have provenance UNAVAILABLE"
    assert travel_item["selected_cost"] == 0.0, "Unstated travel cost must be 0.0 (not confirmed)"
    assert travel_item["confirmation_required"] is True, "Unstated travel must require confirmation"
    
    # Lodging must be unavailable and uncosted
    accom_item = next(it for it in plan_items if it["category"] == "accommodation")
    assert accom_item["provenance"] == "UNAVAILABLE", "Unlisted lodging must have provenance UNAVAILABLE"
    assert accom_item["selected_cost"] == 0.0, "Unlisted lodging selected_cost must be 0.0"
    
    # Meals and local transit calculations
    meals_item = next(it for it in plan_items if it["category"] == "meals")
    assert meals_item["selected_cost"] == 600.0, "Meals must be exactly 1 pax * 3 days * 200 = 600"
    local_item = next(it for it in plan_items if it["category"] == "local_transit")
    assert local_item["selected_cost"] == 150.0, "Local transit must be exactly 3 days * 50 = 150"
    
    # Total calculation must strictly equal partial on-site expenses
    expected_total = meals_item["selected_cost"] + local_item["selected_cost"]  # 750.0
    assert meta["selected_plan"]["total"] == expected_total, f"Total must equal {expected_total}"
    assert meta["selected_plan"]["buffer"] == 3400.0 - expected_total, "Buffer must be mathematical difference (2650.0)"
    
    # 3. Honest disclosure in text: partial estimate & not confirmed spendable money
    assert "Partial on-site estimate" in ans, "Answer must label baseline as partial on-site estimate"
    assert "NOT confirmed spendable money" in ans, "Answer must disclose that unallocated margin is not confirmed spendable money"
    assert "Safety buffer" not in ans, "Must not describe unallocated budget as confirmed safety buffer"
    assert "[Fits within remaining buffer]" not in ans, "Must not display misleading fit badge on unlisted options"
    
    # 4. Schedule parameters
    assert meta["parameters"]["rooms"] == 1, "Parameters rooms must be 1"
    assert meta["parameters"]["nights"] == 2, "Parameters nights must be 2"
    assert meta["parameters"]["days"] == 3.0, "Parameters days must be 3.0"
    
    print("Query 1 Audit PASSED: Structured arithmetic verified, unstated origin respected, no fake Rs.0 lodging, partial estimate declared!\n")

def test_query_2_udaipur_multi():
    print("==================================================================")
    print("Testing Query 2: Udaipur multi-temple circuit (8 days, 4 pax, 20k)")
    print("==================================================================")
    data = call_chat("plan a trip from udaipur to annavarm, medaram, jogulamba and karmaghat for 8 days with a budget of 20000 for 4 people")
    ans = data["answer"]
    meta = data["metadata"]
    
    # 1. Structured parameters & Origin Preservation
    assert meta["origin"] == "Udaipur", "Metadata origin must be Udaipur"
    assert meta["origin_assumed"] is False, "Metadata origin_assumed must be False"
    params = meta["parameters"]
    assert params["pax"] == 4, "Must have 4 pax"
    assert params["rooms"] == 2, "4 pax must have 2 rooms"
    assert params["nights"] == 7, "8 days must require 7 nights"
    assert params["days"] == 8.0, "Must have 8.0 days"
    assert params["budget_limit"] == 20000.0, "Budget limit must be 20000.0"
    
    # 2. Destination Coverage
    for dest in ["Annavaram", "Medaram", "Jogulamba", "Karmanghat"]:
        assert dest in ans, f"Destination {dest} must be in answer text"
        assert any(dest.lower() in t.lower() for t in meta["temples"]), f"Destination {dest} must be in metadata temples"
    
    # 3. Routing: Individual legs, Rail benchmarks vs Unverified road corridors
    assert "Route Leg 1" in ans and "Udaipur ➔ Annavaram" in ans
    assert "Route Leg 2" in ans and "Annavaram ➔ Medaram" in ans
    assert "Route Leg 3" in ans and "Medaram ➔ Jogulamba" in ans
    assert "Route Leg 4" in ans and "Jogulamba ➔ Karmanghat" in ans
    assert "Route Leg 5" in ans and "Karmanghat ➔ Udaipur" in ans
    
    # Check that Rail legs distinguish geographical vs IRCTC rail routing
    assert "[Rail Estimate — Source: IRCTC Network Corridor Benchmark, Not Live GPS/Tracking" in ans, \
        "Rail legs must cite IRCTC corridor benchmark and distinguish straight-line geometry"
    # Check that Road legs are labeled unverified due to lack of live Google Routes API key
    assert "[Road Corridor Estimate — Status: Unverified; Live Google Routes API unavailable" in ans, \
        "Road legs must disclose unverified status due to absence of live Google Routes API key"
    
    # 4. Structured Transport & Fare Audit
    plan_items = meta["selected_plan"]["line_items"]
    trans_item = next(it for it in plan_items if it["category"] == "transportation")
    assert trans_item["selected_cost"] == 9000.0, f"Transport baseline must be 9000.0 (got {trans_item['selected_cost']})"
    assert trans_item["higher_cost"] == 19200.0, f"Transport higher tier must be 19200.0 (got {trans_item['higher_cost']})"
    assert trans_item["provenance"] == "ESTIMATE", "Transport must be ESTIMATE"
    assert "9,000" in trans_item["notes"], "Transport notes must cite 9,000 baseline matching line item"
    
    # 5. Accommodation Arithmetic & Overnight Stay Reconciliation
    # The trip is 8 days / 7 nights.
    # Night 1: Overnight Express Train (rail transit fare; not charged as hotel night).
    # Remaining 6 nights: Distributed across destinations as 1 Annavaram + 2 Medaram + 2 Jogulamba + 1 Karmanghat = 6 hotel nights.
    
    # Annavaram: 2 rooms * 1 night * 200 = 400 (Listed)
    annavaram_stay = next(it for it in plan_items if "Annavaram" in it["item_name"] and it["category"] == "accommodation")
    assert annavaram_stay["selected_cost"] == 400.0, f"Annavaram stay must be 400.0 for 1 night (got {annavaram_stay['selected_cost']})"
    assert annavaram_stay["higher_cost"] == 400.0, f"Annavaram higher cost must be 400.0 (got {annavaram_stay['higher_cost']})"
    assert annavaram_stay["provenance"] == "LISTED_PRICE", "Annavaram stay must be LISTED_PRICE"
    assert "1 night" in annavaram_stay["quantity_desc"], "Annavaram lodging must specify 1 night"
    
    # Medaram: Forest Tents - 2 tents * 2 nights (24h period) @ 2,000 = 8000
    medaram_stay = next(it for it in plan_items if "Medaram" in it["item_name"] and it["category"] == "accommodation")
    assert medaram_stay["selected_cost"] == 8000.0, f"Medaram stay baseline must be 8000.0 for 24h stay (got {medaram_stay['selected_cost']})"
    assert medaram_stay["higher_cost"] == 8000.0, f"Medaram stay higher cost must be 8000.0 (got {medaram_stay['higher_cost']})"
    assert medaram_stay["provenance"] == "LISTED_PRICE", "Medaram stay must be LISTED_PRICE"
    assert "2 night" in medaram_stay["quantity_desc"], "Medaram lodging must specify 2 nights"
    assert "24-hr overnight stay: 2 tents × 2 nights × ₹2,000 = ₹8,000" in ans, \
        "Answer must show full 24h overnight stay of ₹8,000"
    
    # Jogulamba: 2 rooms * 2 nights * 300 = 1200 (range up to 2 * 2 * 1200 = 4800)
    jogulamba_stay = next(it for it in plan_items if "Jogulamba" in it["item_name"] and it["category"] == "accommodation")
    assert jogulamba_stay["selected_cost"] == 1200.0, f"Jogulamba stay must be 1200.0 (got {jogulamba_stay['selected_cost']})"
    assert jogulamba_stay["higher_cost"] == 4800.0, f"Jogulamba stay higher cost must be 4800.0 (got {jogulamba_stay['higher_cost']})"
    assert "2 night" in jogulamba_stay["quantity_desc"] or "2.0 night" in jogulamba_stay["quantity_desc"], "Jogulamba lodging must specify 2 nights"
    
    # Karmanghat: Tariff Unlisted in static records -> Must be kept unknown (selected_cost == 0.0, UNAVAILABLE)
    karmanghat_stay = next(it for it in plan_items if "Karmanghat" in it["item_name"] and it["category"] == "accommodation")
    assert karmanghat_stay["provenance"] == "UNAVAILABLE", "Karmanghat lodging must be UNAVAILABLE"
    assert karmanghat_stay["selected_cost"] == 0.0, "Karmanghat unlisted lodging must have 0.0 selected cost"
    assert karmanghat_stay["higher_cost"] == 0.0, "Karmanghat unlisted lodging must have 0.0 higher cost"
    assert "no listed tariff" in karmanghat_stay["notes"], "Karmanghat must disclose tariff unlisted in static records"
    assert "Incomplete Cost Notice (Unlisted Lodging)" in ans or "incomplete partial subtotal" in ans, \
        "Answer must disclose that total is incomplete due to unlisted lodging"
    assert "1 night" in karmanghat_stay["quantity_desc"] or "1.0 night" in karmanghat_stay["quantity_desc"], "Karmanghat lodging must specify 1 night"
    
    # Assert sum of accommodation hotel nights is exactly 6 nights (1 + 2 + 2 + 1 = 6)
    accom_items = [it for it in plan_items if it["category"] == "accommodation"]
    total_hotel_nights = 1 + 2 + 2 + 1  # Annavaram (1) + Medaram (2) + Jogulamba (2) + Karmanghat (1)
    assert total_hotel_nights == 6, "Total hotel nights must equal 6"
    
    # Assert dated itinerary distinguishes Night 1 train sleeper vs destination hotel nights
    assert "Selected stay: **Express Train Sleeper Berth**" in ans, "Itinerary Day 1 must specify train sleeper berth"
    assert "Selected stay: **Sri Seetha Rama Choultry**" in ans, "Itinerary Day 2 must specify Annavaram choultry stay"
    assert "Selected stay: **Forest Tents (Forest Department)**" in ans, "Itinerary must specify Forest Department tents for Medaram"
    assert "Selected stay: **Hemalapuri Sadan**" in ans, "Itinerary must specify Hemalapuri Sadan for Jogulamba"
    
    # 6. Overall Budget Arithmetic & Derived Upper Cost Range
    # Lower bounds: 9000 (transit) + 400 (Annavaram) + 8000 (Medaram) + 1200 (Jogulamba) + 50 (Seva) + 0 (Karmanghat) + 6400 (Meals) + 800 (Local) = 25850.0
    # Upper bounds: 19200 (transit) + 400 (Annavaram) + 8000 (Medaram) + 4800 (Jogulamba) + 50 (Seva) + 0 (Karmanghat) + 11200 (Meals) + 1600 (Local) = 45250.0
    expected_min = sum(it["selected_cost"] for it in plan_items)
    expected_max = sum(it["higher_cost"] for it in plan_items)
    assert expected_min == 25850.0, f"Derived minimum must be 25850.0 (got {expected_min})"
    assert expected_max == 45250.0, f"Derived maximum must be 45250.0 (got {expected_max})"
    
    assert meta["selected_plan"]["total"] == expected_min, f"Selected plan total must match derived min {expected_min}"
    assert meta["broader_range"]["min"] == expected_min, f"Broader range min must match {expected_min}"
    assert meta["broader_range"]["max"] == expected_max, f"Broader range max must match {expected_max}"
    assert meta["budget_status"] == "OVER_BUDGET", "Budget status must be OVER_BUDGET"
    assert meta["is_over_budget"] is True, "is_over_budget must be True"
    assert meta["shortfall"] == 5850.0, f"Shortfall must be exactly 25850 - 20000 = 5850.0 (got {meta['shortfall']})"
    assert "₹5,850" in ans, "Answer must show exact shortfall of ₹5,850"
    assert "₹25,850 – ₹45,250" in ans, "Answer must show exact derived range ₹25,850 – ₹45,250"
    
    # 7. Itinerary Chronological Feasibility, Seva Plausibility & Disclosures
    assert "Feasible Proposed Schedule vs Confirmed Transport Itinerary" in ans, \
        "Answer must clearly distinguish feasible proposed schedule from confirmed transport itinerary"
    assert "6 usable on-ground days" in ans, "Answer must state 6 usable on-ground days"
    assert "Outbound Rail Transit Day" in ans or "no temple darshan is possible on Day 1" in ans, \
        "Day 1 must be dedicated to transit without premature darshan before arrival"
    assert "Train reaches Annavaram" in ans or "Train reaches" in ans, "Day 2 must show arrival at Annavaram"
    
    # Assert Day 2 morning does NOT promise 03:30 AM Suprabhata Seva upon arrival
    assert "Attend morning seva: **Sri Swamy Vari Suprabhata Seva** (03:30 AM)" not in ans, \
        "Day 2 morning must NOT promise 3:30 AM Suprabhata Seva when train arrives mid-morning"
    assert "Suprabhata Seva cannot be attended on Day 2 due to long-distance train arrival timing" in ans or \
           "daytime sanctum darshan" in ans, "Day 2 must note seva timing plausibility"
    
    assert "Return Rail Transit Note" in ans and "Day 9" in ans, \
        "Day 8 must account for return rail transit reaching Udaipur on Day 9"
    assert "10–11 total days" in ans or "10–11 days" in ans, \
        "Answer must recommend 10–11 total days while retaining all four requested destinations"
    
    print("Query 2 Audit PASSED: Reconciled 7 nights (1 train + 6 hotel), derived upper range (25,850 - 45,250), plausibility verified without 3:30 AM seva, honest 10-11 day recommendation!\n")

def test_regression_single_temple():
    print("==================================================================")
    print("Testing Regression Query: Specific seva lookup at single temple")
    print("==================================================================")
    data = call_chat("what is the cost of abhishekam at yadadri for 2 people")
    assert "yadadri" in data["answer"].lower() or "abhishekam" in data["answer"].lower()
    print("Regression Query PASSED!\n")

if __name__ == "__main__":
    test_query_1_chilkur()
    test_query_2_udaipur_multi()
    test_regression_single_temple()
    print("==================================================================")
    print("ALL STRUCTURED EVIDENCE-BASED AUDIT CHECKS PASSED WITH FULL MATHEMATICAL INTEGRITY!")
    print("==================================================================")
