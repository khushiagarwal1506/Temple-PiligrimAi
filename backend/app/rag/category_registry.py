import re
from typing import Dict, Any, List, Optional, Set

class CategoryMetadata:
    """Represents canonical metadata and lookup patterns for a dataset category."""
    def __init__(
        self,
        key: str,
        display_name: str,
        candidate_files: List[str],
        patterns: List[str],
        list_oriented: bool = False,
        aggregation_supported: bool = True,
        negative_aggregation_supported: bool = True,
        target_keys: Optional[List[str]] = None,
        description: str = ""
    ):
        self.key = key
        self.display_name = display_name
        self.candidate_files = candidate_files
        self.patterns = patterns
        self.list_oriented = list_oriented
        self.aggregation_supported = aggregation_supported
        self.negative_aggregation_supported = negative_aggregation_supported
        self.target_keys = target_keys or []
        self.description = description

class CategoryRegistry:
    """
    Canonical, extensible registry covering all 21 JSON categories in the PilgrimAI knowledge base.
    Provides deterministic mapping from natural-language queries/synonyms to canonical category definitions.
    """

    def __init__(self):
        self._categories: Dict[str, CategoryMetadata] = {}
        self._init_registry()

    def _init_registry(self):
        # 1. Accommodation
        self.register(CategoryMetadata(
            key="accommodation",
            display_name="Accommodation",
            candidate_files=["accommodation.json", "facilities.json"],
            patterns=[
                r"\b(accommodation|accommodations|rooms|cottages|guest\s*houses?|stay|choultries|choultry|lodging|dormitory|dormitories|sadanam|sadanams)\b"
            ],
            list_oriented=True,
            aggregation_supported=True,
            negative_aggregation_supported=True,
            target_keys=["accommodations", "guest_houses", "rooms", "hotels", "cottages"],
            description="Lodging, guest houses, cottages, and room booking options"
        ))

        # 2. Restaurants & Food
        self.register(CategoryMetadata(
            key="restaurants",
            display_name="Restaurants & Dining",
            candidate_files=["restaurants.json", "facilities.json"],
            patterns=[
                r"\b(restaurants?|hotels?|eating\s*places?|food\s*stalls?|eateries|eatery|tiffins?|dining|food\s*spots?|canteen|canteens|food\s*options?)\b"
            ],
            list_oriented=True,
            aggregation_supported=True,
            negative_aggregation_supported=True,
            target_keys=["restaurants", "name", "cuisine", "nearby_restaurants"],
            description="Nearby eateries, restaurants, canteens, and dining options"
        ))

        # 3. Contacts & Administration
        self.register(CategoryMetadata(
            key="contacts",
            display_name="Contacts & Phone Numbers",
            candidate_files=["contacts.json", "emergency_nos.json"],
            patterns=[
                r"\b(contacts?|phone\s*numbers?|phone|telephone|email|office\s*contact|reach\s*out|contact\s*info|enquiry\s*number|devasthanam\s*office)\b"
            ],
            list_oriented=False,
            aggregation_supported=True,
            negative_aggregation_supported=True,
            target_keys=["phone", "email", "office_contacts", "temple_office"],
            description="Temple administrative phone numbers, emails, and office contacts"
        ))

        # 4. Crowd & Peak Times
        self.register(CategoryMetadata(
            key="crowd",
            display_name="Crowd & Peak Hours",
            candidate_files=["crowd.json", "timings.json"],
            patterns=[
                r"\b(crowd|crowds|rush|waiting\s*time|queue\s*time|busy\s*days?|peak\s*hours?|rush\s*hours?|crowd\s*status|darshan\s*waiting)\b"
            ],
            list_oriented=False,
            aggregation_supported=True,
            negative_aggregation_supported=False,
            target_keys=["peak_hours", "busy_days", "waiting_time", "crowd_management"],
            description="Crowd estimates, waiting times, and peak rush days"
        ))

        # 5. Donations & Trusts
        self.register(CategoryMetadata(
            key="donations",
            display_name="Donations & E-Hundi",
            candidate_files=["donations.json"],
            patterns=[
                r"\b(donations?|donating|e-?hundi|hundi|trusts?|schemes?|annadanam\s*trust|contributions?|online\s*donation)\b"
            ],
            list_oriented=True,
            aggregation_supported=True,
            negative_aggregation_supported=True,
            target_keys=["donations", "schemes", "trusts", "bank_details"],
            description="Donation schemes, trust funds, and e-hundi contribution details"
        ))

        # 6. Wheelchair & Elderly Accessibility
        self.register(CategoryMetadata(
            key="wheelchair",
            display_name="Wheelchair & Elderly Accessibility",
            candidate_files=["elderly_ppl.json", "facilities.json"],
            patterns=[
                r"\b(wheelchair|wheel\s*chair|wheelchairs|elderly|senior\s*citizens?|differently\s*abled|handicapped|disabled|accessibility|accessible|lift|lifts|elevators?|ramps?|escalators?)\b"
            ],
            list_oriented=False,
            aggregation_supported=True,
            negative_aggregation_supported=True,
            target_keys=["wheelchair_accessibility", "is_wheelchair_friendly", "has_lift", "details", "ramps"],
            description="Wheelchair availability, ramp access, elevators, and senior citizen assistance"
        ))

        # 7. Emergency Contacts
        self.register(CategoryMetadata(
            key="emergency_contacts",
            display_name="Emergency Contacts & Numbers",
            candidate_files=["emergency_nos.json", "contacts.json"],
            patterns=[
                r"\b(emergency|emergencies|emergency\s*numbers?|emergency\s*contacts?|ambulance|police|medical\s*emergency|hospital|first\s*aid|fire\s*station)\b"
            ],
            list_oriented=False,
            aggregation_supported=True,
            negative_aggregation_supported=True,
            target_keys=["emergency_numbers", "ambulance", "police", "medical"],
            description="Emergency phone numbers for ambulance, police, and medical clinics"
        ))

        # 8. General Facilities & Amenities
        self.register(CategoryMetadata(
            key="facilities",
            display_name="Facilities & Amenities",
            candidate_files=["facilities.json", "temple.json"],
            patterns=[
                r"\b(facilities|facility|amenities|amenity|cloak\s*rooms?|lockers?|parking|drinking\s*water|toilets?|washrooms?|restrooms?|shoe\s*stand|kalyana\s*katta|tonsure)\b"
            ],
            list_oriented=True,
            aggregation_supported=True,
            negative_aggregation_supported=True,
            target_keys=["facilities", "parking", "cloak_room", "drinking_water", "toilets"],
            description="General pilgrim amenities including parking, cloak rooms, drinking water, and washrooms"
        ))

        # 9. Frequently Asked Questions (FAQ)
        self.register(CategoryMetadata(
            key="faq",
            display_name="FAQ",
            candidate_files=["faq.json"],
            patterns=[
                r"\b(faq|faqs|frequently\s*asked|common\s*questions?|general\s*questions?)\b"
            ],
            list_oriented=True,
            aggregation_supported=False,
            negative_aggregation_supported=False,
            target_keys=["faq", "questions", "q", "a"],
            description="Frequently asked questions and answers regarding temple visits"
        ))

        # 10. Festivals & Celebrations
        self.register(CategoryMetadata(
            key="festivals",
            display_name="Festivals & Utsavams",
            candidate_files=["festivals.json"],
            patterns=[
                r"\b(festivals?|utsavams?|utsavalu|brahmotsavams?|annual\s*events?|jatara|kalyanotsavam|celebrations?|special\s*occasions?)\b"
            ],
            list_oriented=True,
            aggregation_supported=True,
            negative_aggregation_supported=True,
            target_keys=["festivals", "festival_name", "utsavams", "name"],
            description="Annual festivals, Brahmotsavams, Jataras, and special utsavams"
        ))

        # 11. Photo Gallery & Media
        self.register(CategoryMetadata(
            key="gallery",
            display_name="Photo Gallery",
            candidate_files=["gallery.json"],
            patterns=[
                r"\b(gallery|photo\s*gallery|photos\s*gallery|image\s*gallery|pictures\s*gallery|media\s*gallery)\b"
            ],
            list_oriented=False,
            aggregation_supported=False,
            negative_aggregation_supported=False,
            target_keys=["gallery", "photos", "images"],
            description="Temple photo gallery and image records"
        ))

        # 12. Temple Maps & Navigation
        self.register(CategoryMetadata(
            key="maps",
            display_name="Temple Maps & Layout",
            candidate_files=["maps.json", "travel.json"],
            patterns=[
                r"\b(maps?|temple\s*map|route\s*map|layout|temple\s*layout|complex\s*map|directions\s*map)\b"
            ],
            list_oriented=False,
            aggregation_supported=False,
            negative_aggregation_supported=False,
            target_keys=["maps", "layout", "map_url"],
            description="Temple complex layout maps and navigation guides"
        ))

        # 13. Nearby Attractions & Sightseeing
        self.register(CategoryMetadata(
            key="nearby_attractions",
            display_name="Nearby Attractions",
            candidate_files=["nearby_attractions.json"],
            patterns=[
                r"\b(nearby\s*attractions?|places\s*to\s*visit|sightseeing|tourist\s*spots?|tourist\s*places?|nearby\s*places?|attractions\s*around)\b"
            ],
            list_oriented=True,
            aggregation_supported=True,
            negative_aggregation_supported=True,
            target_keys=["attractions", "places", "nearby_attractions", "name"],
            description="Tourist spots, waterfalls, historical monuments, and sightseeing places near the temple"
        ))

        # 14. Prasadam & Annadanam
        self.register(CategoryMetadata(
            key="prasadam",
            display_name="Prasadam & Annadanam",
            candidate_files=["prasadam.json", "prasadams.json", "facilities.json"],
            patterns=[
                r"\b(prasadams?|laddu|pulihora|holy\s*offering|food\s*offerings?|annadanam|anna\s*danam|free\s*meals?|nithya\s*annadanam|free\s*food)\b"
            ],
            list_oriented=True,
            aggregation_supported=True,
            negative_aggregation_supported=True,
            target_keys=["prasadam", "items", "food_offerings", "annadanam"],
            description="Prasadam varieties, price lists, counters, and free Annadanam meal timings"
        ))

        # 15. River Bank & Ghats
        self.register(CategoryMetadata(
            key="river_bank",
            display_name="River Bank & Ghats",
            candidate_files=["river_bank.json", "travel.json", "temple.json"],
            patterns=[
                r"\b(river|rivers|river\s*bank|river\s*banks|riverbank|riverbanks|water\s*body|ghat|ghats|holy\s*dip|bathing\s*ghats?|snanam|pushkarni|pushkarini|sangam|sangamam|confluence)\b"
            ],
            list_oriented=False,
            aggregation_supported=True,
            negative_aggregation_supported=True,
            target_keys=["river_name", "ghats", "significance", "safety_rules"],
            description="Associated sacred rivers, bathing ghats, Pushkarnis, and holy dip facilities"
        ))

        # 16. Rules & Guidelines
        self.register(CategoryMetadata(
            key="rules",
            display_name="Rules & Guidelines",
            candidate_files=["rules.json", "temple.json"],
            patterns=[
                r"\b(rules?|dress\s*code|guidelines?|dos\s*and\s*don'?ts|photography|photos?\s*(?:allowed|prohibited|ban|rules?|policy)|cameras?|mobiles?|mobile\s*phones?|phones?|electronics?|electronic\s*(?:ban|devices?|gadgets?)|smartwatches?|devices?|gadgets?|restrictions?|code\s*of\s*conduct|traditions?|prohibited|allowed|banned|bans?|pets?|animals?|smoking|alcohol|footwear|shoes|leather)\b"
            ],
            list_oriented=False,
            aggregation_supported=True,
            negative_aggregation_supported=False,
            target_keys=["rules", "dress_code", "dos", "donts", "guidelines", "restrictions", "electronic_ban", "safety_guidelines"],
            description="Dress codes, photography rules, mobile phone restrictions, electronic bans, pet policies, and pilgrim guidelines"
        ))

        # 17. Sevas & Poojas
        self.register(CategoryMetadata(
            key="sevas",
            display_name="Sevas & Poojas",
            candidate_files=["sevas.json"],
            patterns=[
                r"\b(sevas?|poojas?|pujas?|rituals?|abhishekam|archana|kalyanam|homam|aarthi|harati|darshan|special\s*darshan|sevas\s*list)\b"
            ],
            list_oriented=True,
            aggregation_supported=True,
            negative_aggregation_supported=True,
            target_keys=["sevas", "name", "ritual", "price", "timing"],
            description="Daily, weekly, and special sevas, poojas, rituals, pricing, and timings"
        ))

        # 18. Sub-Temples & Shrines
        self.register(CategoryMetadata(
            key="sub_temples",
            display_name="Sub-Temples & Subsidiary Shrines",
            candidate_files=["sub_temples.json"],
            patterns=[
                r"\b(sub[- ]temples?|shrines?|subsidiary\s*shrines?|upalayam|upalayams?|smaller\s*temples?|other\s*shrines?)\b"
            ],
            list_oriented=True,
            aggregation_supported=True,
            negative_aggregation_supported=True,
            target_keys=["sub_temples", "shrines", "name", "deity"],
            description="Subsidiary shrines and sub-temples located within the main temple complex"
        ))

        # 19. Temple History & Overview
        self.register(CategoryMetadata(
            key="temple",
            display_name="Temple History & Overview",
            candidate_files=["temple.json"],
            patterns=[
                r"\b(history|origin|origins|legend|legends|significance|presiding\s*deity|main\s*deity|architecture|about\s*temple|sthala\s*puranam|mythology|story|background)\b"
            ],
            list_oriented=False,
            aggregation_supported=False,
            negative_aggregation_supported=False,
            target_keys=["name", "deity", "history", "architecture", "significance"],
            description="Historical background, legends, architectural style, and presiding deities"
        ))

        # 20. Timings & Schedules
        self.register(CategoryMetadata(
            key="timings",
            display_name="Timings & Schedules",
            candidate_files=["timings.json", "temple.json"],
            patterns=[
                r"\b(timings?|opening\s*hours?|closing\s*hours?|darshan\s*timings?|schedules?|temple\s*hours?|open\s*time|close\s*time|operating\s*hours?|morning\s*timings?|evening\s*timings?)\b"
            ],
            list_oriented=False,
            aggregation_supported=True,
            negative_aggregation_supported=False,
            target_keys=["timings", "darshan_timings", "morning", "evening", "general_timings"],
            description="Temple opening, closing, darshan, and break timings"
        ))

        # 21. Travel & How to Reach
        self.register(CategoryMetadata(
            key="travel",
            display_name="Travel & How to Reach",
            candidate_files=["travel.json"],
            patterns=[
                r"\b(travel|how\s*to\s*reach|buses?|bus\s*route|trains?|railway\s*stations?|airports?|distances?|transport|transit|routes?|nearest\s*airport|nearest\s*railway)\b"
            ],
            list_oriented=False,
            aggregation_supported=True,
            negative_aggregation_supported=False,
            target_keys=["how_to_reach", "by_bus", "by_train", "by_air", "distance_from"],
            description="Transit details, bus routes, nearest railway stations, airports, and road distances"
        ))

    def register(self, meta: CategoryMetadata):
        """Registers a category definition in the registry."""
        self._categories[meta.key] = meta

    def get_category(self, key: str) -> Optional[CategoryMetadata]:
        """Retrieves category metadata by canonical key."""
        return self._categories.get(key)

    def all_categories(self) -> List[CategoryMetadata]:
        """Returns list of all registered category definitions."""
        return list(self._categories.values())

    def match_categories(self, query: str) -> List[str]:
        """
        Deterministically matches all relevant canonical category keys from query text.
        Handles compound modifiers (e.g. 'wheelchair facilities', 'accommodation facilities')
        so generic 'facilities' does not trigger false multi-category intent.
        """
        q = query.lower()
        matched = []
        for key, meta in self._categories.items():
            for pattern in meta.patterns:
                if re.search(pattern, q):
                    matched.append(key)
                    break

        # Compound modifier resolution: if 'facilities' is matched alongside a specific category
        # where the query contains '<category> facilities' or '<category> facility', keep the specific category
        if "facilities" in matched and len(matched) > 1:
            compound_patterns = [
                r"\b(wheelchair|parking|accommodation|stay|toilet|washroom|river|water|dining|food)\s+(facilities|facility|amenities|amenity)\b"
            ]
            if any(re.search(pat, q) for pat in compound_patterns):
                matched = [k for k in matched if k != "facilities"]

        return matched

    def is_list_oriented(self, category_key: str) -> bool:
        """Returns True if the category is typically list-oriented."""
        meta = self.get_category(category_key)
        return meta.list_oriented if meta else False

category_registry = CategoryRegistry()
