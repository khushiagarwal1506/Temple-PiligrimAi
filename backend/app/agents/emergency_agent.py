"""
PilgrimAI - EmergencyAgent

Specialist agent providing verified, grounded emergency contacts, medical facilities,
police stations, ambulance services, and devasthanam help desks for pilgrimage temples.

Architectural Constraints:
- BaseAgent lifecycle contract: Inherits from BaseAgent, executes via execute(state).
- RAGTool Read-Only Facade: Communicates exclusively via RAGTool. Zero direct Qdrant,
  BM25, or raw JSON file imports.
- Zero Fabrication: Returns only verified records. Never manufactures phone numbers,
  hospital names, police stations, addresses, or opening hours.
- Strict Temple Scoping: Strictly isolates contacts to the user-specified temple.
  Never leaks contacts from another temple.
- Universal Fallback: If no temple is specified, provides verified statutory national
  and state helplines (112, 108, 100, 101) with instructions to specify a temple.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Set

from app.agents.base_agent import BaseAgent
from app.agents.state import AgentState, AgentResponse

logger = logging.getLogger(__name__)

# Universal statutory helplines recorded across the knowledge base
UNIVERSAL_HELPLINES = [
    {"name": "National Emergency Helpline", "number": "112", "desc": "All-in-one emergency response (Police, Fire, Ambulance)"},
    {"name": "State Ambulance & Medical Emergency", "number": "108", "desc": "Toll-free 24/7 emergency medical & ambulance service"},
    {"name": "Police Control Room", "number": "100", "desc": "Immediate police assistance"},
    {"name": "Fire & Rescue Services", "number": "101", "desc": "Emergency fire and disaster rescue"},
    {"name": "Women Safety & Helpline", "number": "181 / SHE Teams", "desc": "Women safety and grievance support"}
]

TEMPLE_CANONICAL_ALIASES: Dict[str, str] = {
    "ahobilam": "Ahobilam",
    "ahobila": "Ahobilam",
    "badrachalam": "Badrachalam",
    "bhadrachalam": "Badrachalam",
    "bhadragiri": "Badrachalam",
    "balkampet": "Balkampet",
    "yellamma": "Balkampet",
    "basara": "Basara",
    "basar": "Basara",
    "birla mandir": "Birla Mandir",
    "birla": "Birla Mandir",
    "chilkur": "Chilkur",
    "chilkur balaji": "Chilkur",
    "visa balaji": "Chilkur",
    "dharmapuri": "Dharmapuri",
    "edupayal": "Edupayal",
    "edupayala": "Edupayal",
    "jogulamba": "Jogulamba(new)",
    "jogulamba(new)": "Jogulamba(new)",
    "alampur": "Jogulamba(new)",
    "kaleshvaram": "Kaleshvaram",
    "kaleshwaram": "Kaleshvaram",
    "karmanghat": "Karmanghat",
    "karmaghat": "Karmanghat",
    "medaram": "Medaram",
    "sammakka": "Medaram",
    "saralamma": "Medaram",
    "ramappa": "Ramappa",
    "srisaila": "Srisaila",
    "srisailam": "Srisaila",
    "mallikarjuna": "Srisaila",
    "thousand pillar": "Thousand Pillar",
    "thousand pillars": "Thousand Pillar",
    "warangal thousand": "Thousand Pillar",
    "ttd": "Ttd",
    "tirumala": "Ttd",
    "tirupati": "Ttd",
    "ujjaini": "Ujjaini",
    "ujjaini mahakali": "Ujjaini",
    "veera satyanarayana": "Veera Satyanarayana",
    "annavaram": "Veera Satyanarayana",
    "annavarm": "Veera Satyanarayana",
    "satyanarayana": "Veera Satyanarayana",
    "vemulavada": "Vemulavada",
    "vemulawada": "Vemulavada",
    "rajarajeshwara": "Vemulavada",
    "rajanna": "Vemulavada",
    "yadadri": "Yadadri",
    "yadagirigutta": "Yadadri"
}

TEMPLE_DISPLAY_NAMES: Dict[str, str] = {
    "Ahobilam": "Ahobilam (Narasimha Swamy)",
    "Badrachalam": "Bhadrachalam (Sita Ramachandra Swamy)",
    "Balkampet": "Balkampet Yellamma",
    "Basara": "Basara Gnana Saraswathi",
    "Birla Mandir": "Birla Mandir (Hyderabad)",
    "Chilkur": "Chilkur Balaji",
    "Dharmapuri": "Dharmapuri Lakshmi Narasimha",
    "Edupayal": "Edupayala Vana Durga Bhavani",
    "Jogulamba(new)": "Alampur Jogulamba",
    "Kaleshvaram": "Kaleshwaram Mukteshwara Swamy",
    "Karmanghat": "Karmanghat Hanuman",
    "Medaram": "Medaram Sammakka Saralamma",
    "Ramappa": "Ramappa (Rudreshwara Temple)",
    "Srisaila": "Srisailam Mallikarjuna Swamy",
    "Thousand Pillar": "Thousand Pillar Temple (Hanamkonda)",
    "Ttd": "Tirumala Tirupati Devasthanam (TTD)",
    "Ujjaini": "Ujjaini Mahakali (Secunderabad)",
    "Veera Satyanarayana": "Annavaram Veera Venkata Satyanarayana",
    "Vemulavada": "Vemulawada Raja Rajeshwara Swamy",
    "Yadadri": "Yadadri Sri Lakshmi Narasimha Swamy"
}


class EmergencyAgent(BaseAgent):
    """
    Specialist agent providing verified, grounded emergency contacts, medical facilities,
    police stations, ambulance services, and devasthanam help desks for pilgrimage temples.
    """

    def __init__(
        self,
        rag_tool: Optional[Any] = None,
        name: str = "emergency",
        description: str = "Specialist agent providing verified, grounded emergency contacts, medical facilities, police stations, and help desks for pilgrimage temples."
    ):
        super().__init__(name=name, description=description)
        self._rag_tool = rag_tool

    @property
    def rag_tool(self) -> Any:
        """Lazily initialize RAGTool to break circular instantiation loops."""
        if self._rag_tool is None:
            from app.agents.tools.rag_tool import RAGTool
            self._rag_tool = RAGTool()
        return self._rag_tool

    @rag_tool.setter
    def rag_tool(self, value: Any) -> None:
        self._rag_tool = value

    def detect_temple(self, query: str) -> Optional[str]:
        """Detects registered canonical temple from query text."""
        if not query or not query.strip():
            return None
        q_norm = query.lower()
        for alias, canonical in sorted(TEMPLE_CANONICAL_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, q_norm):
                return canonical
        return None

    def _normalize_category_items(self, val: Any) -> List[Dict[str, str]]:
        """Normalizes diverse JSON formats (strings, dicts, lists) into clean key-value records."""
        items: List[Dict[str, str]] = []
        if isinstance(val, str) and val.strip():
            items.append({"raw": val.strip()})
        elif isinstance(val, dict):
            name = val.get("name") or val.get("service") or ""
            phone = val.get("phone") or val.get("contact") or val.get("contact_number") or val.get("number") or ""
            alt = val.get("alt_phone") or val.get("alternate_phone") or ""
            addr = val.get("address") or val.get("location") or ""
            details = val.get("details") or val.get("description") or ""
            items.append({
                "name": str(name).strip(),
                "phone": str(phone).strip(),
                "alt_phone": str(alt).strip(),
                "address": str(addr).strip(),
                "details": str(details).strip()
            })
        elif isinstance(val, list):
            for itm in val:
                if isinstance(itm, str) and itm.strip():
                    items.append({"raw": itm.strip()})
                elif isinstance(itm, dict):
                    name = itm.get("name") or itm.get("service") or ""
                    phone = itm.get("phone") or itm.get("contact") or itm.get("contact_number") or itm.get("number") or ""
                    alt = itm.get("alt_phone") or itm.get("alternate_phone") or ""
                    addr = itm.get("address") or itm.get("location") or ""
                    details = itm.get("details") or itm.get("description") or ""
                    items.append({
                        "name": str(name).strip(),
                        "phone": str(phone).strip(),
                        "alt_phone": str(alt).strip(),
                        "address": str(addr).strip(),
                        "details": str(details).strip()
                    })
        return items

    def _format_contact_line(self, item: Dict[str, str]) -> str:
        """Formats an individual contact entry safely without generating fake data."""
        if "raw" in item:
            return f"• {item['raw']}"

        parts = []
        name = item.get("name")
        if name:
            parts.append(f"**{name}**")

        phone_parts = []
        if item.get("phone"):
            phone_parts.append(f"Phone: {item['phone']}")
        if item.get("alt_phone"):
            phone_parts.append(f"Alt: {item['alt_phone']}")
        if phone_parts:
            parts.append(" | ".join(phone_parts))

        if item.get("address"):
            parts.append(f"Location: {item['address']}")

        if item.get("details"):
            parts.append(f"({item['details']})")

        return " • " + " — ".join(parts) if parts else ""

    def _detect_and_build_immediate_action(self, query: str) -> Optional[str]:
        """
        Detects urgent situational crises (unresponsive person, missing child,
        severe dehydration, dangerous flood/weather, lost valuables) and produces
        immediate, actionable, lifesaving safety guidance first.
        """
        q_lower = query.lower()

        # 1. Unresponsive / Unconscious / Medical Collapse
        if re.search(r"\b(unresponsive|unconscious|fainted|passed\s+out|collapsed|not\s+breathing|cardiac|heart\s+attack|cpr)\b", q_lower):
            return (
                "#### ⚠️ IMMEDIATE LIFESAVING PROTOCOL — MEDICAL EMERGENCY:\n"
                "1. **Call Medical Help Immediately**: Dial **108** (Ambulance) or **112** (All-in-one Emergency).\n"
                "2. **Check Responsiveness & Airway**: Tap shoulders firmly and shout. Check if chest is rising. If the person is not breathing normally, begin **Chest Compressions (CPR)** immediately in the center of the chest (100–120 beats/min).\n"
                "3. **Alert Temple Security / Volunteers**: Shout for assistance and request the nearest Automated External Defibrillator (AED) or temple first-aid squad. Clear surrounding crowds to ensure open airflow.\n"
                "4. **Do NOT administer water, drinks, or food** to an unconscious or unresponsive individual."
            )

        # 2. Missing Child in a Crowd
        if re.search(r"\b(missing\s+child|lost\s+child|child\s+lost|kid\s+lost|lost\s+my\s+(?:son|daughter|kid|child|baby))\b", q_lower):
            return (
                "#### ⚠️ IMMEDIATE MISSING CHILD PROTOCOL:\n"
                "1. **Alert Temple Control Room**: Rush immediately to the nearest Devasthanam Help Desk, Control Room, or Police Post to initiate urgent public address announcements across all loudspeakers.\n"
                "2. **Contact Police & Childline**: Call **100** / **112** (Police) and **1098** (National Childline).\n"
                "3. **Seal / Monitor Exit Gates**: Request security volunteers at all perimeter gates to monitor exiting crowds for unaccompanied children.\n"
                "4. **Provide Distinct Details**: Share child's photo, age, clothes color, and language with temple police.\n"
                "5. **Station a Point Person**: Keep one family member stationed at the exact location where the child was separated, as children often try to backtrack."
            )

        # 3. Severe Dehydration / Heat Stroke / Heat Exhaustion
        if re.search(r"\b(dehydrat(?:ed|ion)|heat\s*stroke|heat\s*exhaustion|sunstroke|faint(?:ing|ed)\s+from\s+heat|extreme\s+heat)\b", q_lower):
            return (
                "#### ⚠️ IMMEDIATE FIRST AID — SEVERE DEHYDRATION & HEAT EXHAUSTION:\n"
                "1. **Move Out of Sun**: Immediately move the devotee into a shaded, air-cooled hall, mandapam, or medical tent.\n"
                "2. **Cool the Body**: Loosen tight attire/footwear. Apply cool, damp cloths to neck, forehead, and underarms. Fan continuously.\n"
                "3. **Provide Hydration (If Conscious)**: Give small sips of cool water, ORS (Oral Rehydration Solution), or electrolyte fluid.\n"
                "4. **Emergency Escalation (108)**: If the person is vomiting, delirious, confused, or unable to drink, dial **108** immediately for medical assistance."
            )

        # 4. Flooding / Dangerous Weather / River Inundation
        if re.search(r"\b(flood|flooding|flash\s*flood|waterlogging|cyclone|landslide|river\s+(?:overflow|rising|danger)|dangerous\s+weather)\b", q_lower):
            return (
                "#### ⚠️ DANGEROUS WEATHER & FLOOD SAFETY ADVISORY:\n"
                "1. **Do Not Cross Water**: Never attempt to drive or walk through flooded causeways, overflowing ghats, or submerged road bridges. Turn around immediately.\n"
                "2. **Seek Elevated Concrete Shelter**: Move away from river banks, steep ghat slopes prone to rockfall, and temporary pilgrim sheds. Move to multi-story concrete structures or devasthanam halls.\n"
                "3. **Emergency Rescue**: Dial **112** (National Emergency) or **101** (Fire & Rescue) if stranded or water is rising.\n"
                "4. **Electrical Hazards**: Stay strictly away from downed electrical wires, metal barricades, and transformers."
            )

        # 5. Lost Belongings / Stolen Wallet / Valuables
        if re.search(r"\b(lost\s+(?:belongings|wallet|phone|mobile|bag|purse|luggage|gold|valuables)|stolen|theft|robbery|pickpocket)\b", q_lower):
            return (
                "#### ⚠️ LOST BELONGINGS & ASSISTANCE STEPS:\n"
                "1. **Report to Devasthanam Lost & Found Desk**: Visit the Temple Information Centre or Administrative Office to lodge an entry with token/ticket receipts.\n"
                "2. **Lodge Police GD Entry**: Visit the temple police outpost or dial **100** / **112** to file a Property Missing Report.\n"
                "3. **Block Accounts & SIM**: If mobile phones or wallets were lost, immediately call your bank to freeze ATM/credit cards and contact your telecom operator to suspend the SIM card.\n"
                "4. **Check Cloakrooms**: Confirm with luggage and footwear deposit managers in case items were handed in by temple volunteers."
            )

        return None

    def _build_universal_response(self, query: str = "") -> AgentResponse:
        """Constructs statutory emergency guidance when no specific temple is identified."""
        lines = []
        immediate_action = self._detect_and_build_immediate_action(query)
        if immediate_action:
            lines.extend([
                immediate_action,
                "",
                "---"
            ])

        lines.extend([
            "### 🚨 Statutory National & State Emergency Helplines (India / Telangana)",
            "",
            "If you are facing an immediate life-threatening emergency, please dial the statutory helplines below:",
            ""
        ])
        for h in UNIVERSAL_HELPLINES:
            lines.append(f"• **{h['name']}**: `{h['number']}` — *{h['desc']}*")

        lines.extend([
            "",
            "---",
            "*(To retrieve local hospital addresses, duty doctor clinics, police stations, or devasthanam administrative help desks, please specify your temple. For example: 'Emergency contacts for Yadadri' or 'Hospital near Basara').*"
        ])

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer="\n".join(lines),
            sources=["statutory_emergency_helplines"],
            metadata={"temple": None, "is_universal": True}
        )

    def _run(self, state: AgentState) -> AgentResponse:
        """
        Specialist execution: parses emergency intent, resolves temple, fetches
        verified emergency and contact datasets via RAGTool, and renders a strictly
        grounded, non-hallucinated response.
        """
        query = state.user_query.strip() if state.user_query else ""

        # 1. Resolve Canonical Temple Identity
        canonical_temple: Optional[str] = None

        # Check route_decision metadata first
        route_decision = state.intermediate_results.get("route_decision")
        if route_decision and hasattr(route_decision, "metadata"):
            detected = route_decision.metadata.get("detected_temples") or []
            if detected:
                canonical_temple = detected[0]

        # Check conversation memory resolution
        if not canonical_temple:
            mem = state.intermediate_results.get("memory", {})
            if isinstance(mem, dict) and mem.get("referenced_temple"):
                canonical_temple = mem["referenced_temple"]

        # Check direct regex match on query
        if not canonical_temple:
            canonical_temple = self.detect_temple(query)

        # 2. Check for explicit unsupported/unknown temple
        if not canonical_temple and route_decision and hasattr(route_decision, "metadata"):
            unknown_temple = route_decision.metadata.get("detected_unknown_temple")
            if unknown_temple:
                unsupported_text = (
                    f"I do not have verified emergency records for '{unknown_temple}' in the current PilgrimAI knowledge base. "
                    f"PilgrimAI currently covers 20 registered pilgrimage temples in Telangana.\n\n"
                    f"For immediate emergency assistance anywhere in India:\n"
                    f"• **National Emergency Helpline**: `112`\n"
                    f"• **State Ambulance Service**: `108`\n"
                    f"• **Police Control Room**: `100`\n"
                    f"• **Fire Services**: `101`"
                )
                return AgentResponse(
                    success=True,
                    agent_name=self.name,
                    answer=unsupported_text,
                    sources=[],
                    metadata={"temple": unknown_temple, "supported": False}
                )

        # 3. If no temple resolved, provide universal helplines
        if not canonical_temple:
            return self._build_universal_response(query)

        # 4. Normalize canonical temple name
        canonical_id = TEMPLE_CANONICAL_ALIASES.get(canonical_temple.lower(), canonical_temple)
        display_name = TEMPLE_DISPLAY_NAMES.get(canonical_id, canonical_id)
        temple_folder = canonical_id.lower()

        # 5. Fetch verified datasets strictly via RAGTool facade
        emergency_data = self.rag_tool.get_temple_category_data(canonical_id, "emergency_contacts")
        contacts_data = self.rag_tool.get_temple_category_data(canonical_id, "contacts")

        sources: List[str] = []
        if emergency_data:
            sources.append(f"{temple_folder}/emergency_nos.json")
        if contacts_data:
            sources.append(f"{temple_folder}/contacts.json")

        if not emergency_data and not contacts_data:
            not_found_msg = (
                f"I could not find verified emergency contact information for **{display_name}** "
                f"in the current PilgrimAI knowledge base.\n\n"
                f"For immediate emergency assistance:\n"
                f"• **National Emergency Helpline**: `112`\n"
                f"• **State Ambulance Service**: `108`\n"
                f"• **Police Control Room**: `100`"
            )
            return AgentResponse(
                success=True,
                agent_name=self.name,
                answer=not_found_msg,
                sources=sources,
                metadata={"temple": display_name, "has_records": False}
            )

        # 6. Parse and categorize verified records
        # Police & Security
        police_entries: List[Dict[str, str]] = []
        if isinstance(emergency_data, dict):
            for k in ("police", "police_and_security", "local_police", "local_police_station"):
                if k in emergency_data:
                    police_entries.extend(self._normalize_category_items(emergency_data[k]))

        # Medical & Hospitals
        medical_entries: List[Dict[str, str]] = []
        if isinstance(emergency_data, dict):
            for k in ("medical", "hospitals", "ambulance", "ambulance_services", "medical_emergency_desk"):
                if k in emergency_data:
                    medical_entries.extend(self._normalize_category_items(emergency_data[k]))

        # Fire & Rescue
        fire_entries: List[Dict[str, str]] = []
        if isinstance(emergency_data, dict):
            for k in ("fire", "fire_and_rescue", "fire_and_rescue_services"):
                if k in emergency_data:
                    fire_entries.extend(self._normalize_category_items(emergency_data[k]))

        # Temple Control Room / Command Center / Help Desk
        helpdesk_entries: List[Dict[str, str]] = []
        if isinstance(emergency_data, dict):
            for k in ("temple_control_room", "integrated_command_control", "she_teams_women_safety"):
                if k in emergency_data:
                    helpdesk_entries.extend(self._normalize_category_items(emergency_data[k]))

        # Contacts dataset: Directory & office phones
        office_phones: List[str] = []
        if isinstance(contacts_data, dict):
            directory = contacts_data.get("directory")
            if isinstance(directory, dict):
                for desk, phone in directory.items():
                    if phone and str(phone).strip():
                        office_phones.append(f"• **{desk}**: `{phone}`")
            elif isinstance(contacts_data.get("phones"), list):
                for p in contacts_data["phones"][:4]:
                    office_phones.append(f"• **Devasthanam Line**: `{p}`")

        # 7. Construct clean, structured, non-hallucinated Markdown response
        immediate_action = self._detect_and_build_immediate_action(query)

        output_sections: List[str] = [
            f"### 🚨 Emergency & Medical Information — {display_name}",
            ""
        ]
        if immediate_action:
            output_sections.extend([
                immediate_action,
                "",
                "---",
                f"### 📞 Verified Local & Temple Contacts — {display_name}",
                ""
            ])

        # Medical & Hospital Section
        output_sections.append("#### 🏥 Medical & Hospital Assistance")
        if medical_entries:
            for item in medical_entries:
                line = self._format_contact_line(item)
                if line:
                    output_sections.append(line)
        else:
            output_sections.append("• No dedicated local hospital recorded in temple records. For emergency ambulance, dial `108`.")

        output_sections.append("")

        # Police & Law Enforcement Section
        output_sections.append("#### 👮 Police & Security Services")
        if police_entries:
            for item in police_entries:
                line = self._format_contact_line(item)
                if line:
                    output_sections.append(line)
        else:
            output_sections.append("• Local police station contact not recorded. For immediate police assistance, dial `100`.")

        output_sections.append("")

        # Temple Administrative Office & Help Desk
        output_sections.append("#### 🏛️ Temple Control Room & Help Desk")
        has_desk_info = False
        if helpdesk_entries:
            for item in helpdesk_entries:
                line = self._format_contact_line(item)
                if line:
                    output_sections.append(line)
                    has_desk_info = True

        if office_phones:
            for line in office_phones[:4]:
                output_sections.append(line)
                has_desk_info = True

        if not has_desk_info:
            output_sections.append("• Devasthanam administrative desk phone is not recorded in the verified files.")

        output_sections.append("")

        # Fire & Rescue / Statutory Helplines
        output_sections.append("#### 🚒 Fire, Rescue & Statutory Helplines")
        if fire_entries:
            for item in fire_entries:
                line = self._format_contact_line(item)
                if line:
                    output_sections.append(line)
        output_sections.append("• **National Emergency Number**: `112` | **State Ambulance**: `108` | **Police**: `100` | **Fire**: `101`")

        final_answer = "\n".join(output_sections)

        return AgentResponse(
            success=True,
            agent_name=self.name,
            answer=final_answer,
            sources=sources,
            metadata={
                "temple": display_name,
                "canonical_temple": canonical_id,
                "has_verified_emergency": True,
                "has_medical": bool(medical_entries),
                "has_police": bool(police_entries),
                "has_help_desk": bool(office_phones or helpdesk_entries)
            }
        )
