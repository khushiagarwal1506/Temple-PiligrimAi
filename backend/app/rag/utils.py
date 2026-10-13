import os
import re
from typing import Dict, Any, List

def format_temple_name(dir_name: str) -> str:
    """Format a directory name into a title-cased temple name."""
    cleaned = dir_name.strip()
    return cleaned.title()

def clean_text(text: str) -> str:
    """Clean and normalize whitespace in text while preserving newlines."""
    if not isinstance(text, str):
        return str(text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()

TEMPLE_CANONICAL_DISPLAY: Dict[str, str] = {
    "Ahobilam": "Ahobilam Sri Lakshmi Narasimha Swamy Temple",
    "Badrachalam": "Bhadrachalam Sita Ramachandra Swamy Temple",
    "Balkampet": "Balkampet Yellamma Temple",
    "Basara": "Basara Gnana Saraswathi Temple",
    "Birla Mandir": "Birla Mandir, Hyderabad",
    "Chilkur": "Chilkur Balaji Temple",
    "Dharmapuri": "Dharmapuri Lakshmi Narasimha Swamy Temple",
    "Edupayal": "Edupayala Vana Durga Bhavani Temple",
    "Jogulamba(New)": "Jogulamba Temple, Alampur",
    "Jogulamba(new)": "Jogulamba Temple, Alampur",
    "Jogulamba": "Jogulamba Temple, Alampur",
    "Kaleshvaram": "Kaleshwaram Sri Kaleshwara Mukteshwara Swamy Temple",
    "Karmanghat": "Karmanghat Hanuman Temple",
    "Medaram": "Medaram Sammakka Saralamma Temple",
    "Ramappa": "Ramappa Temple, Palampet",
    "Srisaila": "Srisailam Mallikarjuna Swamy Temple",
    "Thousand Pillar": "Thousand Pillar Temple, Hanamkonda",
    "Ttd": "Tirumala Venkateswara Temple",
    "Ujjaini": "Ujjaini Mahakali Temple, Secunderabad",
    "Veera Satyanarayana": "Annavaram Veera Venkata Satyanarayana Swamy Temple",
    "Vemulavada": "Vemulawada Sri Raja Rajeshwara Swamy Temple",
    "Yadadri": "Yadadri Sri Lakshmi Narasimha Swamy Temple"
}

def get_canonical_display_name(temple_name: str) -> str:
    """
    Returns clean user-facing canonical display name for any temple.
    Guarantees internal directory labels like 'Jogulamba(new)' never leak.
    """
    if not temple_name or not isinstance(temple_name, str):
        return ""
    clean = temple_name.strip()
    for k, v in TEMPLE_CANONICAL_DISPLAY.items():
        if clean.lower() == k.lower():
            return v
    if "jogulamba" in clean.lower():
        return "Jogulamba Temple, Alampur"
    return clean

def sanitize_json_value(val: Any) -> str:
    """Convert any JSON value into a human-readable clean string representation."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return val.strip()
    return str(val)

