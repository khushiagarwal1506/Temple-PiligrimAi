"""
PilgrimAI - Read-Only Temple Coordinate Registry (Phase O2)

Provides a lightweight, strictly read-only repository of verified geographical coordinates
and provenance metadata for all 20 canonical pilgrimage temples in the PilgrimAI ecosystem.

Core Principles:
- Zero mutation of the 417 source temple JSON files.
- Zero runtime dependencies on vector databases, BM25, or RAG models.
- Strict coordinate validation: latitude [-90.0, 90.0], longitude [-180.0, 180.0].
- Transparent provenance preservation (derived_from_temple_maps_json vs needs_verification).
- Case-insensitive and whitespace-resilient canonical and alias resolution.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Default path to the static coordinate asset
_DEFAULT_COORDINATE_FILE = Path(__file__).resolve().parent / "data" / "temple_coordinates.json"

# Canonical Temple Aliases (strictly decoupled from retrieval packages to prevent circular/heavy dependencies)
_CANONICAL_TEMPLE_ALIASES: Dict[str, List[str]] = {
    "Ahobilam": ["ahobilam", "ahobila", "narasimha ahobilam", "sri ahobila narasimha"],
    "Badrachalam": ["badrachalam", "bhadrachalam", "bhadragiri", "rama temple bhadrachalam", "lord rama bhadrachalam", "sree seetha ramachandra"],
    "Balkampet": ["balkampet", "balkampet temple", "yellamma", "balkampet yellamma", "yellamma temple", "balkampet yellamma temple"],
    "Basara": ["basara", "basar", "gnana saraswathi", "saraswathi temple", "sri gnana saraswathi devasthanam"],
    "Birla Mandir": ["birla mandir", "birla", "birla temple"],
    "Chilkur": ["chilkur", "visa balaji", "chilkur balaji", "chilkur balaji temple"],
    "Dharmapuri": ["dharmapuri", "narasimha swamy dharmapuri", "dharmapuri narasimha", "sri lakshmi narasimha swami dharmapuri"],
    "Edupayal": ["edupayal", "edupayala", "edupayala durga", "vanadurga bhavani", "sri edupayala vana durga"],
    "Jogulamba(New)": ["jogulamba", "jogulamba(new)", "alampur", "jogulamba devi", "alampur jogulamba"],
    "Kaleshvaram": ["kaleshvaram", "kaleshwaram", "mukteshwara", "kaleshwara", "sri kaleshwara mukteshwara"],
    "Karmanghat": ["karmanghat", "karmaghat", "karmanghat hanuman", "hanuman temple karmanghat"],
    "Medaram": ["medaram", "sammakka", "saralamma", "jatara medaram", "sammakka saralamma", "sammakka saralamma gaddelu"],
    "Ramappa": ["ramappa", "ramalingeswara", "rudreshwara", "ramappa temple"],
    "Srisaila": ["srisaila", "srisailam", "mallikarjuna", "bhramaramba", "srisailam temple"],
    "Thousand Pillar": ["thousand pillar", "thousand pillars", "rudreswara thousand", "thousand pillar temple"],
    "Ttd": ["ttd", "tirumala", "tirupati", "venkateswara ttd", "lord venkateswara", "tirumala venkateswara"],
    "Ujjaini": [
        "ujjaini", "ujjaini mahakali", "ujjaini mahankali", "mahakali secunderabad",
        "secunderabad mahakali", "secunderabad mahankali", "sri ujjaini mahakali",
        "sri ujjaini mahankali", "ujjaini mahakali temple", "ujjaini mahankali temple",
        "sri ujjaini mahakali temple", "sri ujjaini mahankali temple", "ujjaini temple",
        "ujjaini devasthanam"
    ],
    "Veera Satyanarayana": ["veera satyanarayana", "satyanarayana swamy", "annavaram", "annavarm", "veera venkata satyanarayana", "annavaram satyanarayana"],
    "Vemulavada": ["vemulavada", "vemulawada", "rajarajeshwara", "rajanne", "rajanna", "sri raja rajeshwara"],
    "Yadadri": ["yadadri", "yadagirigutta", "laxmi narasimha yadadri", "narasimha yadadri", "yadagirigutta temple"]
}


class TempleCoordinate(BaseModel):
    """
    Immutable data model representing verified geographical coordinates and provenance
    for a canonical pilgrimage temple.
    """
    canonical_name: str = Field(..., description="Project canonical temple identifier")
    latitude: float = Field(..., description="Geographical latitude in decimal degrees")
    longitude: float = Field(..., description="Geographical longitude in decimal degrees")
    display_name: str = Field(default="", description="Formal display name of the temple")
    district: str = Field(default="", description="Administrative district")
    state: str = Field(default="", description="Indian state (Telangana / Andhra Pradesh)")
    source: str = Field(..., description="Origin and citation of the coordinate data")
    verification_status: str = Field(
        ...,
        description="Verification state: 'derived_from_temple_maps_json' or 'needs_verification'"
    )

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if v < -90.0 or v > 90.0:
            raise ValueError(f"Latitude {v} is outside valid bounds [-90.0, 90.0].")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if v < -180.0 or v > 180.0:
            raise ValueError(f"Longitude {v} is outside valid bounds [-180.0, 180.0].")
        return v

    @field_validator("canonical_name")
    @classmethod
    def validate_canonical_name(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Canonical name cannot be empty.")
        return s


class TempleCoordinateRegistry:
    """
    Read-only registry for querying canonical temple coordinates, verifying provenance,
    and resolving natural language temple aliases to canonical coordinates.
    """

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self._data_path = Path(data_path) if data_path else _DEFAULT_COORDINATE_FILE
        self._coordinates: Dict[str, TempleCoordinate] = {}
        self._alias_map: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Loads and validates the static coordinate dataset from disk."""
        if not self._data_path.exists():
            raise FileNotFoundError(f"Coordinate registry data file not found at: {self._data_path}")

        try:
            with open(self._data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse coordinate registry JSON from {self._data_path}: {e}")

        if not isinstance(raw_data, dict):
            raise ValueError(f"Coordinate registry root element must be a JSON object, got {type(raw_data).__name__}.")

        loaded: Dict[str, TempleCoordinate] = {}
        for key, entry in raw_data.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Coordinate entry for '{key}' must be a dictionary, got {type(entry).__name__}.")

            canonical_name = entry.get("canonical_name", key).strip()
            if not canonical_name:
                raise ValueError(f"Coordinate entry for '{key}' has missing or empty canonical_name.")

            if canonical_name in loaded:
                raise ValueError(f"Duplicate canonical temple name '{canonical_name}' detected in registry.")

            # Validate via Pydantic model
            try:
                coord = TempleCoordinate(**entry)
            except Exception as e:
                raise ValueError(f"Validation error for temple coordinate '{canonical_name}': {e}")

            loaded[canonical_name] = coord

        self._coordinates = loaded
        self._init_alias_map()
        logger.info(f"Loaded {len(self._coordinates)} temple coordinates into read-only registry.")

    def _init_alias_map(self) -> None:
        """Builds normalized reverse lookup table from canonical names and known aliases."""
        alias_map: Dict[str, str] = {}

        # 1. Map lowercased canonical names
        for canonical_name in self._coordinates:
            norm = canonical_name.strip().lower()
            alias_map[norm] = canonical_name

        # 2. Map aliases, checking for collisions
        for canonical_name, aliases in _CANONICAL_TEMPLE_ALIASES.items():
            if canonical_name in self._coordinates:
                for alias in aliases:
                    norm = alias.strip().lower()
                    if norm in alias_map and alias_map[norm] != canonical_name:
                        raise ValueError(
                            f"Alias collision detected: '{norm}' maps to both '{alias_map[norm]}' and '{canonical_name}'"
                        )
                    alias_map[norm] = canonical_name

        self._alias_map = alias_map

    def resolve_alias(self, temple_name: str) -> Optional[str]:
        """
        Deterministically resolves a natural language temple name or alias to its
        canonical identifier. Returns None if unknown or ambiguous.
        """
        if not temple_name or not isinstance(temple_name, str):
            return None

        clean = temple_name.strip().lower()
        if not clean:
            return None

        # Direct alias/canonical match
        if clean in self._alias_map:
            return self._alias_map[clean]

        # Check stripped punctuation/parentheses variants
        clean_no_punct = clean.replace("(", "").replace(")", "").replace("-", " ")
        if clean_no_punct in self._alias_map:
            return self._alias_map[clean_no_punct]

        # Check common religious suffix stripping (temple, mandir, devasthanam, etc.)
        stripped = re.sub(r"\s+(?:temple|mandir|devasthanam|devalayam|kovil|gudi|matha|swamy)\s*$", "", clean).strip()
        if stripped and stripped in self._alias_map:
            return self._alias_map[stripped]

        stripped_no_punct = re.sub(r"\s+(?:temple|mandir|devasthanam|devalayam|kovil|gudi|matha|swamy)\s*$", "", clean_no_punct).strip()
        if stripped_no_punct and stripped_no_punct in self._alias_map:
            return self._alias_map[stripped_no_punct]

        return None

    def has_coordinate(self, temple_name: str, only_verified: bool = False) -> bool:
        """
        Checks whether a canonical coordinate exists for the given name or alias.
        If only_verified is True, returns True only if verification_status == 'derived_from_temple_maps_json'.
        """
        canonical = self.resolve_alias(temple_name)
        if not canonical or canonical not in self._coordinates:
            return False

        if only_verified:
            coord = self._coordinates[canonical]
            return coord.verification_status == "derived_from_temple_maps_json"

        return True

    def get_coordinate(self, temple_name: str, only_verified: bool = False) -> Optional[TempleCoordinate]:
        """
        Retrieves the TempleCoordinate model for a temple name or alias.
        Returns a fresh copy to guarantee runtime immutability.
        """
        canonical = self.resolve_alias(temple_name)
        if not canonical or canonical not in self._coordinates:
            return None

        coord = self._coordinates[canonical]
        if only_verified and coord.verification_status != "derived_from_temple_maps_json":
            return None

        # Return a copy to prevent in-place mutation
        return coord.model_copy()

    def get_all_coordinates(self, only_verified: bool = False) -> Dict[str, TempleCoordinate]:
        """
        Returns a dictionary of all registered coordinates keyed by canonical name.
        """
        result = {}
        for canonical, coord in self._coordinates.items():
            if only_verified and coord.verification_status != "derived_from_temple_maps_json":
                continue
            result[canonical] = coord.model_copy()
        return result

    def get_provenance(self, temple_name: str) -> Optional[Dict[str, Any]]:
        """
        Returns structured provenance details for the specified temple.
        """
        coord = self.get_coordinate(temple_name)
        if not coord:
            return None

        return {
            "canonical_name": coord.canonical_name,
            "latitude": coord.latitude,
            "longitude": coord.longitude,
            "source": coord.source,
            "verification_status": coord.verification_status,
            "district": coord.district,
            "state": coord.state
        }


# Global singleton instance for easy import and zero-overhead reuse
coordinate_registry = TempleCoordinateRegistry()
