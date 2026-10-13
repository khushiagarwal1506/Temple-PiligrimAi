import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from app.rag.config import settings
from app.rag.loader import TempleDatasetLoader
from app.rag.utils import format_temple_name
from app.rag.category_registry import category_registry, CategoryRegistry

logger = logging.getLogger(__name__)

class KnowledgeRegistry:
    """
    In-memory knowledge structure index over the 20 temples and ~21 JSON categories.
    Provides instant deterministic lookups for available files, parent IDs, complete parents,
    and category mappings without modifying source data or rebuilding indexes.
    """

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        parent_store_path: Optional[str] = None,
        registry: Optional[CategoryRegistry] = None
    ):
        self.dataset_dir = Path(dataset_dir or settings.DATASET_DIR)
        self.parent_store_path = Path(parent_store_path or (Path(settings.QDRANT_DIR) / "parent_store.json"))
        self.category_registry = registry or category_registry
        self.loader = TempleDatasetLoader(dataset_dir=str(self.dataset_dir))

        self._temple_folders: Dict[str, Path] = {}
        self._temple_categories: Dict[str, Set[str]] = {}
        self._temple_files: Dict[str, Dict[str, List[str]]] = {}
        self._parent_store: Dict[str, str] = {}
        self._initialized = False

        self._discover_and_index()

    def _discover_and_index(self):
        """Discovers all temple directories and indexes category/file mappings."""
        folders = self.loader.discover_temple_folders()
        for f in folders:
            canonical_name = format_temple_name(f.name)
            self._temple_folders[canonical_name] = f
            self._temple_categories[canonical_name] = set()
            self._temple_files[canonical_name] = {}

            # Discover all json files present in this temple directory
            for json_file in f.glob("*.json"):
                rel_path = f"{f.name.lower()}/{json_file.name}"
                stem = json_file.stem.lower()

                # Match stem with registered categories
                matched_cat = stem
                if stem == "elderly_ppl":
                    matched_cat = "wheelchair"
                elif stem == "emergency_nos":
                    matched_cat = "emergency_contacts"

                self._temple_categories[canonical_name].add(matched_cat)
                if matched_cat not in self._temple_files[canonical_name]:
                    self._temple_files[canonical_name][matched_cat] = []
                self._temple_files[canonical_name][matched_cat].append(rel_path)

        # Load parent store if available
        if self.parent_store_path.exists():
            try:
                with open(self.parent_store_path, "r", encoding="utf-8") as f:
                    self._parent_store = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load parent_store from {self.parent_store_path}: {e}")

        self._initialized = True

    def get_temples(self) -> List[str]:
        """Returns list of all discovered canonical temple names (alphabetically sorted)."""
        return sorted(list(self._temple_folders.keys()))

    def _canonical_key(self, temple: str) -> Optional[str]:
        """Resolves any casing or formatting variant to the indexed canonical temple key."""
        if not temple:
            return None
        if temple in self._temple_folders:
            return temple
        t_low = temple.lower().strip()
        for k in self._temple_folders:
            if k.lower().strip() == t_low:
                return k
        for k, f in self._temple_folders.items():
            if f.name.lower().strip() == t_low:
                return k
        # Strip parentheses/spaces for alias tolerance e.g. 'jogulamba(new)' -> 'jogulamba'
        t_clean = re.sub(r"[\(\)]", "", t_low).replace("new", "").strip()
        for k, f in self._temple_folders.items():
            k_clean = re.sub(r"[\(\)]", "", k.lower()).replace("new", "").strip()
            if k_clean == t_clean:
                return k
        return None

    def get_temple_folder(self, temple: str) -> Optional[Path]:
        """Retrieves Path object for a canonical temple (case-insensitive & alias-tolerant)."""
        key = self._canonical_key(temple)
        return self._temple_folders.get(key) if key else None

    def get_categories(self) -> List[str]:
        """Returns all registered category keys."""
        return [c.key for c in self.category_registry.all_categories()]

    def get_categories_for_temple(self, temple: str) -> List[str]:
        """Returns list of categories available for the given temple."""
        key = self._canonical_key(temple) or temple
        cats = self._temple_categories.get(key, set())
        return sorted(list(cats))

    def has_category(self, temple: str, category: str) -> bool:
        """Returns True if the temple has recorded files/data for the category."""
        key = self._canonical_key(temple) or temple
        return category in self._temple_categories.get(key, set())

    def get_source_files(self, temple: str, category: str) -> List[str]:
        """Returns list of relative source file paths for a temple and category."""
        key = self._canonical_key(temple) or temple
        temple_file_map = self._temple_files.get(key, {})
        if category in temple_file_map:
            return temple_file_map[category]

        # Check candidate files from registry as fallback
        folder = self.get_temple_folder(key)
        if not folder:
            return []

        cat_meta = self.category_registry.get_category(category)
        if not cat_meta:
            return []

        found = []
        for cand in cat_meta.candidate_files:
            p = folder / cand
            if p.exists():
                found.append(f"{folder.name.lower()}/{cand}")
        return found

    def get_all_temples_with_category(self, category: str) -> List[str]:
        """Returns list of temples that have data for the specified category."""
        matching = []
        for temple in self.get_temples():
            if self.has_category(temple, category):
                matching.append(temple)
        return matching

    def _format_raw_category_data(self, temple: str, category: str, data: Any) -> str:
        """
        Formats raw category JSON into clean readable document text.
        Internal RAG fallback helper to avoid exposing raw JSON directly to agents.
        """
        cat_title = category.replace("_", " ").title()
        lines = [f"Temple: {temple}", f"Category: {cat_title}", ""]
        if isinstance(data, list):
            for idx, item in enumerate(data, 1):
                if isinstance(item, dict):
                    name = item.get("name") or item.get("festival_name") or item.get("title") or f"Item {idx}"
                    parts = [f"{idx}. {name}"]
                    for k, v in item.items():
                        if k not in ("name", "festival_name", "title") and v:
                            parts.append(f"   - {k.replace('_', ' ').title()}: {v}")
                    lines.append("\n".join(parts))
                else:
                    lines.append(f"- {item}")
        elif isinstance(data, dict):
            for k, v in data.items():
                k_title = k.replace("_", " ").title()
                if isinstance(v, list):
                    lines.append(f"{k_title}:")
                    for sub in v:
                        lines.append(f"  - {sub}")
                elif isinstance(v, dict):
                    lines.append(f"{k_title}:")
                    for sub_k, sub_v in v.items():
                        lines.append(f"  - {sub_k.replace('_', ' ').title()}: {sub_v}")
                else:
                    lines.append(f"{k_title}: {v}")
        else:
            lines.append(str(data))
        return "\n".join(lines)

    def get_complete_parent(self, temple: str, category: str) -> Optional[str]:
        """
        Retrieves the unified complete parent text from parent_store if indexed.
        Checks canonical complete parent ID conventions, matching parent prefix keys,
        and safely falls back to formatted raw records if indexed parent is absent.
        """
        clean_temple = temple.lower().replace(" ", "_")
        clean_cat = category.lower().replace(" ", "_")
        
        # 1. Check standard complete parent ID format
        primary_id = f"parent_{clean_temple}_{clean_cat}_complete"
        if primary_id in self._parent_store:
            return self._parent_store[primary_id]

        # 2. Check alternate alias IDs (e.g. wheelchair -> elderly_ppl)
        alt_cats = []
        if category == "wheelchair":
            alt_cats.append("elderly_ppl")
        elif category == "emergency_contacts":
            alt_cats.append("emergency_nos")
        elif category == "contacts":
            alt_cats.append("contact")

        for alt_cat in alt_cats:
            alt_id = f"parent_{clean_temple}_{alt_cat}_complete"
            if alt_id in self._parent_store:
                return self._parent_store[alt_id]

        # 3. Check for matching prefix parent keys in parent_store (e.g. single-parent hashed files or multi-parents)
        search_prefixes = [f"parent_{clean_temple}_{clean_cat}_"] + [f"parent_{clean_temple}_{ac}_" for ac in alt_cats]
        matching_keys = []
        for pfx in search_prefixes:
            matched = [k for k in self._parent_store if k.startswith(pfx)]
            if matched:
                matching_keys.extend(matched)

        if matching_keys:
            sorted_keys = sorted(list(set(matching_keys)))
            return "\n\n".join(self._parent_store[k] for k in sorted_keys)

        # 4. Safe internal fallback to verified source JSON via get_temple_raw_data
        raw_data = self.get_temple_raw_data(temple, category)
        if raw_data is not None:
            return self._format_raw_category_data(temple, category, raw_data)

        return None

    def get_temple_raw_data(self, temple: str, category: str) -> Optional[Any]:
        """
        Reads source JSON data for a temple and category directly from disk.
        Read-only operation.
        """
        folder = self.get_temple_folder(temple)
        if not folder:
            return None

        cat_meta = self.category_registry.get_category(category)
        candidates = cat_meta.candidate_files if cat_meta else [f"{category}.json"]

        for cand in candidates:
            file_path = folder / cand
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
        return None

knowledge_registry = KnowledgeRegistry()
