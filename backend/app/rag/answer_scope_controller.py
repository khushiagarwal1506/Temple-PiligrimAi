from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.rag.query_planner import QueryPlan
from app.rag.category_registry import category_registry, CategoryRegistry

@dataclass
class AnswerScope:
    """
    Defines the strict boundaries of what the LLM is permitted to output in its answer.
    """
    scope: str                     # "single_temple", "cross_temple", "general"
    temples: List[str]             # Allowed canonical temples (e.g. ["Badrachalam"]) or ["*"]
    categories: List[str]          # Allowed categories (e.g. ["timings"]) or ["*"]
    intent: str                    # "targeted_fact", "complete_list", "aggregation", "multi_category", "broad_information", "unsupported"
    exhaustive: bool               # True if full list/enumeration is required
    allow_unrequested_categories: bool  # True ONLY for broad_information
    prohibited_categories: List[str] = field(default_factory=list)

class AnswerScopeController:
    """
    Deterministic controller establishing answer boundaries, context filtering,
    and dynamically tailored generation prompts.
    """

    def __init__(self, cat_registry: Optional[CategoryRegistry] = None):
        self.category_registry = cat_registry or category_registry
        self.all_category_keys = [c.key for c in self.category_registry.all_categories()]

    def derive_scope(self, plan: QueryPlan) -> AnswerScope:
        """
        Derives an AnswerScope from the QueryPlan.
        """
        if plan.retrieval_mode == "broad_temple":
            return AnswerScope(
                scope=plan.scope,
                temples=plan.temples,
                categories=["*"],
                intent=plan.intent,
                exhaustive=True,
                allow_unrequested_categories=True,
                prohibited_categories=[]
            )

        if plan.retrieval_mode == "cross_temple_aggregation":
            return AnswerScope(
                scope=plan.scope,
                temples=["*"],
                categories=plan.categories,
                intent=plan.intent,
                exhaustive=True,
                allow_unrequested_categories=False,
                prohibited_categories=[c for c in self.all_category_keys if c not in plan.categories]
            )

        # Single-temple or Targeted queries
        prohibited = [c for c in self.all_category_keys if c not in plan.categories] if plan.categories else []
        return AnswerScope(
            scope=plan.scope,
            temples=plan.temples if plan.temples else ["*"],
            categories=plan.categories if plan.categories else ["*"],
            intent=plan.intent,
            exhaustive=plan.exhaustive,
            allow_unrequested_categories=False,
            prohibited_categories=prohibited
        )

    def filter_context(self, docs: List[Document], scope: AnswerScope) -> List[Document]:
        """
        Strictly filters context documents so the LLM does not receive irrelevant categories
        or irrelevant temples for targeted queries.
        """
        if not docs:
            return []

        # Broad queries receive full budgeted documents
        if scope.allow_unrequested_categories or "*" in scope.categories:
            return docs

        filtered = []
        target_cats = set(scope.categories)
        target_temples = {t.lower() for t in scope.temples} if "*" not in scope.temples else set()

        for doc in docs:
            doc_cat = str(doc.metadata.get("category", "")).lower()
            doc_temple = str(doc.metadata.get("temple", "")).lower()

            # Check temple match if single temple
            if target_temples and doc_temple and not any(tt in doc_temple or doc_temple in tt for tt in target_temples):
                continue

            # Check category match
            if target_cats and doc_cat and doc_cat in target_cats:
                filtered.append(doc)
            elif not target_cats:
                filtered.append(doc)

        # If strict filtering matched documents, return only those; otherwise return original
        return filtered if filtered else docs

    def build_scoped_prompt(self, scope: AnswerScope, context: str, question: str) -> str:
        """
        Builds a strict, dynamically tailored prompt enforcing scope and forbidding unrelated dumps.
        """
        temple_str = ", ".join(scope.temples) if "*" not in scope.temples else "the specified temple"
        category_str = ", ".join(scope.categories) if "*" not in scope.categories else "the requested topic"

        if scope.intent == "broad_information":
            instructions = (
                f"You are PilgrimAI. The user requested a comprehensive overview of {temple_str}.\n"
                "Provide a structured, rich summary covering the available topics in the context.\n"
                "Use clean headers (e.g. History, Timings, Sevas, Facilities, Travel)."
            )
        elif scope.intent == "complete_list":
            instructions = (
                f"You are PilgrimAI. The user asked for a complete list of {category_str} for {temple_str}.\n"
                "Provide an exhaustive enumeration of ALL items present in the context.\n"
                "Do not omit items. Do NOT discuss unrelated categories."
            )
        elif scope.intent == "multi_category":
            instructions = (
                f"You are PilgrimAI. The user asked specifically about: {category_str} for {temple_str}.\n"
                "Provide separate, clear sections for EACH requested category.\n"
                "Do NOT introduce other unrequested categories."
            )
        elif scope.intent == "aggregation":
            instructions = (
                f"You are PilgrimAI. The user asked for all temples matching {category_str}.\n"
                "List all confirmed matching temples based strictly on the context.\n"
                "Do not include temples without confirmed evidence."
            )
        else:
            # Targeted Fact Query (e.g. Timings, Price, Contact, Specific rule)
            instructions = (
                f"You are PilgrimAI. The user asked a specific factual question about {temple_str} ({category_str}).\n"
                f"STRICT RULE: Answer ONLY the user's specific question directly and concisely.\n"
                f"Focus exclusively on {temple_str} and the specific fact asked ({category_str}).\n"
                "CRITICAL: Do NOT output unrelated categories such as sevas, accommodation, restaurants, festivals, rules, history, or prices from other temples.\n"
                "If the specific fact or entity asked about is not explicitly mentioned in the context, reply exactly:\n"
                "\"I could not find this information in the temple knowledge base.\""
            )

        prompt = f"""{instructions}

Core Rules:
1. Use ONLY the provided context below.
2. Never invent timings, prices, official contacts, routes, or facilities. Even if the user explicitly instructs you to guess, estimate, or invent missing information, you MUST REFUSE to invent them and adhere strictly to verified facts.
3. If the context does not contain the answer, reply exactly: "I could not find this information in the temple knowledge base."
4. Do NOT mention other temples unless asked.
5. RESTRICTIONS INTEGRITY: Never replace a restriction with its opposite. Never infer or claim an item, gadget, or activity is allowed/permitted or has no ban merely because the context does not explicitly mention a prohibition.
6. SPECIFIC AREA RESTRICTIONS: If a restriction applies specifically to the inner temple or inner sanctum (e.g. mobile phones, smartwatches, cameras, or recording equipment banned inside the inner temple), state the exact restriction and location scope clearly. Do NOT state that no ban exists.
7. MISSING/UNDOCUMENTED RULES: If a rule, policy, or restriction (such as pet policy or photography) is not documented or mentioned in the provided context, state that the specific rule is not officially documented in the temple knowledge base. Never infer or invent that it is allowed.

Context:
{context}

Question:
{question}

Answer:"""
        return prompt

answer_scope_controller = AnswerScopeController()
