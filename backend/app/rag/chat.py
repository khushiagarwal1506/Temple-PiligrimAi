import logging
import json
import requests
import re
from typing import Dict, Any, List, Optional, Set
from langchain_core.documents import Document

from app.rag.config import settings
from app.rag.embeddings import get_embeddings_model
from app.rag.vectordb import LocalQdrantManager
from app.rag.bm25 import LocalBM25Manager
from app.rag.rrf import reciprocal_rank_fusion
from app.rag.reranker import LocalCrossEncoderReranker
from app.rag.guardrail import RelevanceGuardrail, DEFAULT_REFUSAL_MESSAGE
from app.rag.prompt import get_chat_prompt_template
from app.rag.llm import LocalLLMService
from app.agents.cross_temple_aggregation_agent import CrossTempleAggregationAgent
from app.rag.category_registry import category_registry, CategoryRegistry
from app.rag.knowledge_registry import knowledge_registry, KnowledgeRegistry
from app.rag.query_planner import query_planner, QueryPlanner, QueryPlan
from app.rag.completeness_validator import (
    completeness_validator,
    CompletenessValidator,
    CONFIRMED_AVAILABLE,
    CONFIRMED_UNAVAILABLE,
    INFORMATION_NOT_AVAILABLE
)
from app.rag.context_builder import context_builder, ContextBuilder
from app.rag.answer_scope_controller import answer_scope_controller, AnswerScopeController, AnswerScope
from app.rag.answer_grounding_validator import answer_grounding_validator, AnswerGroundingValidator

logger = logging.getLogger(__name__)

class PilgrimRAGService:
    """
    Main 100% Local RAG Service for PilgrimAI Telangana Temples.
    Universal Retrieval Orchestrator supporting:
    - Route 1: Targeted Query (Frozen Top-3 RAG baseline with strict answer scoping)
    - Route 2: Complete Category Retrieval (Single-temple exhaustive inventory)
    - Route 3: Cross-Temple Aggregation (20-temple positive/negative aggregation)
    - Route 4: Multi-Category Retrieval (Balanced multi-topic context)
    - Route 5: Broad Temple Overview (Priority-budgeted structured overview)
    """

    def __init__(
        self,
        planner: Optional[QueryPlanner] = None,
        k_registry: Optional[KnowledgeRegistry] = None,
        c_registry: Optional[CategoryRegistry] = None,
        validator: Optional[CompletenessValidator] = None,
        builder: Optional[ContextBuilder] = None,
        scope_ctrl: Optional[AnswerScopeController] = None,
        grounding_val: Optional[AnswerGroundingValidator] = None
    ):
        self.embeddings = get_embeddings_model()
        self.qdrant_manager = LocalQdrantManager()
        self.bm25_manager = LocalBM25Manager()
        self.bm25_manager.load_index()
        self.parent_store = self.qdrant_manager.load_parent_store()
        self.reranker = LocalCrossEncoderReranker()
        self.guardrail = RelevanceGuardrail()
        self.prompt_template = get_chat_prompt_template()
        self.llm = LocalLLMService()

        # Orchestration layer components
        self.query_planner = planner or query_planner
        self.knowledge_registry = k_registry or knowledge_registry
        self.category_registry = c_registry or category_registry
        self.completeness_validator = validator or completeness_validator
        self.context_builder = builder or context_builder
        self.scope_controller = scope_ctrl or answer_scope_controller
        self.grounding_validator = grounding_val or answer_grounding_validator
        self.aggregation_agent = CrossTempleAggregationAgent(
            cat_registry=self.category_registry,
            k_registry=self.knowledge_registry,
            validator=self.completeness_validator,
            builder=self.context_builder,
            planner=self.query_planner
        )

    def _detect_temple(self, query: str) -> Optional[str]:
        """
        Delegates temple detection to QueryPlanner's word-boundary matching.
        """
        return self.query_planner.detect_temple(query)

    def _detect_list_category(self, query: str) -> Optional[str]:
        """
        Backward-compatible helper for detecting list-oriented categories.
        """
        plan = self.query_planner.plan(query)
        if plan.intent == "complete_list" and plan.categories:
            return plan.categories[0]
        return None

    def retrieve(self, query: str, top_k: int = settings.FINAL_RERANK_TOP_K) -> List[Document]:
        """
        Retrieves relevant documents according to the QueryPlan.
        - cross_temple_aggregation: returns all matching temple documents (no Top-3 truncation)
        - complete_category: returns complete unified parent document
        - multi_category: returns documents for each requested category
        - broad_temple: returns structured category documents for the temple
        - targeted: executes frozen hybrid Qdrant+BM25 -> RRF -> Cross-Encoder Top-3 path
        """
        if not query or not query.strip():
            return []

        plan = self.query_planner.plan(query)

        # -------------------------------------------------------------
        # Route 1: Cross-Temple Aggregation Retrieval
        # -------------------------------------------------------------
        if plan.retrieval_mode == "cross_temple_aggregation" and plan.categories:
            prop_key = plan.categories[0]
            agg_res = self.aggregation_agent.aggregate_temples(prop_key, condition=plan.condition, query=query)
            docs = []
            for m in agg_res["matches"]:
                docs.append(Document(
                    page_content=f"Temple: {m['temple']}\nProperty: {prop_key}\nEvidence: {m['evidence']}",
                    metadata={
                        "temple": m["temple"],
                        "path": m["source_file"],
                        "category": prop_key,
                        "is_aggregation": True,
                        "rerank_score": 10.0
                    }
                ))
            return docs

        # -------------------------------------------------------------
        # Route 2: Complete-Category Retrieval for Single-Temple List
        # -------------------------------------------------------------
        if plan.retrieval_mode == "complete_category" and plan.temples and plan.categories:
            filter_temple = plan.temples[0]
            cat = plan.categories[0]
            parent_text = self.knowledge_registry.get_complete_parent(filter_temple, cat)
            if parent_text:
                src_files = self.knowledge_registry.get_source_files(filter_temple, cat)
                rel_path = src_files[0] if src_files else f"{filter_temple.lower()}/{cat}.json"
                clean_temple = filter_temple.lower().replace(" ", "_")
                clean_cat = cat.lower().replace(" ", "_")
                return [
                    Document(
                        page_content=parent_text,
                        metadata={
                            "temple": filter_temple,
                            "category": cat,
                            "path": rel_path,
                            "parent_id": f"parent_{clean_temple}_{clean_cat}_complete",
                            "is_parent_context": True,
                            "is_complete_list": True,
                            "rerank_score": 10.0
                        }
                    )
                ]

        # -------------------------------------------------------------
        # Route 3: Multi-Category Retrieval
        # -------------------------------------------------------------
        if plan.retrieval_mode == "multi_category" and plan.temples and plan.categories:
            filter_temple = plan.temples[0]
            multi_docs = []
            for cat in plan.categories:
                parent_text = self.knowledge_registry.get_complete_parent(filter_temple, cat)
                src_files = self.knowledge_registry.get_source_files(filter_temple, cat)
                rel_path = src_files[0] if src_files else f"{filter_temple.lower()}/{cat}.json"
                if parent_text:
                    multi_docs.append(Document(
                        page_content=parent_text,
                        metadata={
                            "temple": filter_temple,
                            "category": cat,
                            "path": rel_path,
                            "is_parent_context": True,
                            "rerank_score": 10.0
                        }
                    ))
                else:
                    cat_docs = self.retrieve_with_metadata(query, temple=filter_temple, category=cat, top_k=2)
                    multi_docs.extend(cat_docs)
            return multi_docs

        # -------------------------------------------------------------
        # Route 4: Broad Temple Retrieval
        # -------------------------------------------------------------
        if plan.retrieval_mode == "broad_temple" and plan.temples:
            filter_temple = plan.temples[0]
            broad_docs = []
            for cat in ["temple", "timings", "sevas", "festivals", "facilities", "travel", "accommodation"]:
                parent_text = self.knowledge_registry.get_complete_parent(filter_temple, cat)
                if parent_text:
                    src_files = self.knowledge_registry.get_source_files(filter_temple, cat)
                    broad_docs.append(Document(
                        page_content=parent_text,
                        metadata={
                            "temple": filter_temple,
                            "category": cat,
                            "path": src_files[0] if src_files else f"{filter_temple.lower()}/{cat}.json",
                            "is_parent_context": True,
                            "rerank_score": 10.0
                        }
                    ))
            if broad_docs:
                return broad_docs

        # -------------------------------------------------------------
        # Route 5: Normal Targeted Query Retrieval (Frozen Hybrid Top-3 Baseline)
        # -------------------------------------------------------------
        filter_temple = plan.temples[0] if plan.temples else None

        dense_candidates = self.qdrant_manager.search_candidates(
            query=query,
            embeddings=self.embeddings,
            top_k=settings.CANDIDATE_POOL_TOP_K,
            filter_temple=filter_temple
        )

        bm25_candidates = self.bm25_manager.search_candidates(
            query=query,
            top_k=settings.CANDIDATE_POOL_TOP_K
        )
        if filter_temple:
            bm25_candidates = [
                d for d in bm25_candidates
                if str(d.metadata.get("temple", "")).lower() == filter_temple.lower()
            ]

        fused_candidates = reciprocal_rank_fusion(
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
            top_k=settings.CANDIDATE_POOL_TOP_K
        )

        if not fused_candidates:
            return []

        reranked_child_docs = self.reranker.rerank(
            query=query,
            candidates=fused_candidates,
            top_k=top_k
        )

        resolved_docs = []
        seen_parent_ids = set()

        for child_doc in reranked_child_docs:
            parent_id = child_doc.metadata.get("parent_id")
            parent_text = self.parent_store.get(parent_id) if parent_id else None

            if parent_text and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                meta = child_doc.metadata.copy()
                meta["is_parent_context"] = True
                resolved_docs.append(Document(page_content=parent_text, metadata=meta))
            elif not parent_text:
                resolved_docs.append(child_doc)

        return resolved_docs[:top_k]

    def retrieve_with_metadata(
        self,
        query: str,
        temple: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = settings.FINAL_RERANK_TOP_K
    ) -> List[Document]:
        """
        Retrieves candidate documents filtered by optional temple or category metadata.
        """
        filter_temple = temple.lower() if temple else None

        dense_candidates = self.qdrant_manager.search_candidates(
            query=query,
            embeddings=self.embeddings,
            top_k=settings.CANDIDATE_POOL_TOP_K,
            filter_temple=filter_temple
        )

        bm25_candidates = self.bm25_manager.search_candidates(
            query=query,
            top_k=settings.CANDIDATE_POOL_TOP_K
        )

        if filter_temple:
            bm25_candidates = [
                d for d in bm25_candidates
                if str(d.metadata.get("temple", "")).lower() == filter_temple.lower()
            ]

        fused = reciprocal_rank_fusion(dense_candidates, bm25_candidates, top_k=settings.CANDIDATE_POOL_TOP_K)
        reranked = self.reranker.rerank(query, fused, top_k=top_k)

        if category:
            cat_clean = category.lower()
            reranked = [
                d for d in reranked
                if cat_clean in str(d.metadata.get("category", "")).lower()
            ]

        return reranked[:top_k]

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Executes end-to-end local offline RAG pipeline via QueryPlanner routing
        and strict AnswerScope enforcement.
        """
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        # Plan retrieval strategy and derive strict AnswerScope
        plan = self.query_planner.plan(cleaned_question)
        scope = self.scope_controller.derive_scope(plan)

        # -------------------------------------------------------------
        # Route 1: Cross-Temple Aggregation (All 20 Temples)
        # -------------------------------------------------------------
        if plan.retrieval_mode == "cross_temple_aggregation" and plan.categories:
            prop_key = plan.categories[0]
            agg_res = self.aggregation_agent.aggregate_temples(prop_key, condition=plan.condition, query=cleaned_question)
            formatted_ans = self.aggregation_agent.format_response(agg_res, cleaned_question)
            return {
                "question": cleaned_question,
                "answer": formatted_ans,
                "retrieved_chunks": agg_res["matching_temples_count"],
                "sources": agg_res["sources"],
                "confidence_score": 10.0,
                "is_aggregation": True,
                "plan": plan,
                "scope": scope
            }

        # -------------------------------------------------------------
        # Route 2: Single-Temple Complete-Category Path
        # -------------------------------------------------------------
        if plan.retrieval_mode == "complete_category" and plan.temples and plan.categories:
            filter_temple = plan.temples[0]
            cat = plan.categories[0]
            parent_text = self.knowledge_registry.get_complete_parent(filter_temple, cat)
            src_files = self.knowledge_registry.get_source_files(filter_temple, cat)
            rel_path = src_files[0] if src_files else f"{filter_temple.lower()}/{cat}.json"

            if parent_text:
                context_str = self.context_builder.build_complete_category_context(
                    temple=filter_temple,
                    category=cat,
                    full_parent_text=parent_text,
                    source_path=rel_path
                )
                prompt_str = self.scope_controller.build_scoped_prompt(scope, context_str, cleaned_question)
                answer_text = self._call_ollama_generate(prompt_str)
                sanitized_ans = self.grounding_validator.sanitize_answer(answer_text.strip(), scope, [])
                return {
                    "question": cleaned_question,
                    "answer": sanitized_ans,
                    "retrieved_chunks": 1,
                    "sources": [rel_path],
                    "confidence_score": 10.0,
                    "is_complete_list": True,
                    "plan": plan,
                    "scope": scope
                }

        # -------------------------------------------------------------
        # Route 3: Multi-Category Query Path
        # -------------------------------------------------------------
        if plan.retrieval_mode == "multi_category" and plan.temples and plan.categories:
            filter_temple = plan.temples[0]
            category_docs: Dict[str, List[Document]] = {}
            sources_set: Set[str] = set()

            for cat in plan.categories:
                parent_text = self.knowledge_registry.get_complete_parent(filter_temple, cat)
                src_files = self.knowledge_registry.get_source_files(filter_temple, cat)
                src_path = src_files[0] if src_files else f"{filter_temple.lower()}/{cat}.json"

                if parent_text:
                    category_docs[cat] = [Document(page_content=parent_text, metadata={"path": src_path, "category": cat, "temple": filter_temple})]
                    sources_set.add(src_path)
                else:
                    cat_docs = self.retrieve_with_metadata(cleaned_question, temple=filter_temple, category=cat, top_k=2)
                    if cat_docs:
                        category_docs[cat] = cat_docs
                        for d in cat_docs:
                            s = d.metadata.get("path") or d.metadata.get("source_file")
                            if s:
                                sources_set.add(s)

            if category_docs:
                context_str = self.context_builder.build_multi_category_context(filter_temple, category_docs)
                prompt_str = self.scope_controller.build_scoped_prompt(scope, context_str, cleaned_question)
                answer_text = self._call_ollama_generate(prompt_str)
                sanitized_ans = self.grounding_validator.sanitize_answer(answer_text.strip(), scope, [])
                return {
                    "question": cleaned_question,
                    "answer": sanitized_ans,
                    "retrieved_chunks": sum(len(docs) for docs in category_docs.values()),
                    "sources": sorted(list(sources_set)),
                    "confidence_score": 10.0,
                    "is_multi_category": True,
                    "plan": plan,
                    "scope": scope
                }

        # -------------------------------------------------------------
        # Route 4: Broad Temple Information Path
        # -------------------------------------------------------------
        if plan.retrieval_mode == "broad_temple" and plan.temples:
            filter_temple = plan.temples[0]
            category_texts: Dict[str, str] = {}
            sources_set: Set[str] = set()

            for cat in ["temple", "timings", "sevas", "festivals", "facilities", "travel", "accommodation", "restaurants", "prasadam", "rules", "contacts"]:
                text = self.knowledge_registry.get_complete_parent(filter_temple, cat)
                if text:
                    category_texts[cat] = text
                    src_files = self.knowledge_registry.get_source_files(filter_temple, cat)
                    if src_files:
                        sources_set.add(src_files[0])

            if category_texts:
                context_str = self.context_builder.build_broad_temple_context(filter_temple, category_texts, max_categories=6)
                prompt_str = self.scope_controller.build_scoped_prompt(scope, context_str, cleaned_question)
                answer_text = self._call_ollama_generate(prompt_str)
                return {
                    "question": cleaned_question,
                    "answer": answer_text.strip(),
                    "retrieved_chunks": len(category_texts),
                    "sources": sorted(list(sources_set)),
                    "confidence_score": 10.0,
                    "is_broad_temple": True,
                    "plan": plan,
                    "scope": scope
                }

        # -------------------------------------------------------------
        # Route 5: Normal Targeted Query Path (Hybrid Top-3 Baseline + AnswerScope Enforcement)
        # -------------------------------------------------------------
        filter_temple = plan.temples[0] if plan.temples else None

        dense_candidates = self.qdrant_manager.search_candidates(
            query=cleaned_question,
            embeddings=self.embeddings,
            top_k=settings.CANDIDATE_POOL_TOP_K,
            filter_temple=filter_temple
        )
        bm25_candidates = self.bm25_manager.search_candidates(
            query=cleaned_question,
            top_k=settings.CANDIDATE_POOL_TOP_K
        )
        if filter_temple:
            bm25_candidates = [
                d for d in bm25_candidates
                if str(d.metadata.get("temple", "")).lower() == filter_temple.lower()
            ]
        fused_candidates = reciprocal_rank_fusion(
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
            top_k=settings.CANDIDATE_POOL_TOP_K
        )

        if not fused_candidates:
            return {
                "question": cleaned_question,
                "answer": DEFAULT_REFUSAL_MESSAGE,
                "retrieved_chunks": 0,
                "sources": [],
                "confidence_score": 0.0,
                "plan": plan,
                "scope": scope
            }

        reranked_docs = self.reranker.rerank(
            query=cleaned_question,
            candidates=fused_candidates,
            top_k=settings.FINAL_RERANK_TOP_K
        )

        # Relevance Guardrail Check
        is_relevant, max_confidence = self.guardrail.evaluate_retrieval(reranked_docs)
        if not is_relevant:
            return {
                "question": cleaned_question,
                "answer": DEFAULT_REFUSAL_MESSAGE,
                "retrieved_chunks": len(reranked_docs),
                "sources": [],
                "confidence_score": 0.0,
                "plan": plan,
                "scope": scope
            }

        # Resolve Parent Context
        resolved_docs = []
        seen_parent_ids = set()

        for doc in reranked_docs:
            parent_id = doc.metadata.get("parent_id")
            parent_text = self.parent_store.get(parent_id) if parent_id else None

            if parent_text and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                meta = doc.metadata.copy()
                meta["is_parent_context"] = True
                resolved_docs.append(Document(page_content=parent_text, metadata=meta))
            elif not parent_text:
                resolved_docs.append(doc)

        # Context Filtering: Keep only scope-relevant chunks for targeted queries
        scoped_docs = self.scope_controller.filter_context(resolved_docs[:settings.FINAL_RERANK_TOP_K], scope)

        sources_set = set()
        for doc in scoped_docs:
            source_path = doc.metadata.get("path") or doc.metadata.get("source_file")
            if source_path:
                sources_set.add(source_path)

        context_str = self.context_builder.build_targeted_context(scoped_docs)
        prompt_str = self.scope_controller.build_scoped_prompt(scope, context_str, cleaned_question)

        # Generate answer via local Ollama instance
        raw_answer = self._call_ollama_generate(prompt_str)
        
        # Validate grounding & sanitize answer
        sanitized_ans = self.grounding_validator.sanitize_answer(raw_answer.strip(), scope, scoped_docs)
        grounding_report = self.grounding_validator.validate(sanitized_ans, scope, scoped_docs)

        return {
            "question": cleaned_question,
            "answer": sanitized_ans,
            "retrieved_chunks": len(scoped_docs),
            "sources": sorted(list(sources_set)),
            "confidence_score": max(0.0, float(max_confidence)),
            "plan": plan,
            "scope": scope,
            "grounding_validation": grounding_report
        }

    def _call_ollama_generate(self, prompt: str) -> str:
        """Call local Ollama REST API endpoint to generate response using Qwen2.5."""
        res = self.llm.generate(prompt)
        if res.get("error"):
            err = res["error"]
            if "Failed to connect to local Ollama" in err or "ConnectionRefusedError" in err or "connect" in err.lower():
                return "⚠️ **Ollama Connection Error**: Local Ollama is not running. Please start Ollama locally (`ollama run qwen2.5:1.5b`)."
            return DEFAULT_REFUSAL_MESSAGE
        return res.get("response", "")
