from __future__ import annotations

import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field

from cv_rag.answer.citations import enforce_cross_doc_support, retrieve_for_answer, validate_answer_citations
from cv_rag.answer.mlx_runner import mlx_generate, mlx_generate_stream
from cv_rag.answer.models import AnswerEvent, AnswerRunRequest, AnswerRunResult
from cv_rag.answer.prompts import build_prompt as build_mode_answer_prompt
from cv_rag.answer.prompts import build_repair_prompt as build_mode_repair_prompt
from cv_rag.answer.routing import AnswerMode, RouteDecision, make_route_decision, mode_from_value
from cv_rag.answer.routing import route as route_answer_mode
from cv_rag.retrieval.hybrid import HybridRetriever
from cv_rag.retrieval.models import RetrievedChunk
from cv_rag.retrieval.relevance import extract_entity_like_tokens
from cv_rag.shared.errors import CitationValidationError, GenerationError
from cv_rag.shared.settings import Settings


@dataclass(slots=True)
class _PreparedAnswerContext:
    decision: RouteDecision
    chunks: list[RetrievedChunk]
    maybe_relevant: list[RetrievedChunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class AnswerService:
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        settings: Settings,
        generate_fn: Callable[..., str] = mlx_generate,
        stream_generate_fn: Callable[..., Generator[str, None, None]] = mlx_generate_stream,
    ) -> None:
        self.retriever = retriever
        self.settings = settings
        self._generate = generate_fn
        self._stream_generate = stream_generate_fn

    def _prepare(self, request: AnswerRunRequest) -> _PreparedAnswerContext:
        mode_value = request.mode.strip().casefold()
        if mode_value not in {"auto", "single", "compare", "survey", "implement", "evidence"}:
            raise ValueError("Invalid mode. Use: auto|single|compare|survey|implement|evidence")

        router_strategy = request.router_strategy.strip().casefold()
        if router_strategy not in {"rules", "llm", "hybrid"}:
            raise ValueError("Invalid router strategy. Use: rules|llm|hybrid")

        if request.router_top_k <= 0:
            raise ValueError("router_top_k must be > 0")

        entity_tokens = extract_entity_like_tokens(request.question)
        prelim_chunks, maybe_relevant = retrieve_for_answer(
            retriever=self.retriever,
            question=request.question,
            k=request.router_top_k,
            max_per_doc=2,
            section_boost=request.section_boost,
            entity_tokens=entity_tokens,
        )

        if self.retriever._is_irrelevant_result(
            request.question,
            prelim_chunks[: max(3, len(prelim_chunks))],
            self.settings.relevance_vector_threshold,
        ):
            raise LookupError("Not found in indexed corpus. Try: cv-rag ingest-ids 2104.00680 1911.11763")

        if not prelim_chunks:
            raise LookupError("Not found in indexed corpus. Try: cv-rag ingest-ids 2104.00680 1911.11763")

        router_model_id = request.router_model or request.model
        if mode_value == "auto":
            decision = route_answer_mode(
                question=request.question,
                prelim_chunks=prelim_chunks,
                model_id=router_model_id,
                strategy=router_strategy,
            )
        else:
            forced_mode = mode_from_value(mode_value)
            decision = make_route_decision(
                forced_mode,
                notes=f"Manual mode override: {forced_mode.value}.",
                confidence=1.0,
            )

        final_k = request.k if request.k is not None else decision.k
        final_max_per_doc = request.max_per_doc if request.max_per_doc is not None else decision.max_per_doc
        effective_section_boost = max(request.section_boost, decision.section_boost_hint)
        chunks, _ = retrieve_for_answer(
            retriever=self.retriever,
            question=request.question,
            k=final_k,
            max_per_doc=final_max_per_doc,
            section_boost=effective_section_boost,
            entity_tokens=entity_tokens,
        )

        if not chunks:
            raise LookupError("No sources retrieved for this question")

        warnings: list[str] = []
        if decision.mode is AnswerMode.COMPARE:
            cross_doc_decision = enforce_cross_doc_support(request.question, chunks)
            chunks = cross_doc_decision.filtered_chunks
            warnings.extend(cross_doc_decision.warnings)
            if cross_doc_decision.should_refuse:
                raise LookupError(
                    "Refusing to answer comparison: need at least 2 sources from each of the top 2 papers."
                )

        return _PreparedAnswerContext(
            decision=decision,
            chunks=chunks,
            maybe_relevant=maybe_relevant,
            warnings=warnings,
        )

    def run(self, request: AnswerRunRequest) -> AnswerRunResult:
        context = self._prepare(request)
        warnings = list(context.warnings)

        prompt = build_mode_answer_prompt(
            context.decision.mode,
            request.question,
            context.chunks,
            route_preface=context.decision.preface,
        )

        answer_text = self._generate(
            model=request.model,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
        )

        valid, reason = validate_answer_citations(answer_text, len(context.chunks))
        if not valid:
            warnings.append("Draft failed citation check; attempting repair")
            repair_prompt = build_mode_repair_prompt(
                context.decision.mode,
                request.question,
                context.chunks,
                answer_text,
                route_preface=context.decision.preface,
            )
            repaired = self._generate(
                model=request.model,
                prompt=repair_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                seed=request.seed,
            )
            valid, reason = validate_answer_citations(repaired, len(context.chunks))
            if valid:
                answer_text = repaired

        if not valid and not request.no_refuse:
            raise CitationValidationError(reason, answer_text)

        return AnswerRunResult(
            answer=answer_text,
            sources=context.chunks,
            route=context.decision,
            citation_valid=valid,
            citation_reason=reason,
            warnings=warnings,
            maybe_relevant=context.maybe_relevant,
        )

    def stream(self, request: AnswerRunRequest) -> Generator[AnswerEvent, None, None]:
        t0 = time.perf_counter()

        try:
            context = self._prepare(request)
        except LookupError as exc:
            yield AnswerEvent(event="error", data={"message": str(exc)})
            return
        except ValueError as exc:
            yield AnswerEvent(event="error", data={"message": str(exc)})
            return

        yield AnswerEvent(
            event="route",
            data={
                "mode": context.decision.mode.value,
                "targets": context.decision.targets,
                "k": context.decision.k,
                "max_per_doc": context.decision.max_per_doc,
                "confidence": context.decision.confidence,
                "notes": context.decision.notes,
                "preface": context.decision.preface,
            },
        )

        for warning in context.warnings:
            yield AnswerEvent(event="warning", data={"message": warning})

        yield AnswerEvent(event="sources", data=context.chunks)

        prompt = build_mode_answer_prompt(
            context.decision.mode,
            request.question,
            context.chunks,
            route_preface=context.decision.preface,
        )

        collected_tokens: list[str] = []
        try:
            for token_chunk in self._stream_generate(
                model=request.model,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                seed=request.seed,
            ):
                collected_tokens.append(token_chunk)
                yield AnswerEvent(event="token", data=token_chunk)
        except GenerationError:
            full_text = self._generate(
                model=request.model,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                seed=request.seed,
            )
            collected_tokens = [full_text]
            yield AnswerEvent(event="token", data=full_text)

        answer_text = "".join(collected_tokens).strip()
        if not answer_text:
            yield AnswerEvent(event="error", data={"message": "LLM returned empty output."})
            return

        valid, reason = validate_answer_citations(answer_text, len(context.chunks))
        if not valid:
            yield AnswerEvent(event="warning", data={"message": "Draft failed citation check; attempting repair"})
            repair_prompt = build_mode_repair_prompt(
                context.decision.mode,
                request.question,
                context.chunks,
                answer_text,
                route_preface=context.decision.preface,
            )
            repaired = self._generate(
                model=request.model,
                prompt=repair_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                seed=request.seed,
            )
            valid, reason = validate_answer_citations(repaired, len(context.chunks))
            if valid:
                answer_text = repaired
                yield AnswerEvent(event="repair", data=repaired)

        if not valid and not request.no_refuse:
            yield AnswerEvent(
                event="error",
                data={"message": f"Citation validation failed: {reason}", "draft": answer_text},
            )
            return

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result = AnswerRunResult(
            answer=answer_text,
            sources=context.chunks,
            route=context.decision,
            citation_valid=valid,
            citation_reason=reason,
            warnings=context.warnings,
            maybe_relevant=context.maybe_relevant,
        )
        yield AnswerEvent(
            event="done",
            data={
                "answer": result.answer,
                "sources": result.sources,
                "route": result.route,
                "citation_valid": result.citation_valid,
                "citation_reason": result.citation_reason,
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )
