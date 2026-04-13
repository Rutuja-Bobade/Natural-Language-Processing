"""
Google ADK: deterministic SequentialAgent pipeline (no LLM calls).

Steps: preprocess → lexical (TF-IDF) → semantic (SBERT) → combine & sentence pairs.
Session state keys: raw_a, raw_b, top_k (inputs); pre_a, pre_b, ... (intermediate); report (PlagiarismReport).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types

from bert_module import document_embedding_similarity
from preprocessing import preprocess_text
from similarity import PlagiarismReport, hybrid_score, top_sentence_pairs
from tfidf_module import tfidf_cosine_similarity
from utils import split_sentences


def _evt(
    ctx: InvocationContext,
    author: str,
    text: str,
    *,
    state_delta: dict[str, Any] | None = None,
) -> Event:
    actions = EventActions(state_delta=state_delta or {})
    return Event(
        invocation_id=ctx.invocation_id,
        author=author,
        content=types.Content(
            role="model",
            parts=[types.Part(text=text)],
        ),
        actions=actions,
    )


class PreprocessAgent(BaseAgent):
    name: str = "preprocess_agent"
    description: str = "Lowercase, tokenize, remove stopwords, lemmatize."

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        raw_a = ctx.session.state["raw_a"]
        raw_b = ctx.session.state["raw_b"]

        ctx.session.state["pre_a"] = preprocess_text(raw_a)
        ctx.session.state["pre_b"] = preprocess_text(raw_b)

        ra = split_sentences(raw_a)
        rb = split_sentences(raw_b)

        ctx.session.state["raw_sents_a"] = ra
        ctx.session.state["raw_sents_b"] = rb
        ctx.session.state["pre_sents_a"] = [preprocess_text(s) for s in ra]
        ctx.session.state["pre_sents_b"] = [preprocess_text(s) for s in rb]

        yield _evt(
            ctx,
            self.name,
            "Preprocessing finished.",
            state_delta={
                "pre_a": ctx.session.state["pre_a"],
                "pre_b": ctx.session.state["pre_b"],
                "raw_sents_a": ctx.session.state["raw_sents_a"],
                "raw_sents_b": ctx.session.state["raw_sents_b"],
                "pre_sents_a": ctx.session.state["pre_sents_a"],
                "pre_sents_b": ctx.session.state["pre_sents_b"],
            },
        )


class LexicalAgent(BaseAgent):
    name: str = "lexical_agent"
    description: str = "Document-level TF-IDF cosine similarity."

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        pa = ctx.session.state["pre_a"]
        pb = ctx.session.state["pre_b"]

        ctx.session.state["tfidf_doc"] = tfidf_cosine_similarity(pa, pb)
        td = float(ctx.session.state["tfidf_doc"])

        yield _evt(
            ctx,
            self.name,
            f"TF-IDF document similarity: {td:.4f}",
            state_delta={"tfidf_doc": td},
        )


class SemanticAgent(BaseAgent):
    name: str = "semantic_agent"
    description: str = "Document-level Sentence-BERT cosine similarity."

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        pa = ctx.session.state["pre_a"]
        pb = ctx.session.state["pre_b"]

        ctx.session.state["bert_doc"] = document_embedding_similarity(pa, pb)
        bd = float(ctx.session.state["bert_doc"])

        yield _evt(
            ctx,
            self.name,
            f"SBERT document similarity: {bd:.4f}",
            state_delta={"bert_doc": bd},
        )


class CombineAgent(BaseAgent):
    name: str = "combine_agent"
    description: str = "Hybrid score, top sentence pairs, plagiarism percent."

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        from similarity import PlagiarismReport, section_weighted_hybrid

        top_k = int(ctx.session.state.get("top_k", 10))
        t = float(ctx.session.state["tfidf_doc"])
        s = float(ctx.session.state["bert_doc"])

        h_doc = hybrid_score(t, s)

        raw_a = ctx.session.state["raw_a"]
        raw_b = ctx.session.state["raw_b"]

        sec = section_weighted_hybrid(
            raw_a,
            raw_b,
            ctx.session.state["pre_a"],
            ctx.session.state["pre_b"],
        )

        display_h = sec if sec is not None else h_doc

        pairs = top_sentence_pairs(
            ctx.session.state["raw_sents_a"],
            ctx.session.state["raw_sents_b"],
            ctx.session.state["pre_sents_a"],
            ctx.session.state["pre_sents_b"],
            top_k=top_k,
        )

        report = PlagiarismReport(
            tfidf_doc=t,
            bert_doc=s,
            hybrid_doc=h_doc,
            plagiarism_percent=round(float(display_h) * 100.0, 2),
            top_pairs=pairs,
            section_weighted_hybrid=sec,
            meta={"used_section_weighting": sec is not None},
        )

        ctx.session.state["report"] = report

        yield _evt(
            ctx,
            self.name,
            f"Hybrid doc: {h_doc:.4f} | Plagiarism estimate: {report.plagiarism_percent}%",
            state_delta={"report": asdict(report)},
        )


def build_plagiarism_workflow() -> SequentialAgent:
    """Root agent for use with google.adk.runners.Runner or `adk run`."""
    return SequentialAgent(
        name="plagiarism_sequential",
        description="Hybrid TF-IDF + SBERT plagiarism pipeline.",
        sub_agents=[
            PreprocessAgent(),
            LexicalAgent(),
            SemanticAgent(),
            CombineAgent(),
        ],
    )


async def run_pipeline_with_runner(
    raw_a: str,
    raw_b: str,
    *,
    top_k: int = 10,
    app_name: str = "semantic_plag",
    user_id: str = "local_user",
    session_id: str = "session_1",
) -> dict[str, Any]:
    """
    Run the SequentialAgent via ADK Runner (in-memory session).
    Returns session.state after the run.
    """
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import (
        InMemorySessionService,
    )

    session_service = InMemorySessionService()

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={
            "raw_a": raw_a,
            "raw_b": raw_b,
            "top_k": top_k,
        },
    )

    runner = Runner(
        app_name=app_name,
        agent=build_plagiarism_workflow(),
        session_service=session_service,
    )

    msg = types.Content(
        role="user", parts=[types.Part(text="Run plagiarism analysis.")]
    )

    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=msg,
    ):
        pass

    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    st = session.state

    if hasattr(st, "to_dict"):
        return st.to_dict()

    return dict(st)