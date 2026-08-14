"""
LLM Roundtable HTTP API for local n8n (no Streamlit).

Default: 4 author models + 1 merger/judge (Opus). Human-framed critique/refine prompts.
Also supports LLM-as-a-judge: after refine, a judge picks the best revised version per author (no merge).

  export OPENROUTER_API_KEY=sk-or-...
  python n8n_server.py

  GET  /health
  POST /v1/runs              body: { "prompt": "...", "models"?: [...], "merger_model"?: "...", "judge_model"?: "..." }
  POST /v1/runs/{id}/primary
  POST /v1/runs/{id}/critiques
  POST /v1/runs/{id}/refine
  POST /v1/runs/{id}/merge
  POST /v1/runs/{id}/judge
  GET  /v1/runs/{id}
  GET  /v1/runs/{id}/summary
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
TFIDF_MAX_FEATURES = 4096
OUTPUT_DIR = Path("outputs/n8n")

DEFAULT_MODELS = [
    "openai/gpt-5.6-sol",
    "moonshotai/kimi-k3",
    "qwen/qwen3.7-max",
    "google/gemini-3.1-pro-preview",
]
DEFAULT_MERGER = "anthropic/claude-opus-4.8"
DEFAULT_JUDGE = "anthropic/claude-opus-4.8"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_WORKERS = 8
DEFAULT_MAX_TOKENS = 8192

# Per-model OpenRouter / provider extras for "high" reasoning effort.
DEFAULT_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "anthropic/claude-opus-4.8": {
        "thinking": {"type": "enabled", "budget_tokens": 32000},
        "output_config": {"effort": "high"},
    },
    "openai/gpt-5.6-sol": {
        "reasoning": {"effort": "high"},
    },
    "moonshotai/kimi-k3": {
        "reasoning_effort": "high",
    },
    "qwen/qwen3.7-max": {
        "enable_thinking": True,
        "thinking_budget": 32768,
    },
    "google/gemini-3.1-pro-preview": {
        "thinking_level": "high",
    },
}

JUDGE_TEMPLATE = """You are an expert bioinformaticist in the domain of biomedical ontologies. Several revised answers to the same prompt were produced by one system after it incorporated several independent critiques by experts. Your job is to decide which single revised answer is the best response to the original prompt.

The author was originally asked:
ORIGINAL PROMPT: {prompt}
The author then produced the following revised versions, each incorporating a different critique:
REVISED VERSIONS: {revised_versions}

Select exactly one version. Do not merge, combine, rewrite, correct or extend the versions, and do not combine or create a new answer from parts of several versions.

Evaluate each version strictly against the requirements stated in the original prompt. Use the following criteria:
Is the formula well-defined and consistent (e.g., factors combine in a way that is quantifiable and produces a score)?
Do the factors target distinct aspects of utility based on the target ontology’s content, without overlapping or duplicating one another?
For each factor, does the version clearly address all four required elements: what it measures, why it matters, how it influences the score, and what assumptions it relies on?
Are excluded factors and assumptions specifically stated?
Does the version follow the required output format?

Do not reward a version for its length, for listing more factors, for hedging, or for appearing more confident. Do not consider which critique prompted which version.

Return your answer in the following format:
Selected version (label given in REVISED VERSIONS)
A concise rationale (2-3 sentences) on why this version was chosen
"""

CRITIQUE_TEMPLATE = """You are an experienced critic in the domain of biomedical ontologies evaluating someone else's answer.

Critique the answer below: What are its strengths, weaknesses, and areas for improvement? For each weakness, explain how it can be improved.

I posed the following problem to another human expert:
\tORIGINAL PROMPT: {prompt}
They returned this result:
ANSWER: {response_to_evaluate}

Be concise and specific with every point, avoiding vague praise or criticism. Address each flaw by stating what is wrong and how to improve on it.

Return your answer in the following format:
1. Strengths
2. Weaknesses and gaps
3. Concrete improvement recommendations (how, not just what)
"""

REFINEMENT_TEMPLATE = """You are the original author revising your own work in response to expert feedback.

Produce an improved version of your original answer that incorporates the critique below.

I previously asked you:
ORIGINAL PROMPT: {prompt}
You responded with the following:
YOUR ANSWER: {original_response}
I passed your response to a human for critique. They returned:
CRITIQUE: {critique}

Address each point raised in the critique; if you disagree with a point, say so and justify why. Keep what was already strong; do not regress on correct material.

Maintain the original output requirements (plain-text notation, formula + factors + assumptions).
"""

MERGE_VERSION_BLOCK_TEMPLATE = """### Version after critique from {critic_label}
{text}
"""

MERGE_TEMPLATE = """You are a response aggregator. Several revised answers to the same prompt were produced by one author after it incorporated several independent critiques. Your job is to merge them into one definitive answer.

Merge the revised versions below into ONE final answer that is at least as good as any individual version in every aspect. Each version was the same author's attempt to improve the same underlying answer in response to a different critique.

The author was originally asked:
ORIGINAL PROMPT: {prompt}
The author then produced the following revised versions, each one incorporating a different critique:
REVISED VERSIONS: {revised_versions}

Unify notation, terminology, and structure so the result reads as one consistently authored response rather than stitched-together drafts. Where revisions conflict, keep the better-justified claim and drop the weaker one. Preserve every correct improvement that any revision introduced; do not regress to a weaker formulation that an earlier revision had already fixed. Judge content on its merits, not on which critique prompted it. You are not bound to any single version as a base. Do not mention the merge, the existence of multiple drafts, or the critiques. Write as if this were a single original answer to the prompt.

Output a single coherent answer. Do not output multiple versions, a difference report, or a side-by-side comparison. Maintain the original output requirements (plain-text notation, formula + factors + assumptions).
"""


def get_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_BASE,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/narenkhatwani/llm-roundtable",
            "X-Title": "LLM Roundtable n8n",
        },
    )


def chat_complete(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra:
        # Provider-specific / OpenRouter reasoning knobs (thinking, reasoning, etc.)
        kwargs["extra_body"] = extra
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def model_extra(run: "RunState", model: str) -> dict[str, Any]:
    return dict(run.model_configs.get(model) or {})

def tfidf_cosine(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        stop_words="english",
        smooth_idf=True,
        sublinear_tf=False,
        norm="l2",
    )
    try:
        m = vec.fit_transform([a, b])
        return float(cosine_similarity(m[0:1], m[1:2])[0, 0])
    except ValueError:
        return 0.0


def estimate_api_calls(n: int) -> int:
    return n + n * (n - 1) + n * (n - 1) + n


def estimate_judge_api_calls(n: int) -> int:
    """Primary + critiques + refine + one judge call per author (replaces merge)."""
    return n + n * (n - 1) + n * (n - 1) + n


@dataclass
class RunState:
    run_id: str
    prompt: str
    models: list[str]
    merger_model: str
    judge_model: str = DEFAULT_JUDGE
    model_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    temperature: float = DEFAULT_TEMPERATURE
    max_workers: int = DEFAULT_MAX_WORKERS
    phase: str = "created"
    primary: list[str | None] = field(default_factory=list)
    critiques: list[list[str | None]] = field(default_factory=list)
    refined: list[list[str | None]] = field(default_factory=list)
    merged: list[str | None] = field(default_factory=list)
    judgments: list[str | None] = field(default_factory=list)
    cosines: list[float] = field(default_factory=list)
    error: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


RUNS: dict[str, RunState] = {}


class CreateRunBody(BaseModel):
    prompt: str = Field(..., min_length=1)
    models: Optional[List[str]] = None
    merger_model: Optional[str] = None
    judge_model: Optional[str] = None
    model_configs: Optional[dict[str, dict[str, Any]]] = None
    temperature: Optional[float] = None
    max_workers: Optional[int] = None


def require_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=500,
            detail="Set OPENROUTER_API_KEY in the environment before starting n8n_server.py",
        )
    return key


def get_run(run_id: str) -> RunState:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return run


def run_primary(client: OpenAI, run: RunState) -> None:
    n = len(run.models)
    out: list[str | None] = [None] * n

    def one(i: int) -> tuple[int, str]:
        model = run.models[i]
        text = chat_complete(
            client,
            model,
            [{"role": "user", "content": run.prompt}],
            run.temperature,
            extra=model_extra(run, model),
        )
        return i, text

    with concurrent.futures.ThreadPoolExecutor(max_workers=run.max_workers) as ex:
        futs = [ex.submit(one, i) for i in range(n)]
        for fut in concurrent.futures.as_completed(futs):
            i, text = fut.result()
            out[i] = text
    run.primary = out
    run.phase = "primary"


def run_critiques(client: OpenAI, run: RunState) -> None:
    n = len(run.models)
    grid: list[list[str | None]] = [[None] * n for _ in range(n)]
    tasks = [(i, j) for i in range(n) for j in range(n) if j != i]

    def work(i: int, j: int) -> tuple[int, int, str]:
        model = run.models[j]
        msg = CRITIQUE_TEMPLATE.format(
            prompt=run.prompt,
            response_to_evaluate=run.primary[i] or "",
        )
        text = chat_complete(
            client,
            model,
            [{"role": "user", "content": msg}],
            run.temperature,
            extra=model_extra(run, model),
        )
        return i, j, text

    with concurrent.futures.ThreadPoolExecutor(max_workers=run.max_workers) as ex:
        futs = [ex.submit(work, i, j) for i, j in tasks]
        for fut in concurrent.futures.as_completed(futs):
            i, j, text = fut.result()
            grid[i][j] = text
    run.critiques = grid
    run.phase = "critiques"


def run_refine(client: OpenAI, run: RunState) -> None:
    n = len(run.models)
    refined: list[list[str | None]] = [[None] * n for _ in range(n)]
    tasks = [(i, j) for i in range(n) for j in range(n) if j != i]

    def work(i: int, j: int) -> tuple[int, int, str]:
        model = run.models[i]
        msg = REFINEMENT_TEMPLATE.format(
            prompt=run.prompt,
            original_response=run.primary[i] or "",
            critique=run.critiques[i][j] or "",
        )
        text = chat_complete(
            client,
            model,
            [{"role": "user", "content": msg}],
            run.temperature,
            extra=model_extra(run, model),
        )
        return i, j, text

    with concurrent.futures.ThreadPoolExecutor(max_workers=run.max_workers) as ex:
        futs = [ex.submit(work, i, j) for i, j in tasks]
        for fut in concurrent.futures.as_completed(futs):
            i, j, text = fut.result()
            refined[i][j] = text
    run.refined = refined
    run.phase = "refine"


def run_merge(client: OpenAI, run: RunState) -> None:
    n = len(run.models)
    merged: list[str | None] = [None] * n
    for i in range(n):
        blocks: list[str] = []
        for j in range(n):
            if j == i:
                continue
            t = run.refined[i][j]
            if t:
                blocks.append(
                    MERGE_VERSION_BLOCK_TEMPLATE.format(
                        critic_label=f"LLM-{j + 1} ({run.models[j]})",
                        text=t,
                    )
                )
        if not blocks:
            merged[i] = None
            continue
        msg = MERGE_TEMPLATE.format(
            prompt=run.prompt,
            revised_versions="\n".join(blocks),
        )
        merged[i] = chat_complete(
            client,
            run.merger_model,
            [{"role": "user", "content": msg}],
            run.temperature,
            extra=model_extra(run, run.merger_model),
        )
    run.merged = merged
    run.cosines = [
        tfidf_cosine(run.primary[i] or "", run.merged[i] or "") for i in range(n)
    ]
    run.phase = "merged"
    _save_run(run)


def run_judge(client: OpenAI, run: RunState) -> None:
    """Per author: judge selects the best of that author's refined versions."""
    n = len(run.models)
    judgments: list[str | None] = [None] * n
    for i in range(n):
        blocks: list[str] = []
        for j in range(n):
            if j == i:
                continue
            t = run.refined[i][j]
            if t:
                blocks.append(
                    MERGE_VERSION_BLOCK_TEMPLATE.format(
                        critic_label=f"LLM-{j + 1} ({run.models[j]})",
                        text=t,
                    )
                )
        if not blocks:
            judgments[i] = None
            continue
        msg = JUDGE_TEMPLATE.format(
            prompt=run.prompt,
            revised_versions="\n".join(blocks),
        )
        judgments[i] = chat_complete(
            client,
            run.judge_model,
            [{"role": "user", "content": msg}],
            run.temperature,
            extra=model_extra(run, run.judge_model),
        )
    run.judgments = judgments
    run.phase = "judged"
    _save_run(run, filename="judge_run.json")


def _save_run(run: RunState, filename: str = "roundtable_run.json") -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT_DIR / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run.run_id,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": run.phase,
        "models": run.models,
        "merger_model": run.merger_model,
        "judge_model": run.judge_model,
        "model_configs": run.model_configs,
        "temperature": run.temperature,
        "primary_prompt": run.prompt,
        "primary_responses": run.primary,
        "critiques": run.critiques,
        "refined_responses": run.refined,
        "merged_responses": run.merged,
        "judgments": run.judgments,
        "tfidf_cosine_primary_vs_merged": run.cosines,
    }
    path = run_dir / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def build_summary(run: RunState) -> dict[str, Any]:
    rows = []
    for i, model in enumerate(run.models):
        sim = run.cosines[i] if i < len(run.cosines) else None
        judgment = run.judgments[i] if i < len(run.judgments) else None
        rows.append(
            {
                "author_index": i + 1,
                "model": model,
                "tfidf_cosine_primary_vs_merged": sim,
                "primary_preview": ((run.primary[i] or "")[:400] if run.primary else ""),
                "merged_preview": ((run.merged[i] or "")[:400] if run.merged else ""),
                "judgment_preview": ((judgment or "")[:400] if judgment else ""),
                "primary": run.primary[i] if run.primary else None,
                "merged": run.merged[i] if run.merged else None,
                "judgment": judgment,
            }
        )

    if run.phase == "judged":
        lines = [
            f"# LLM-as-judge run `{run.run_id}`",
            f"- Phase: **{run.phase}**",
            f"- Authors: {len(run.models)} | Judge: `{run.judge_model}`",
            f"- Estimated API calls: {estimate_judge_api_calls(len(run.models))}",
            "",
            "## Judgments",
            "",
        ]
        for row in rows:
            lines.append(f"### LLM {row['author_index']} — `{row['model']}`")
            lines.append(row["judgment"] or "_empty_")
            lines.append("")
        api_calls = estimate_judge_api_calls(len(run.models))
    else:
        lines = [
            f"# Roundtable run `{run.run_id}`",
            f"- Phase: **{run.phase}**",
            f"- Authors: {len(run.models)} | Merger: `{run.merger_model}`",
            f"- Estimated API calls: {estimate_api_calls(len(run.models))}",
            "",
            "## TF-IDF cosine (primary vs merged)",
            "",
        ]
        for row in rows:
            sim = row["tfidf_cosine_primary_vs_merged"]
            sim_s = f"{sim:.4f}" if isinstance(sim, float) else "n/a"
            lines.append(f"- **LLM {row['author_index']}** `{row['model']}`: {sim_s}")

        lines.extend(["", "## Merged answers", ""])
        for row in rows:
            lines.append(f"### LLM {row['author_index']} — `{row['model']}`")
            lines.append(row["merged"] or "_empty_")
            lines.append("")
        api_calls = estimate_api_calls(len(run.models))

    return {
        "run_id": run.run_id,
        "phase": run.phase,
        "models": run.models,
        "merger_model": run.merger_model,
        "judge_model": run.judge_model,
        "api_calls": api_calls,
        "authors": rows,
        "markdown": "\n".join(lines),
        "output_dir": str((OUTPUT_DIR / run.run_id).resolve()),
    }


app = FastAPI(title="LLM Roundtable for n8n", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    key_set = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    return {
        "ok": True,
        "openrouter_key_set": key_set,
        "default_models": DEFAULT_MODELS,
        "default_merger": DEFAULT_MERGER,
        "default_judge": DEFAULT_JUDGE,
        "api_calls_for_defaults": estimate_api_calls(len(DEFAULT_MODELS)),
        "judge_api_calls_for_defaults": estimate_judge_api_calls(len(DEFAULT_MODELS)),
    }


@app.post("/v1/runs")
def create_run(body: CreateRunBody) -> dict[str, Any]:
    require_api_key()
    models = body.models or list(DEFAULT_MODELS)
    if len(models) != 4:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly 4 author models, got {len(models)}",
        )
    merger = (body.merger_model or DEFAULT_MERGER).strip()
    judge = (body.judge_model or DEFAULT_JUDGE).strip()
    configs = dict(DEFAULT_MODEL_CONFIGS)
    if body.model_configs:
        for key, val in body.model_configs.items():
            configs[key.strip()] = dict(val or {})
    run_id = uuid.uuid4().hex[:8]
    run = RunState(
        run_id=run_id,
        prompt=body.prompt.strip(),
        models=[m.strip() for m in models],
        merger_model=merger,
        judge_model=judge,
        model_configs=configs,
        temperature=body.temperature if body.temperature is not None else DEFAULT_TEMPERATURE,
        max_workers=body.max_workers if body.max_workers is not None else DEFAULT_MAX_WORKERS,
    )
    RUNS[run_id] = run
    return {
        "run_id": run_id,
        "phase": run.phase,
        "models": run.models,
        "merger_model": run.merger_model,
        "judge_model": run.judge_model,
        "model_configs": run.model_configs,
        "api_calls": estimate_api_calls(len(run.models)),
        "judge_api_calls": estimate_judge_api_calls(len(run.models)),
        "created_at": run.created_at,
    }


@app.post("/v1/runs/{run_id}/primary")
def phase_primary(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    client = get_client(require_api_key())
    try:
        run_primary(client, run)
    except Exception as e:
        run.error = str(e)
        run.phase = "error"
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        "run_id": run.run_id,
        "phase": run.phase,
        "primary_responses": run.primary,
    }


@app.post("/v1/runs/{run_id}/critiques")
def phase_critiques(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    if run.phase not in {"primary", "critiques"}:
        raise HTTPException(status_code=400, detail=f"Need primary first (phase={run.phase})")
    client = get_client(require_api_key())
    try:
        run_critiques(client, run)
    except Exception as e:
        run.error = str(e)
        run.phase = "error"
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        "run_id": run.run_id,
        "phase": run.phase,
        "critique_count": sum(
            1 for row in run.critiques for cell in row if cell is not None
        ),
    }


@app.post("/v1/runs/{run_id}/refine")
def phase_refine(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    if run.phase not in {"critiques", "refine"}:
        raise HTTPException(status_code=400, detail=f"Need critiques first (phase={run.phase})")
    client = get_client(require_api_key())
    try:
        run_refine(client, run)
    except Exception as e:
        run.error = str(e)
        run.phase = "error"
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        "run_id": run.run_id,
        "phase": run.phase,
        "refine_count": sum(1 for row in run.refined for cell in row if cell is not None),
    }


@app.post("/v1/runs/{run_id}/merge")
def phase_merge(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    if run.phase not in {"refine", "merged"}:
        raise HTTPException(status_code=400, detail=f"Need refine first (phase={run.phase})")
    client = get_client(require_api_key())
    try:
        run_merge(client, run)
    except Exception as e:
        run.error = str(e)
        run.phase = "error"
        raise HTTPException(status_code=500, detail=str(e)) from e
    return build_summary(run)


@app.post("/v1/runs/{run_id}/judge")
def phase_judge(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    if run.phase not in {"refine", "judged"}:
        raise HTTPException(status_code=400, detail=f"Need refine first (phase={run.phase})")
    client = get_client(require_api_key())
    try:
        run_judge(client, run)
    except Exception as e:
        run.error = str(e)
        run.phase = "error"
        raise HTTPException(status_code=500, detail=str(e)) from e
    return build_summary(run)


@app.get("/v1/runs/{run_id}")
def get_run_state(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    return {
        "run_id": run.run_id,
        "phase": run.phase,
        "models": run.models,
        "merger_model": run.merger_model,
        "judge_model": run.judge_model,
        "error": run.error,
        "has_primary": bool(run.primary),
        "has_critiques": bool(run.critiques),
        "has_refined": bool(run.refined),
        "has_merged": bool(run.merged),
        "has_judgments": bool(run.judgments),
        "cosines": run.cosines,
    }


@app.get("/v1/runs/{run_id}/summary")
def get_summary(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    if run.phase not in {"merged", "judged"}:
        raise HTTPException(status_code=400, detail=f"Run not finished (phase={run.phase})")
    return build_summary(run)


if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        print("WARNING: OPENROUTER_API_KEY is not set.")
    print(
        f"Roundtable API on http://127.0.0.1:8787  "
        f"({len(DEFAULT_MODELS)} authors + merger, "
        f"~{estimate_api_calls(len(DEFAULT_MODELS))} LLM calls/run; "
        f"judge mode ~{estimate_judge_api_calls(len(DEFAULT_MODELS))} calls/run)"
    )
    uvicorn.run(app, host="127.0.0.1", port=8787, log_level="info")
