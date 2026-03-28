"""
LLM Roundtable: multi-model critique → refine → merge (Streamlit + OpenRouter).
Extends the AMIA-style pipeline with a merge step per author LLM and comparison to primary.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Must stay in sync between cosine metric and on-screen TF-IDF breakdown
TFIDF_MAX_FEATURES = 4096
TFIDF_STOP_WORDS = "english"


def _tfidf_vectorizer_kwargs(**overrides: object) -> dict:
    base: dict = {
        "max_features": TFIDF_MAX_FEATURES,
        "stop_words": TFIDF_STOP_WORDS,
        "smooth_idf": True,
        "sublinear_tf": False,
    }
    base.update(overrides)
    return base

# Curated OpenRouter slugs (labels for UI). Verified against https://openrouter.ai/api/v1/models
CURATED_MODELS: list[tuple[str, str]] = [
    ("Anthropic — Claude Opus 4.6", "anthropic/claude-opus-4.6"),
    ("Anthropic — Claude Opus 4.5", "anthropic/claude-opus-4.5"),
    ("Anthropic — Claude Opus 4.1", "anthropic/claude-opus-4.1"),
    ("Anthropic — Claude Sonnet 4.6", "anthropic/claude-sonnet-4.6"),
    ("Anthropic — Claude Sonnet 4.5", "anthropic/claude-sonnet-4.5"),
    ("Anthropic — Claude Sonnet 4", "anthropic/claude-sonnet-4"),
    ("Anthropic — Claude 3.7 Sonnet", "anthropic/claude-3.7-sonnet"),
    ("Anthropic — Claude 3.5 Sonnet", "anthropic/claude-3.5-sonnet"),
    ("OpenAI — GPT-5.4 Pro", "openai/gpt-5.4-pro"),
    ("OpenAI — GPT-5.4", "openai/gpt-5.4"),
    ("OpenAI — GPT-5.4 Mini", "openai/gpt-5.4-mini"),
    ("OpenAI — GPT-5.4 Nano", "openai/gpt-5.4-nano"),
    ("OpenAI — GPT-5.3 Codex", "openai/gpt-5.3-codex"),
    ("OpenAI — GPT-5.2 Pro", "openai/gpt-5.2-pro"),
    ("OpenAI — GPT-5.2", "openai/gpt-5.2"),
    ("OpenAI — GPT-5.2 Chat", "openai/gpt-5.2-chat"),
    ("OpenAI — GPT-5.1", "openai/gpt-5.1"),
    ("OpenAI — GPT-5 Pro", "openai/gpt-5-pro"),
    ("OpenAI — GPT-5", "openai/gpt-5"),
    ("OpenAI — GPT-5 Mini", "openai/gpt-5-mini"),
    ("OpenAI — GPT-5 Nano", "openai/gpt-5-nano"),
    ("OpenAI — o3 Pro", "openai/o3-pro"),
    ("OpenAI — o3", "openai/o3"),
    ("OpenAI — o3 Mini", "openai/o3-mini"),
    ("OpenAI — o1", "openai/o1"),
    ("OpenAI — GPT-4o", "openai/gpt-4o"),
    ("OpenAI — GPT-4o Mini", "openai/gpt-4o-mini"),
    ("Google — Gemini 3.1 Pro (preview)", "google/gemini-3.1-pro-preview"),
    ("Google — Gemini 3.1 Flash Lite (preview)", "google/gemini-3.1-flash-lite-preview"),
    ("Google — Gemini 3 Flash (preview)", "google/gemini-3-flash-preview"),
    ("Google — Gemini 2.5 Pro", "google/gemini-2.5-pro"),
    ("Google — Gemini 2.5 Flash", "google/gemini-2.5-flash"),
    ("Google — Gemini 2.5 Flash Lite", "google/gemini-2.5-flash-lite"),
    ("Google — Gemini 2.0 Flash", "google/gemini-2.0-flash-001"),
    ("DeepSeek — V3.2", "deepseek/deepseek-v3.2"),
    ("DeepSeek — V3.2 (Speciale)", "deepseek/deepseek-v3.2-speciale"),
    ("DeepSeek — V3.1 Terminus", "deepseek/deepseek-v3.1-terminus"),
    ("DeepSeek — Chat V3.1", "deepseek/deepseek-chat-v3.1"),
    ("DeepSeek — R1", "deepseek/deepseek-r1"),
    ("DeepSeek — R1 (0528)", "deepseek/deepseek-r1-0528"),
    ("Mistral — Mistral Large 2411", "mistralai/mistral-large-2411"),
    ("Meta — Llama 3.3 70B Instruct", "meta-llama/llama-3.3-70b-instruct"),
    ("Meta — Llama 3.1 70B Instruct", "meta-llama/llama-3.1-70b-instruct"),
    ("xAI — Grok 4.1 Fast", "x-ai/grok-4.1-fast"),
    ("xAI — Grok 4 Fast", "x-ai/grok-4-fast"),
    ("xAI — Grok 3", "x-ai/grok-3"),
    ("xAI — Grok 3 Mini", "x-ai/grok-3-mini"),
]

DEFAULT_SLOT_SLUGS = [
    "openai/gpt-5.4",
    "anthropic/claude-opus-4.6",
    "google/gemini-2.5-pro",
    "deepseek/deepseek-r1",
]


def curated_dropdown_options() -> tuple[list[str], list[str | None]]:
    """Returns (labels, slugs) with first row = custom (slug None)."""
    labels = ["Custom — type OpenRouter slug below"] + [pair[0] for pair in CURATED_MODELS]
    slugs: list[str | None] = [None] + [pair[1] for pair in CURATED_MODELS]
    return labels, slugs


def default_curated_index_for_slug(wanted: str) -> int:
    for idx, (_, slug) in enumerate(CURATED_MODELS, start=1):
        if slug == wanted:
            return idx
    return 0


def resolve_model_choice(
    choice_index: int,
    labels: list[str],
    slugs: list[str | None],
    custom_text: str,
) -> str:
    if choice_index == 0:
        return custom_text.strip()
    if 0 <= choice_index < len(slugs) and slugs[choice_index]:
        return slugs[choice_index].strip()
    return custom_text.strip()


def get_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_BASE,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/narenkhatwani/llm-roundtable",
            "X-Title": "LLM Roundtable",
        },
    )


def chat_complete(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int = 8192,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    content = choice.message.content
    if not content:
        return ""
    return content.strip()


def critique_user_message(
    original_prompt: str,
    responder_label: str,
    response_text: str,
) -> str:
    return f"""You are an expert critic. Another system ({responder_label}) answered a research prompt.
Your job is to give a detailed, constructive critique: strengths, gaps, errors, missing factors, and concrete suggestions.

## Original prompt
{original_prompt}

## Response to critique (from {responder_label})
{response_text}

Respond with a structured critique (bullet points welcome). Be specific and actionable."""


DEFAULT_REFINEMENT_TEMPLATE = """I previously asked you the following:

{original_prompt}

You responded with:
---
{original_response}
---

I passed your response to another system ({critic_label}) for critique. Here is the critique:
---
{critique_text}
---

Improve your original answer based on this critique. Address the critique directly while keeping correct content from your first answer. Repeat the full improved answer (not only a diff)."""


REFINEMENT_TEMPLATE_PLACEHOLDERS = (
    "{original_prompt}",
    "{original_response}",
    "{critic_label}",
    "{critique_text}",
)


def refine_user_message(
    template: str,
    original_prompt: str,
    original_response: str,
    critic_label: str,
    critique_text: str,
) -> str:
    try:
        return template.format(
            original_prompt=original_prompt,
            original_response=original_response,
            critic_label=critic_label,
            critique_text=critique_text,
        )
    except KeyError as e:
        raise ValueError(
            f"Improvement template contains unknown placeholder {e}. "
            f"Use only: {', '.join(REFINEMENT_TEMPLATE_PLACEHOLDERS)}"
        ) from e


def merge_user_message(
    original_prompt: str,
    author_label: str,
    refined_blocks: list[tuple[str, str]],
) -> str:
    parts = [
        f"You synthesize multiple revised answers to the same question. The same author ({author_label}) "
        "produced these versions after incorporating different peer critiques. Merge them into ONE coherent "
        "final answer: unify notation, resolve contradictions by preferring better-supported claims, and keep "
        "the strongest structure. Do not mention the merge process or that there were multiple drafts.\n",
        f"## Question\n{original_prompt}\n",
        "## Revised versions\n",
    ]
    for idx, (critic_label, text) in enumerate(refined_blocks, start=1):
        parts.append(f"### Version after critique from {critic_label}\n{text}\n")
    parts.append("\n## Your unified answer\n")
    return "".join(parts)


def tfidf_cosine(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    vec = TfidfVectorizer(**_tfidf_vectorizer_kwargs(norm="l2"))
    try:
        m = vec.fit_transform([a, b])
        sim = cosine_similarity(m[0:1], m[1:2])[0, 0]
        return float(sim)
    except ValueError:
        return 0.0


def tfidf_pair_breakdown(
    text_a: str,
    text_b: str,
    top_n: int = 18,
) -> dict | None:
    """
    Fit TF-IDF on exactly [text_a, text_b] with the same settings as tfidf_cosine.
    Returns raw tf, sklearn idf, tf×idf (unnormalized), L2-normalized cosine, and top terms per doc.
    """
    if not text_a.strip() or not text_b.strip():
        return None
    kw = _tfidf_vectorizer_kwargs()
    try:
        vec_raw = TfidfVectorizer(**kw, norm=None)
        X = vec_raw.fit_transform([text_a, text_b]).toarray()
    except ValueError:
        return None

    idf = np.asarray(vec_raw.idf_, dtype=float)
    names = vec_raw.get_feature_names_out()
    n_samples = 2

    # Document frequency: number of docs (of 2) with at least one occurrence
    cv = CountVectorizer(vocabulary=vec_raw.vocabulary_)
    C = cv.fit_transform([text_a, text_b])
    df = np.asarray(C.astype(bool).sum(axis=0)).ravel().astype(int)

    idf_from_df = np.log((n_samples + 1) / (df + 1)) + 1.0
    idf_matches = bool(np.allclose(idf, idf_from_df))

    # Raw tf-idf weights (same matrix sklearn uses before row L2 norm)
    w = X.copy()
    # Recover integer term counts: tf = w / idf (sklearn: sublinear_tf=False)
    tf = np.zeros_like(w)
    safe = idf > 1e-15
    tf[:, safe] = w[:, safe] / idf[safe]

    n0 = float(np.linalg.norm(w[0]))
    n1 = float(np.linalg.norm(w[1]))
    cosine_manual = float(np.dot(w[0], w[1]) / (n0 * n1)) if n0 > 1e-15 and n1 > 1e-15 else 0.0

    vec_l2 = TfidfVectorizer(**kw, norm="l2")
    Xl2 = vec_l2.fit_transform([text_a, text_b])
    cosine_display = float(cosine_similarity(Xl2[0:1], Xl2[1:2])[0, 0])

    def top_rows(doc_idx: int) -> list[dict[str, float | int | str]]:
        row = w[doc_idx]
        order = np.argsort(-np.abs(row))
        out: list[dict[str, float | int | str]] = []
        for j in order[:top_n]:
            if row[j] == 0.0:
                continue
            tfc = tf[doc_idx, j]
            tfc_i = int(round(tfc)) if abs(tfc - round(tfc)) < 1e-5 else float(tfc)
            out.append(
                {
                    "term": str(names[j]),
                    "tf (count)": tfc_i,
                    "df (of 2 docs)": int(df[j]),
                    "idf": float(idf[j]),
                    "tf × idf": float(row[j]),
                }
            )
        return out

    # Worked IDF check rows for a few terms that appear in at least one document
    sample_terms: list[dict[str, float | int | str]] = []
    active = np.where((w[0] != 0) | (w[1] != 0))[0]
    for j in active[:8]:
        sample_terms.append(
            {
                "term": str(names[j]),
                "df": int(df[j]),
                "idf (sklearn)": float(idf[j]),
                "ln((1+n)/(1+df))+1": float(math.log((1 + n_samples) / (1 + df[j])) + 1),
            }
        )

    return {
        "vocab_size": int(len(names)),
        "cosine_l2": cosine_display,
        "cosine_manual_raw_vectors": cosine_manual,
        "cosine_match": bool(abs(cosine_display - cosine_manual) < 1e-5),
        "primary_top": top_rows(0),
        "merged_top": top_rows(1),
        "idf_matches_df_formula": idf_matches,
        "sample_idf_rows": sample_terms[:6],
    }


def render_tfidf_methodology_expander() -> None:
    """Static help: how TF-IDF cosine is defined in this app (matches scikit-learn)."""
    with st.expander("TF–IDF cosine: definitions & formulas (used in this app)", expanded=False):
        st.markdown(
            f"""
**What is being compared?**  
For each author we take two strings: **primary** response and **merged** response.  
We treat them as a **2-document corpus** (nothing else—no full paper corpus).

**Preprocessing (scikit-learn `TfidfVectorizer`)**  
- Tokenize into **word** features (default token pattern).  
- **Lowercasing** and English **stop words** removed (`stop_words="{TFIDF_STOP_WORDS}"`).  
- Vocabulary = up to **{TFIDF_MAX_FEATURES}** terms, kept by overall frequency in *these two* documents.

---

### Term frequency **TF**

For term **t** in document **d**:

- **TF** = **raw count** of **t** in **d** after the steps above.  
- We use `sublinear_tf=False` (no log-scaling of counts).

*Example:* if “ontology” appears 3 times in the primary text, then `tf("ontology", primary) = 3`.

---

### Document frequency **DF** and inverse DF **IDF**

Here **n = 2** (primary + merged).

- **df(t)** = number of documents (0, 1, or 2) where **t** appears **at least once**.

With sklearn’s default **`smooth_idf=True`** (what this app uses):

$$
\\mathrm{{idf}}(t) = \\ln\\!\\left(\\frac{{1 + n}}{{1 + \\mathrm{{df}}(t)}}\\right) + 1
$$

Natural logarithm (`ln`).

**Worked examples** (plug in *n = 2*):

| df(t) | Formula | Numeric idf(t) (approx.) |
|:-----:|---------|---------------------------|
| 1 (term in **one** doc only) | ln((1+2)/(1+1)) + 1 = ln(3/2) + 1 | **≈ 1.405** |
| 2 (term in **both** docs) | ln((1+2)/(1+2)) + 1 = ln(1) + 1 | **= 1.0** |

---

### Raw TF-IDF weight

For each cell (document **d**, term **t**):

$$
\\mathrm{{tfidf}}_{{\\mathrm{{raw}}}}(d,t) = \\mathrm{{tf}}(d,t) \\times \\mathrm{{idf}}(t)
$$

---

### Cosine similarity **between** primary and merged

1. Build one vector per document over the **same** vocabulary (sparse row).  
2. Apply **L2 normalization** to each row (`norm="l2"` in the vectorizer).  
3. **Cosine similarity** = dot product of the two **L2-normalized** rows (unit vectors).

Equivalently, if **u** and **v** are the **unnormalized** raw TF-IDF rows:

$$
\\cos = \\frac{{\\mathbf{{u}} \\cdot \\mathbf{{v}}}}{{\\|\\mathbf{{u}}\\|\\,\\|\\mathbf{{v}}\\|}}
$$

So it measures **overlap of weighted word counts**, not deep semantics (unlike embedding cosine in your paper).

---

**Note:** IDF is computed from **only these two** responses, so it is a **local** weighting, not Wikipedia-scale IDF.
"""
        )


def build_run_bundle(
    run_id: str,
    models: list[str],
    merger_model: str,
    temperature: float,
    max_workers: int,
    user_prompt: str,
    refinement_template: str,
    primary: list[str | None],
    critiques: list[list[str | None]],
    refined: list[list[str | None]],
    merged: list[str | None],
) -> dict:
    return {
        "run_id": run_id,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "merger_model": merger_model,
        "temperature": temperature,
        "max_workers": max_workers,
        "primary_prompt": user_prompt,
        "refinement_template": refinement_template,
        "primary_responses": primary,
        "critiques": critiques,
        "refined_responses": refined,
        "merged_responses": merged,
    }


def save_run_bundle_to_dir(bundle: dict, base_dir: str) -> Path:
    root = Path(base_dir).expanduser().resolve()
    run_dir = root / bundle["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "roundtable_run.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    return run_dir


def validate_refinement_template(template: str) -> None:
    refine_user_message(
        template,
        "PROMPT",
        "RESPONSE",
        "CRITIC",
        "CRITIQUE",
    )


def run_primary(
    client: OpenAI,
    models: list[str],
    user_prompt: str,
    temperature: float,
    max_workers: int,
) -> list[str | None]:
    n = len(models)
    out: list[str | None] = [None] * n

    def one(i: int) -> tuple[int, str]:
        text = chat_complete(
            client,
            models[i],
            [{"role": "user", "content": user_prompt}],
            temperature,
        )
        return i, text

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(one, i) for i in range(n)]
        for fut in concurrent.futures.as_completed(futs):
            i, text = fut.result()
            out[i] = text
    return out


def run_critiques(
    client: OpenAI,
    models: list[str],
    primary: list[str],
    user_prompt: str,
    temperature: float,
    max_workers: int,
) -> list[list[str | None]]:
    """critiques[i][j] = model j's critique of primary[i]; diagonal unused."""
    n = len(models)
    grid: list[list[str | None]] = [[None] * n for _ in range(n)]

    tasks: list[tuple[int, int]] = [(i, j) for i in range(n) for j in range(n) if j != i]

    def work(i: int, j: int) -> tuple[int, int, str]:
        label_i = f"LLM-{i + 1} ({models[i]})"
        msg = critique_user_message(user_prompt, label_i, primary[i] or "")
        text = chat_complete(
            client,
            models[j],
            [{"role": "user", "content": msg}],
            temperature,
        )
        return i, j, text

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(work, i, j) for i, j in tasks]
        for fut in concurrent.futures.as_completed(futs):
            i, j, text = fut.result()
            grid[i][j] = text
    return grid


def run_refine(
    client: OpenAI,
    models: list[str],
    primary: list[str],
    critiques: list[list[str | None]],
    user_prompt: str,
    refinement_template: str,
    temperature: float,
    max_workers: int,
) -> list[list[str | None]]:
    """refined[i][j] = model i's revision using critique from j; diagonal None."""
    n = len(models)
    refined: list[list[str | None]] = [[None] * n for _ in range(n)]

    tasks: list[tuple[int, int]] = [(i, j) for i in range(n) for j in range(n) if j != i]

    def work(i: int, j: int) -> tuple[int, int, str]:
        critic_label = f"LLM-{j + 1} ({models[j]})"
        msg = refine_user_message(
            refinement_template,
            user_prompt,
            primary[i] or "",
            critic_label,
            critiques[i][j] or "",
        )
        text = chat_complete(
            client,
            models[i],
            [{"role": "user", "content": msg}],
            temperature,
        )
        return i, j, text

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(work, i, j) for i, j in tasks]
        for fut in concurrent.futures.as_completed(futs):
            i, j, text = fut.result()
            refined[i][j] = text
    return refined


def run_merges(
    client: OpenAI,
    merger_model: str,
    models: list[str],
    refined: list[list[str | None]],
    user_prompt: str,
    temperature: float,
) -> list[str | None]:
    n = len(models)
    merged: list[str | None] = [None] * n
    for i in range(n):
        blocks: list[tuple[str, str]] = []
        for j in range(n):
            if j == i:
                continue
            t = refined[i][j]
            if t:
                blocks.append((f"LLM-{j + 1} ({models[j]})", t))
        if not blocks:
            merged[i] = None
            continue
        author_label = f"LLM-{i + 1} ({models[i]})"
        msg = merge_user_message(user_prompt, author_label, blocks)
        merged[i] = chat_complete(
            client,
            merger_model,
            [{"role": "user", "content": msg}],
            temperature,
        )
    return merged


def init_session_state() -> None:
    if "openrouter_key" not in st.session_state:
        st.session_state.openrouter_key = ""
    if "run_id" not in st.session_state:
        st.session_state.run_id = None
    if "primary" not in st.session_state:
        st.session_state.primary = None
    if "critiques" not in st.session_state:
        st.session_state.critiques = None
    if "refined" not in st.session_state:
        st.session_state.refined = None
    if "merged" not in st.session_state:
        st.session_state.merged = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "primary_prompt" not in st.session_state:
        st.session_state.primary_prompt = ""
    if "refinement_template" not in st.session_state:
        st.session_state.refinement_template = DEFAULT_REFINEMENT_TEMPLATE
    if "last_run_bundle" not in st.session_state:
        st.session_state.last_run_bundle = None
    if "_primary_txt_fingerprint" not in st.session_state:
        st.session_state._primary_txt_fingerprint = None
    if "_refinement_txt_fingerprint" not in st.session_state:
        st.session_state._refinement_txt_fingerprint = None


@st.dialog("OpenRouter API key")
def openrouter_dialog() -> None:
    st.caption("Stored only in this browser session (session state). Use secrets for deployment.")
    key = st.text_input("API key", type="password", value=st.session_state.openrouter_key or "")
    if st.button("Save", type="primary"):
        st.session_state.openrouter_key = key.strip()
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="LLM Roundtable", layout="wide")
    init_session_state()

    st.title("LLM Roundtable")
    st.caption("Cross-model critique → refine → merge")
    st.markdown(
        "Pipeline: **n** models answer the prompt → each response is critiqued by the other **n−1** → "
        "each author model refines using each critique → **n−1** refinements are **merged** by a merger model → "
        "compare **merged** vs that author's **primary** response (TF-IDF cosine + side-by-side)."
    )
    st.info(
        "**Where results are stored:** outputs are kept in **this browser session** (Streamlit `session_state`) "
        "until you refresh or close the tab—they are **not** written to disk unless you use **Save / download** below."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Set OpenRouter API key…"):
            openrouter_dialog()
        secret_key = ""
        try:
            secret_key = str(st.secrets["OPENROUTER_API_KEY"])
        except Exception:
            pass
        effective_key = (st.session_state.openrouter_key or secret_key or "").strip()
        key_ok = bool(effective_key)
        st.caption("Key set ✓" if key_ok else "Key not set — click to configure (or use secrets OPENROUTER_API_KEY).")
    with c2:
        n_llms = st.number_input("Number of LLMs (n)", min_value=2, max_value=10, value=4, step=1)
    with c3:
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)

    n = int(n_llms)
    dd_labels, dd_slugs = curated_dropdown_options()
    st.subheader("Models")
    st.caption(
        "Pick a model from the dropdown (OpenRouter catalog snapshot) or choose **Custom** and paste any slug from [openrouter.ai/models](https://openrouter.ai/models)."
    )
    models: list[str] = []
    cols = st.columns(min(3, n))
    for i in range(n):
        default_slug = DEFAULT_SLOT_SLUGS[i] if i < len(DEFAULT_SLOT_SLUGS) else "openai/gpt-4o-mini"
        default_idx = default_curated_index_for_slug(default_slug)
        with cols[i % len(cols)]:
            st.markdown(f"**LLM {i + 1}**")
            choice_i = st.selectbox(
                "Model",
                options=list(range(len(dd_labels))),
                format_func=lambda j: dd_labels[j],
                index=default_idx,
                key=f"model_pick_{i}",
                label_visibility="collapsed",
            )
            custom_i = (
                st.text_input(
                    "Custom slug",
                    value=default_slug,
                    key=f"model_custom_{i}",
                    placeholder="e.g. openai/gpt-5.4",
                    label_visibility="collapsed",
                )
                if choice_i == 0
                else ""
            )
            resolved = resolve_model_choice(choice_i, dd_labels, dd_slugs, custom_i)
            models.append(resolved)
            if choice_i != 0:
                st.caption(f"`{resolved}`")

    st.markdown("**Merger model** (synthesizes n−1 refinements per author)")
    merger_default_slug = models[0] if models else "openai/gpt-5.4"
    merger_idx = default_curated_index_for_slug(merger_default_slug)
    mc1, mc2 = st.columns([1, 1])
    with mc1:
        merger_choice = st.selectbox(
            "Merger dropdown",
            options=list(range(len(dd_labels))),
            format_func=lambda j: dd_labels[j],
            index=merger_idx,
            key="merger_pick",
            label_visibility="collapsed",
        )
    with mc2:
        merger_custom = (
            st.text_input(
                "Merger custom slug",
                value=merger_default_slug,
                key="merger_custom",
                placeholder="Merger OpenRouter slug",
                label_visibility="collapsed",
            )
            if merger_choice == 0
            else ""
        )
    merger_model = resolve_model_choice(merger_choice, dd_labels, dd_slugs, merger_custom)
    if merger_choice != 0:
        st.caption(f"Merger: `{merger_model}`")

    st.subheader("Prompts")
    st.caption("Upload `.txt` files below, or type in the boxes. Expand **How should my `.txt` files look?** for exact examples.")
    with st.expander("How should my `.txt` files look?", expanded=False):
        st.markdown(
            "### 1. Primary prompt (`primary.txt`)\n"
            "- **Encoding:** UTF-8 plain text.\n"
            "- **Content:** Whatever you want every roundtable model to answer—questions, instructions, rubrics, etc.\n"
            "- **No placeholders** in this file; the whole file becomes the user message for the primary phase.\n"
            "- **Layout:** Normal paragraphs and line breaks are kept.\n\n"
            "**Example `primary.txt`:**"
        )
        st.code(
            "You are helping with a medical informatics research design.\n\n"
            "Propose a utility formula for deciding whether a new concept should be\n"
            "inserted into an existing ontology. List factors and use plain-text math notation.\n",
            language="text",
        )
        st.markdown(
            "### 2. Improvement / refinement template (`refinement.txt`)\n"
            "- **Encoding:** UTF-8 plain text.\n"
            "- **Purpose:** This is the **user** message sent to the **original author** model when it revises its answer after seeing a critique.\n"
            "- **Required placeholders** (spell them exactly, including the curly braces):\n"
            "  - `{original_prompt}` — the same text as your primary prompt\n"
            "  - `{original_response}` — that model’s first answer\n"
            "  - `{critic_label}` — which model wrote the critique\n"
            "  - `{critique_text}` — the critique body\n"
            "- The app fills these in automatically; **do not** put your own text inside the braces.\n"
            "- If you need a **literal** `{` or `}` in the template (rare), double them: `{{` and `}}` (Python `str.format` rules).\n\n"
            "**Example `refinement.txt`** (same structure as the default in the text box below):"
        )
        st.code(
            "I previously asked you the following:\n\n"
            "{original_prompt}\n\n"
            "You responded with:\n"
            "---\n"
            "{original_response}\n"
            "---\n\n"
            "I passed your response to another system ({critic_label}) for critique. "
            "Here is the critique:\n"
            "---\n"
            "{critique_text}\n"
            "---\n\n"
            "Improve your original answer based on this critique. Address the critique directly "
            "while keeping correct content from your first answer. "
            "Repeat the full improved answer (not only a diff).\n",
            language="text",
        )
    up_pri, up_ref = st.columns(2)
    with up_pri:
        f_pri = st.file_uploader("Upload primary prompt (.txt)", type=["txt"], key="upload_primary_txt")
        if f_pri is not None:
            raw = f_pri.getvalue()
            fp = hashlib.sha256(raw).hexdigest()
            if fp != st.session_state._primary_txt_fingerprint:
                st.session_state.primary_prompt = raw.decode("utf-8", errors="replace")
                st.session_state._primary_txt_fingerprint = fp
    with up_ref:
        f_ref = st.file_uploader(
            "Upload improvement / refinement template (.txt)",
            type=["txt"],
            key="upload_refinement_txt",
            help="Must include placeholders {original_prompt}, {original_response}, {critic_label}, {critique_text}",
        )
        if f_ref is not None:
            raw_r = f_ref.getvalue()
            fpr = hashlib.sha256(raw_r).hexdigest()
            if fpr != st.session_state._refinement_txt_fingerprint:
                st.session_state.refinement_template = raw_r.decode("utf-8", errors="replace")
                st.session_state._refinement_txt_fingerprint = fpr

    st.text_area(
        "Primary prompt",
        height=160,
        placeholder="Your research / task prompt… (or upload a .txt above)",
        key="primary_prompt",
    )
    st.caption(
        "Improvement prompt: user message sent to the **author** model during refinement. "
        "Placeholders: `{original_prompt}`, `{original_response}`, `{critic_label}`, `{critique_text}`."
    )
    st.text_area(
        "Improvement (refinement) template",
        height=220,
        key="refinement_template",
    )

    with st.expander("Save & export (disk / JSON)", expanded=False):
        out_dir = st.text_input(
            "Output folder (created under your current working directory)",
            value="outputs",
            help="Each run saves to `<folder>/<run_id>/roundtable_run.json` when auto-save or Save now is used.",
        )
        auto_save = st.checkbox("Save to disk automatically after a successful run", value=False)
        st.caption(
            "Use **Download run as JSON** to grab the latest completed run without writing to the server filesystem."
        )

    max_workers = st.slider("Parallel API workers", min_value=1, max_value=32, value=8)

    est_calls = n + n * (n - 1) + n * (n - 1) + n
    st.caption(
        f"Approx. API calls per full run: **{est_calls}** "
        f"(primary {n} + critiques {n * (n - 1)} + refinements {n * (n - 1)} + merges {n})."
    )

    user_prompt = (st.session_state.primary_prompt or "").strip()
    refinement_template = (st.session_state.refinement_template or "").strip()

    run = st.button(
        "Run full pipeline",
        type="primary",
        disabled=not key_ok or not user_prompt or not refinement_template,
    )

    if run:
        st.session_state.last_error = None
        try:
            validate_refinement_template(refinement_template)
        except ValueError as e:
            st.session_state.last_error = str(e)
            st.error(str(e))
        else:
            client = get_client(effective_key)
            run_uuid = str(uuid.uuid4())[:8]
            st.session_state.run_id = run_uuid
            try:
                with st.status("Running pipeline…", expanded=True) as status:
                    st.write("Phase 1: primary responses")
                    primary = run_primary(client, models, user_prompt, temperature, max_workers)
                    st.session_state.primary = primary

                    st.write("Phase 2: cross-model critiques")
                    critiques = run_critiques(
                        client, models, primary, user_prompt, temperature, max_workers
                    )
                    st.session_state.critiques = critiques

                    st.write("Phase 3: refinements (critique fed back to original author)")
                    refined = run_refine(
                        client,
                        models,
                        primary,
                        critiques,
                        user_prompt,
                        refinement_template,
                        temperature,
                        max_workers,
                    )
                    st.session_state.refined = refined

                    st.write("Phase 4: merge refinements per author")
                    merged = run_merges(
                        client,
                        merger_model.strip(),
                        models,
                        refined,
                        user_prompt,
                        temperature,
                    )
                    st.session_state.merged = merged
                    bundle = build_run_bundle(
                        run_uuid,
                        models,
                        merger_model.strip(),
                        temperature,
                        max_workers,
                        user_prompt,
                        refinement_template,
                        primary,
                        critiques,
                        refined,
                        merged,
                    )
                    st.session_state.last_run_bundle = bundle
                    if auto_save:
                        try:
                            saved_to = save_run_bundle_to_dir(bundle, out_dir.strip() or "outputs")
                            st.success(f"Saved run to `{saved_to}`")
                        except OSError as oe:
                            st.warning(f"Could not save to disk: {oe}")
                    status.update(label="Pipeline complete", state="complete", expanded=False)
            except Exception as e:
                st.session_state.last_error = repr(e)
                st.error(f"Run failed: {e}")

    if st.session_state.last_run_bundle:
        b = st.session_state.last_run_bundle
        j = json.dumps(b, ensure_ascii=False, indent=2)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label="Download latest run (JSON)",
                data=j,
                file_name=f"roundtable_run_{b.get('run_id', 'export')}.json",
                mime="application/json",
            )
        with c2:
            if st.button("Save latest run to disk now"):
                try:
                    saved_to = save_run_bundle_to_dir(b, out_dir.strip() or "outputs")
                    st.success(f"Wrote `{saved_to / 'roundtable_run.json'}`")
                except OSError as oe:
                    st.error(str(oe))

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    primary = st.session_state.primary
    merged = st.session_state.merged
    if primary and merged:
        st.divider()
        st.subheader("Merge vs primary (per author LLM)")
        render_tfidf_methodology_expander()
        for i in range(len(models)):
            p = primary[i] or ""
            m = merged[i] or ""
            sim = tfidf_cosine(p, m)
            wc_p = len(p.split())
            wc_m = len(m.split())
            with st.expander(f"LLM {i + 1}: {models[i]}", expanded=(i == 0)):
                st.metric(
                    "TF-IDF cosine(primary, merged)",
                    f"{sim:.4f}",
                    help="L2-normalized bag-of-words TF-IDF cosine (see expander above for formulas).",
                )
                st.caption(f"Words — primary: {wc_p}, merged: {wc_m}")
                with st.expander("Worked TF / IDF calculations for this primary vs merged pair", expanded=False):
                    bd = tfidf_pair_breakdown(p, m, top_n=20)
                    if bd is None:
                        st.caption("Not enough tokenizable text to build TF-IDF vectors.")
                    else:
                        st.markdown(
                            f"Shared **2-document** corpus; vocabulary size **{bd['vocab_size']}** "
                            f"(cap {TFIDF_MAX_FEATURES}, English stop words removed)."
                        )
                        st.markdown(
                            "**IDF** — sklearn `idf_` equals "
                            f"`ln((1+n)/(1+df))+1` with *n* = 2: **{'yes' if bd['idf_matches_df_formula'] else 'no'}**."
                        )
                        if bd["sample_idf_rows"]:
                            st.markdown("Sample terms (verify **df** → **idf**):")
                            st.dataframe(bd["sample_idf_rows"], hide_index=True, use_container_width=True)
                        st.markdown("**Primary** — top terms by **|tf × idf|** (raw weights, before row L2 norm):")
                        st.dataframe(
                            bd["primary_top"] if bd["primary_top"] else [{"note": "(no nonzero weights)"}],
                            hide_index=True,
                            use_container_width=True,
                        )
                        st.markdown("**Merged** — same:")
                        st.dataframe(
                            bd["merged_top"] if bd["merged_top"] else [{"note": "(no nonzero weights)"}],
                            hide_index=True,
                            use_container_width=True,
                        )
                        st.markdown(
                            f"**Cosine (displayed metric)** = **{bd['cosine_l2']:.6f}** — dot product of **L2-normalized** "
                            f"raw TF-IDF rows. Same as cosine of unnormalized rows: **{bd['cosine_manual_raw_vectors']:.6f}** "
                            f"(match: **{'yes' if bd['cosine_match'] else 'no'}**)."
                        )
                col_pri, col_mer = st.columns(2)
                with col_pri:
                    st.markdown("**Primary**")
                    st.markdown(p or "_empty_")
                with col_mer:
                    st.markdown("**Merged refinements**")
                    st.markdown(m or "_empty_")

    if st.session_state.refined and st.checkbox("Show intermediate: critiques & refinements", value=False):
        crit = st.session_state.critiques
        ref = st.session_state.refined
        for i in range(len(models)):
            st.markdown(f"### Author LLM {i + 1}")
            for j in range(len(models)):
                if j == i:
                    continue
                st.markdown(f"**Critique by LLM {j + 1}**")
                st.markdown((crit[i][j] if crit else "") or "_missing_")
                st.markdown(f"**Refined by LLM {i + 1} after that critique**")
                st.markdown((ref[i][j] if ref else "") or "_missing_")


if __name__ == "__main__":
    main()
