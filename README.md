# CandidateLens AI — Candidate Recommendation Engine

A simple web app that ranks resumes against a job description using cosine similarity (Also have option of "E5 embeddings + cosine"). Designed with simple UI (clean cards, minimal jargon) with toggles for advanced ranking and AI “Why this fit?” explanations powered by Groq.

- **CPU by default**, GPU optional when available.
- **One‑screen workflow:** Paste JD → Upload resumes → Rank candidates.
- **Evidence‑based fit cards:** verdict pill, donut chart, (optional) AI rationale grounded in snippets.
- **Export:** Download a CSV of **all** candidates (not just Top‑K). AI summary only for Top‑K to control cost.
- **Exact sentence control:** Choose 2–10 sentences for AI rationale (no more cut‑offs).

---

## Table of Contents
- [Webapp link (Streamlit Cloud)](#webapp-link-streamlit-cloud)
- [Why Sentence Toggle (makeshift, cost-aware)](#why-sentence-toggle-makeshift-cost-aware)
- [Core Features](#core-features)
- [Why These Choices](#why-these-choices)
- [System Architecture](#system-architecture)
- [How Matching Works (Theory)](#how-matching-works-theory)
- [AI Explanations (Groq)](#ai-explanations-groq)
- [Install & Run Locally](#install--run-locally)
- [Environment Variables](#environment-variables)
- [Deploy to Streamlit Cloud](#deploy-to-streamlit-cloud)
- [Using the App](#using-the-app)
- [CSV Output](#csv-output)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [Security & Privacy](#security--privacy)
- [Extending the Project](#extending-the-project)
- [Roadmap](#roadmap)
- [License](#license)

---

## Webapp link (Streamlit Cloud)
`https://candidatelensai.streamlit.app/`

---

## Why Sentence Toggle (makeshift, cost-aware)
**TL;DR:** The sentence toggle lets you pick an exact number of sentences (2–10) for the AI explanation. We use this to keep responses short, predictable, and **cheap**.

- **Speed & token usage:** We don’t have paid inference/LLM resources here. Exact sentence count prevents long rambles that burn tokens and slow down results.
- **No truncation weirdness:** Instead of cutting off mid‑sentence, we ask the model for *exactly* N sentences. If it writes fewer, we top‑up with a tiny follow‑up. If it writes more, we trim.
- **Consistent UX:** Recruiters want quick, skimmable context. This gives that — the same length every time.
- **Makeshift by design:** This is a practical control for a budget setup. If/when we switch to a paid LLM or a server‑side controller, we can swap this out for smarter length/quality settings.

---

## Core Features

### Input
- Paste Job Description (plain text)
- Upload multiple resumes as **.pdf** or **.txt**

### Ranking
- Embeddings: `intfloat/e5-base-v2` (free, strong retrieval quality)
- **Cosine‑only (default):** fast, simple match score
- **Optional Hybrid:** BM25 + embeddings, single balance slider (under “Advanced ranking”)

### Fit Cards
- Verdict pill (Strong / Partial / Weak) + Confidence
- Donut chart (green/red)
- Evidence expander with top matching snippets
- Optional AI summary (exact **N sentences**, 2–10)

### Export
- Download CSV with **all** candidates (sorted)
- AI summary is filled only for Top‑K shown in the UI; others left blank to save tokens

### Ops & UX
- CPU default, GPU optional (`cpu`/`auto`/`cuda`)
- Tokenizer‑safe clipping avoids “>512” crashes
- Two‑phase progress bars: **scoring → AI summaries**
- Clear errors on unreadable PDFs

---

## Why These Choices
**E5‑base‑v2 vs others.** E5 is trained for retrieval. Using `query:`/`passage:` formatting gives solid JD↔resume results. It’s small enough for CPU.

**Cosine‑only default.** HR users value clarity/speed. Hybrid is available but tucked away.

**Chunked resumes.** Resumes are long. We split into ~340‑token chunks (~48 overlap). Keeps local relevance (like a “Projects” section) without truncation.

**Coverage & verdict.** Coverage counts how many chunks are relevant (avoids single‑spike flukes). Verdict = function of match score + coverage; confidence also looks at the margin vs #2 chunk.

**Sentence‑controlled AI.** Recruiters asked for fixed‑length summaries. You choose **2–10 sentences**; the app trims or tops‑up to hit the exact count.

---

## System Architecture
```
app.py
 ├─ Sidebar: device, Cosine-only, Top‑K, AI Summary toggle, sentence count
 ├─ JD panel: paste/upload JD (.txt)
 ├─ Resume upload: .pdf or .txt (multiple)
 ├─ Rank Candidates button
 │
 ├─ jd_resume_matching.py
 │   ├─ condense_jd()        # keeps signal, token‑safe
 │   ├─ chunk_text_for_e5()  # splits long resumes
 │   ├─ encode_query()/encode_passages()
 │   ├─ bm25_scores() (optional)
 │   ├─ score_resume_vs_jd()
 │   └─ rank_candidates()    # aggregates, sorts, attaches raw_index
 │
 ├─ fit_card.py
 │   ├─ extract_skills_from_jd()
 │   ├─ find_matches()       # map skills to snippets
 │   └─ build_fit_card()     # verdict, confidence, recommendation, gaps
 │
 ├─ summary_generator.py
 │   └─ generate_ai_fit_summary()  # Groq Llama‑3.1, exact N sentences
 │
 └─ text_extraction.py / name_extraction.py
     └─ PDF/TXT parsing and naive name guess
```

---

## How Matching Works (Theory)

### Preprocessing
- JD condensed to the most relevant lines (responsibilities/requirements/skills), clipped to ≤510 tokens
- Resumes chunked into overlapping pieces; each clipped to ≤510 tokens

### Embeddings
- Model: `intfloat/e5-base-v2`
- Prompts: `query: ...` for JD; `passage: ...` for resume chunks
- Disk caching (`.cache_emb`) speeds up reruns

### Scoring
- Cosine similarity per chunk (JD vs chunk)
- Optional BM25 keyword scores per chunk
- Blended per‑chunk score = `α·cosine_norm + (1−α)·bm25_norm + section_bonus`
- Aggregate:
  - `max(blended)` (best chunk)
  - `coverage` (how many chunks exceed a threshold)
- Final candidate score = `max_blended + coverage * coverage_boost`

### Verdict & Confidence
- **Strong Fit:** `cosine_max ≥ 0.72` and `coverage ≥ 2`
- **Partial Fit:** `cosine_max ≥ 0.58`
- **Else:** Weak Fit
- Confidence considers score strength + margin vs #2 chunk

### Display
- Sorted results → Top‑K cards: match %, verdict, confidence, optional AI rationale, and evidence

---

## AI Explanations (Groq)
- Toggle: “AI summary of fit (Groq)”
- When ON, calls Groq (OpenAI‑compatible) — default `llama-3.1-8b-instant`
- We pass: **verdict**, JD/resume excerpts (clipped), matched evidence, and gaps
- Cost control: Summaries only run for Top‑K on screen (CSV has all rows)

---

## Install & Run Locally

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Optional (for AI):**
```bash
export GROQ_API_KEY=your_groq_key_here
# optional: export GROQ_MODEL=llama-3.1-8b-instant
```

**Launch:**
```bash
streamlit run app.py
```

App opens at <http://localhost:8501>. Device defaults to CPU; switch to `auto`/`cuda` in the sidebar.

---

## Environment Variables
- `GROQ_API_KEY` — required for AI summaries
- `GROQ_MODEL` — optional; default `llama-3.1-8b-instant`

If no key is set, the app still runs; summaries are skipped.

---

## Deploy to Streamlit Cloud
1. Push to GitHub
2. Streamlit Cloud → **New app** → pick repo/branch → `app.py`
3. **Secrets**:
   ```toml
   GROQ_API_KEY = "your_groq_key_here"
   GROQ_MODEL = "llama-3.1-8b-instant"
   ```
4. (Optional) Set Python 3.10/3.11
5. Deploy

---

## Using the App
- Paste JD or upload a `.txt` JD
- Upload resumes (`.pdf`/`.txt`)
- Sidebar knobs:
  - Compute device: `cpu` / `auto` / `cuda`
  - Cosine‑only (default ON); turn OFF to reveal hybrid + sliders
  - Top‑K (1–10)
  - AI summary toggle + **sentence count (2–10)**
- Click **Rank Candidates**
- Review top cards → “Why this fit” (if AI ON) → expand **Evidence**
- **Download results (.csv)** to export all candidates

---

## CSV Output
Columns (typical):
- `id`, `match_score_cosine`, `coverage_snippets`, `hybrid_score`
- `verdict`, `confidence`, `recommendation`
- `matched_skills`, `gaps`
- `ai_summary` (filled only for Top‑K when AI is ON)

---

## Performance Tips
- CPU is fine for 10–100 resumes; GPU helps at 200+
- Embedding cache speeds reruns (`.cache_emb`)
- Chunk size ~340 tokens; overlap ~48
- Keep Top‑K small to speed AI

---

## Troubleshooting
- **Groq 429/5xx** → retry/backoff already included; lower Top‑K or turn AI OFF
- **Wrong summary for candidate** → we pass `raw_index` to keep mapping correct
- **Unreadable PDFs** → scanned PDFs need OCR (not included by default)

---

## Security & Privacy
- Files are processed in memory/temp during a session
- No external calls unless AI is ON; then only clipped snippets go to Groq

---

## Extending the Project
- Better name extraction (spaCy NER)
- Cross‑encoder reranker for top‑N
- Rich skill ontology (ESCO/ONET) normalization
- Feedback loop to learn org‑specific weights

---

## Roadmap
- OCR path for scanned PDFs
- Per‑company tuning profiles
- In‑app annotation for hiring feedback

