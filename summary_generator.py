# helper to ask groq for a short recruiter-style explanation.
# btw: this writes exactly N sentences (we top-up or trim if needed).

import os, time, requests, re
from dotenv import load_dotenv
import streamlit as st  # using this only to read secrets when hosted

load_dotenv()
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

def _get_groq_key() -> str | None:
    # first try environment variable, then streamlit secrets (on cloud)
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GROQ_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        return None

def _clip_chars_by_tokens_approx(s: str, max_tokens: int) -> str:
    # rough rule: around 4 chars is one token (close enough for this use)
    if not s:
        return ""
    return s[: max_tokens * 4]

_SENT_RE = re.compile(r'\s*(.*?[\.!\?])(?:\s+|$)', re.DOTALL)

def _split_sentences(text: str):
    bits, i = [], 0
    t = (text or "").strip()
    while i < len(t):
        m = _SENT_RE.match(t, i)
        if not m:
            break
        bits.append(m.group(1).strip())
        i = m.end()
    return [b for b in bits if b]

def _join_first_n_sentences(text: str, n: int) -> str:
    sents = _split_sentences(text)
    return " ".join(sents[:n]).strip()

def _post_json_with_retries(headers, data, max_attempts=4, base_wait=1.2):
    last = "Unknown error"
    for k in range(max_attempts):
        try:
            r = requests.post(API_URL, headers=headers, json=data, timeout=30)
            r.raise_for_status()
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            wait_s = base_wait * (k + 1)
            try:
                body = e.response.json()
                msg = (body.get("error", {}) or {}).get("message", "")
                last = f"{status} HTTPError: {msg}" if msg else f"{status} HTTPError"
                m = re.search(r"try again in ([0-9.]+)s", msg or "")
                if m:
                    wait_s = float(m.group(1)) + 0.5
            except Exception:
                last = f"{status} HTTPError"
            if status in (429, 500, 502, 503, 504):
                time.sleep(wait_s); continue
            break
        except requests.exceptions.RequestException as e:
            last = str(e) or "RequestException"
            time.sleep(base_wait * (k + 1)); continue
    return f"[AI summary error: {last}]"

def generate_ai_fit_summary(
    verdict: str,
    jd_text: str,
    resume_text: str,
    matched_skills: list,
    gaps: list,
    n_sentences: int = 4,
) -> str:
    api_key = _get_groq_key()
    if not api_key:
        return "_(AI summary skipped: GROQ_API_KEY not set.)_"

    # clip context so it's focused and safe length wise
    JD_TOK = 500
    CV_TOK = 700
    jd = _clip_chars_by_tokens_approx(jd_text.strip(), JD_TOK)
    cv = _clip_chars_by_tokens_approx(resume_text.strip(), CV_TOK)

    # a little grounding: surface a couple matched skills + any gaps
    ev_lines = []
    for e in (matched_skills or [])[:4]:
        sk = e.get("jd_skill", "")
        qt = e.get("quote", "")
        if sk or qt:
            ev_lines.append(f"- {sk}: {qt}")
    gap_lines = [f"- {g}" for g in (gaps or [])[:4]]

    prompt = f"""You are an expert technical recruiter. Explain why this candidate is a **{verdict}** for the job.
Use ONLY the information below; do not invent facts. Use the exact phrase “{verdict}” in the first sentence and do not contradict it.

Write EXACTLY {n_sentences} sentences in one compact paragraph. Keep it crisp and recruiter-friendly. No bullet points.

Job Description (excerpt):
{jd}

Resume (excerpt):
{cv}

Matched evidence:
{os.linesep.join(ev_lines) if ev_lines else "(none)"}

Key gaps:
{os.linesep.join(gap_lines) if gap_lines else "(none)"}
"""

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": min(1200, max(160, n_sentences * 48)),
    }

    text = _post_json_with_retries(headers, data)
    if text.startswith("[AI summary error:"):
        return text

    # trim if too long, or ask a tiny top-up if it was short
    sents = _split_sentences(text)
    if len(sents) > n_sentences:
        return " ".join(sents[:n_sentences]).strip()

    if len(sents) < n_sentences:
        remaining = n_sentences - len(sents)
        follow = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": " ".join(sents)},
                {"role": "user", "content": f"Add {remaining} new sentence(s) to make it exactly {n_sentences}. No bullets. No repeating points."},
            ],
            "temperature": 0.2,
            "max_tokens": max(80, remaining * 48),
        }
        more = _post_json_with_retries(headers, follow)
        if more and not more.startswith("[AI summary error:"):
            combined = (" ".join(sents) + " " + more.strip()).strip()
            return _join_first_n_sentences(combined, n_sentences)

    return " ".join(sents[:n_sentences]).strip()
