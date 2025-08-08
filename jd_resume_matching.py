from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from utils.timer import timer
from utils.cache import DiskCache, sha256_text
from utils.device import choose_device
from typing import List, Dict, Tuple, Callable, Optional
from collections import Counter
import re

# embedding model name. kept here so it's easy to swap later if we ever want
MODEL_NAME = "intfloat/e5-base-v2"

EMBEDDER = None
TOK = None
DEVICE = "cpu"
CACHE = DiskCache(".cache_emb")  # tiny disk cache so reruns are faster

def init_model(device_pref: str = "cpu"):
    """
    loads the embedding model on requested device. default cpu so it runs anywhere.
    """
    global EMBEDDER, TOK, DEVICE
    device = choose_device(device_pref)
    if EMBEDDER is not None and device == DEVICE:
        return DEVICE
    DEVICE = device
    EMBEDDER = SentenceTransformer(MODEL_NAME, device=DEVICE)
    TOK = AutoTokenizer.from_pretrained(MODEL_NAME)
    # try not to exceed max length. just being careful, seen this explode before.
    try:
        max_len = getattr(TOK, "model_max_length", 512)
        EMBEDDER.max_seq_length = min(max_len, 510)
    except Exception:
        pass
    return DEVICE

# init on import so things work on cpu out of the box
init_model("cpu")

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def _token_count(t: str) -> int:
    return len(TOK.encode(t, add_special_tokens=False))

def _clip_ids_to(text: str, keep: int = 510) -> str:
    ids = TOK.encode(text or "", add_special_tokens=True)
    if len(ids) > keep:
        ids = ids[:keep]
    return TOK.decode(ids, skip_special_tokens=True)

def condense_jd(jd_text: str, hard_token_cap: int = 400) -> str:
    """
    tries to keep the most important lines from the JD. this is pretty heuristic.
    """
    jd_text = clean_text(jd_text)
    lines = re.split(r"(?:•|-|\n|\r|\u2022)", jd_text)
    lines = [l.strip(" .;:") for l in lines if l and len(l.strip()) > 0]
    if len(lines) < 5:
        lines = re.split(r"(?<=[.!?])\s+", jd_text)

    must_terms = ["responsibilities","requirements","must","required","you will","experience","skills",
                  "nice to have","qualifications","we're looking for","preferred","expertise"]
    skillish = re.findall(r"[A-Za-z0-9\+\#\.\-]{2,}", jd_text)
    top_terms = {w.lower() for w, _ in Counter(skillish).most_common(50)}

    def line_score(l):
        s = l.lower()
        score = 0
        score += sum(1 for mt in must_terms if mt in s)
        score += sum(1 for w in s.split() if w in top_terms)
        score += (1 if len(l) < 220 else 0)
        return score

    ranked = sorted(lines, key=line_score, reverse=True)
    selected = []
    for l in ranked:
        selected.append(l)
        if _token_count(" ".join(selected)) > hard_token_cap:
            break
    condensed = " ".join(selected)
    return _clip_ids_to(condensed, keep=510)

def chunk_text_for_e5(text: str, max_tokens=340, overlap=48):
    """
    split long resumes into overlapping chunks. avoids losing sections.
    """
    text = clean_text(text)
    ids = TOK.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    stride = max_tokens - overlap
    while start < len(ids):
        piece = ids[start:start+max_tokens]
        chunk = TOK.decode(piece)
        chunk = _clip_ids_to(chunk, keep=510)
        chunks.append(chunk)
        start += stride
    return chunks

def encode_query(jd_text: str):
    jd = f"query: {clean_text(jd_text)}"
    jd = _clip_ids_to(jd, keep=510)
    key = sha256_text(jd)
    cached = CACHE.get("emb_q", key)
    if cached is not None:
        return cached
    emb = EMBEDDER.encode(jd, convert_to_tensor=True, device=DEVICE)
    CACHE.set("emb_q", key, emb)
    return emb

def encode_passages(passages: List[str]):
    prefixed = [f"passage: {clean_text(p)}" for p in passages]
    prefixed = [_clip_ids_to(p, keep=510) for p in prefixed]
    key = sha256_text("\n".join(prefixed))
    cached = CACHE.get("emb_p", key)
    if cached is not None:
        return cached
    emb = EMBEDDER.encode(prefixed, convert_to_tensor=True, device=DEVICE)
    CACHE.set("emb_p", key, emb)
    return emb

def bm25_scores(query: str, passages: List[str]) -> List[float]:
    tokens = [p.lower().split() for p in passages]
    bm25 = BM25Okapi(tokens)
    q = query.lower().split()
    return bm25.get_scores(q).tolist()

def section_bonus_score(text: str, bonus=0.02) -> float:
    s = text.lower()
    for h in ["skills", "experience", "work experience", "projects"]:
        if h in s[:140]:
            return bonus  # small nudge if a chunk looks like a skills section
    return 0.0

def _normalize(v: List[float]) -> List[float]:
    if not v:
        return v
    mn, mx = min(v), max(v)
    if mx - mn < 1e-9:
        return [0.0 for _ in v]
    return [(x - mn) / (mx - mn) for x in v]

def score_resume_vs_jd(jd_emb, jd_text: str, resume_text: str,
                       use_bm25: bool = False, alpha: float = 0.75,
                       chunk_tokens=340, overlap=48,
                       rel_threshold=0.30, coverage_boost=0.03, cap_coverage=5,
                       section_boost_val=0.02):
    """
    score a single resume by chunk then combine. super standard ranking thing.
    """
    chunks = chunk_text_for_e5(resume_text, max_tokens=chunk_tokens, overlap=overlap)
    if not chunks:
        return {"final_score": 0.0, "max_sim": 0.0, "coverage": 0, "per_chunk": [], "chunks": []}

    emb = encode_passages(chunks)
    sims = util.cos_sim(jd_emb, emb).squeeze(0).tolist()
    sims_n = _normalize(sims)

    if use_bm25:
        bm25 = bm25_scores(condense_jd(jd_text, 300), chunks)
        bm25_n = _normalize(bm25)
    else:
        bm25_n = [0.0 for _ in sims_n]

    blended = [alpha*s + (1-alpha)*b + section_bonus_score(chunks[i], bonus=section_boost_val)
               for i, (s, b) in enumerate(zip(sims_n, bm25_n))]

    max_cosine = max(sims) if sims else 0.0
    top_sorted = sorted(sims, reverse=True)
    top2 = top_sorted[1] if len(top_sorted) > 1 else 0.0
    coverage = sum(1 for s in blended if s >= rel_threshold)
    coverage = min(coverage, cap_coverage)
    final_score = (max(blended) if blended else 0.0) + coverage * coverage_boost

    return {
        "final_score": float(final_score),
        "max_sim": float(max(blended) if blended else 0.0),
        "cosine_max": float(max_cosine),
        "cosine_top2": float(top2),
        "coverage": int(coverage),
        "per_chunk": [{"chunk": i, "blend": float(blended[i]), "cosine": float(sims[i])} for i in range(len(chunks))],
        "chunks": chunks,
    }

def top_support_snippets(score_out, top_k=2):
    pairs = sorted(
        zip(score_out["chunks"], [p["blend"] for p in score_out["per_chunk"]]),
        key=lambda x: x[1], reverse=True
    )[:top_k]
    return [{"snippet": p[0], "score": float(p[1])} for p in pairs]

def rank_candidates(jd_text: str, resumes: List[Dict],
                    cosine_only: bool = True,
                    use_bm25: bool = False, alpha=0.75,
                    chunk_tokens=340, overlap=48,
                    rel_threshold=0.30, coverage_boost=0.03, section_boost_val=0.02,
                    progress_cb: Optional[Callable[[int, int, str], None]] = None
                   ) -> Tuple[List[Dict], dict]:
    """
    loop over resumes, score them, collect results.
    optional progress_cb lets the UI tick a bar.
    """
    timings = {}
    total = len(resumes)
    if progress_cb:
        progress_cb(0, total, "Embedding & scoring")

    with timer("condense_jd", timings):
        jd_condensed = condense_jd(jd_text, hard_token_cap=400)
    with timer("encode_q", timings):
        jd_emb = encode_query(jd_condensed)

    results = []
    with timer("score_resume", timings):
        for i, r in enumerate(resumes):  # track original index so summaries map right
            out = score_resume_vs_jd(
                jd_emb, jd_condensed, r["text"],
                use_bm25=use_bm25, alpha=alpha,
                chunk_tokens=chunk_tokens, overlap=overlap,
                rel_threshold=rel_threshold, coverage_boost=coverage_boost, section_boost_val=section_boost_val
            )
            results.append({
                "id": r.get("id", "Unknown"),
                "raw_index": i,
                "final_score": out["final_score"],
                "max_sim": out["max_sim"],
                "cosine_max": out["cosine_max"],
                "cosine_top2": out["cosine_top2"],
                "coverage": out["coverage"],
                "per_chunk": out["per_chunk"],
                "support_snippets": top_support_snippets(out, top_k=2)
            })

            if progress_cb:
                progress_cb(i + 1, total, "Embedding & scoring")

    # sort by what the user picked (cosine only vs blended final score)
    results.sort(key=lambda x: x["cosine_max"] if cosine_only else x["final_score"], reverse=True)
    return results, timings
