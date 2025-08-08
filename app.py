import os, time, tempfile
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from text_extraction import extract_text_from_file
from name_extraction import extract_candidate_name
from jd_resume_matching import rank_candidates, condense_jd, init_model
from fit_card import build_fit_card
from summary_generator import generate_ai_fit_summary

st.set_page_config(page_title="CandidateLens AI — Candidate Recommendation (v8.1)", layout="wide")

BRAND = {
    "name": "CandidateLens AI",
    "tagline": "Smart candidate recommendations — built for hiring excellence",
    "primary": "#1f77b4",
    "accent":  "#f39c12",
    "success": "#2ecc71",
    "danger":  "#e74c3c",
    "muted":   "#7f8c8d",
}

def hero():
    st.markdown(f"""
    <div style="background:{BRAND['primary']};padding:18px 22px;border-radius:14px;margin-bottom:14px">
      <div style="color:white;font-size:36px;font-weight:800;text-align:center">{BRAND['name']}</div>
      <div style="color:white;opacity:.95;text-align:center;font-size:20px">{BRAND['tagline']}</div>
    </div>
    """, unsafe_allow_html=True)

hero()

with st.sidebar:
    st.header("Settings")
    device_pref = st.selectbox("Compute Device", ["cpu", "auto", "cuda"], index=0, help="CPU is safest. 'auto' uses GPU if available.")
    actual_device = init_model(device_pref)
    st.caption(f"Using device: **{actual_device}**")

    cosine_only = st.toggle("Cosine-only ranking (simple & fast)", value=True,
                            help="Fast, simple match based on the words in the JD and resumes.")
    use_bm25 = False
    alpha = 0.75
    chunk_tokens = 340
    overlap = 48
    section_boost_amt = 0.02

    if not cosine_only:
        st.markdown("**Advanced ranking options**")
        use_bm25 = st.checkbox("Use Hybrid Ranking (improves keyword coverage)", value=True,
                               help="Blend keyword matching (BM25) with semantic similarity.")
        alpha = st.slider("Balance keywords ↔ semantics", 0.0, 1.0, 0.75, 0.05,
                          help="0 = keywords only, 1 = semantics only")
        with st.expander("Engineer settings (optional)"):
            chunk_tokens = st.slider("Chunk size (tokens)", 280, 420, 340, 10)
            overlap = st.slider("Chunk overlap (tokens)", 32, 96, 48, 8)
            section_boost_amt = st.slider("Section boost (Skills/Experience)", 0.0, 0.06, 0.02, 0.005)

    topk = st.slider("How many candidates to show", 1, 10, 5, 1)
    ai_summary_on = st.toggle(
        "AI summary of fit (Groq)",
        value=True,
        help="ON: calls Groq to explain why the candidate is a great/okay/bad fit. OFF: show minimal list (name + score + donut)."
    )
    st.caption("Set GROQ_API_KEY in your environment to enable AI summaries.")
    st.divider()
    dl_ready = st.checkbox("Prepare CSV for download after ranking", value=True)

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("Job Description")
    jd_text = st.text_area("Paste JD here", height=260, placeholder="Paste the job description…")
    jd_file = st.file_uploader("or Upload JD (.txt)", type=["txt"], accept_multiple_files=False)
    if jd_file:
        jd_text = jd_file.read().decode("utf-8", errors="ignore")
    # (Condensed preview intentionally removed)

with right:
    st.subheader("Resumes")
    files = st.file_uploader("Upload resumes (.pdf or .txt)", type=["pdf", "txt"], accept_multiple_files=True)
    run = st.button("Rank Candidates", type="primary", use_container_width=True)

def donut_plotly(score_percent: float, key=None):
    fig = go.Figure(data=[go.Pie(
        values=[max(0, min(100.0, score_percent)), 100 - max(0, min(100.0, score_percent))],
        hole=0.65,
        marker=dict(colors=[BRAND["success"], BRAND["danger"]]),
        textinfo="none"
    )])
    fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0),
                      annotations=[dict(text=f"{score_percent:.1f}%", x=0.5, y=0.5, font_size=18, showarrow=False)])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)

if run and jd_text and files:
    # Read resumes
    resumes = []
    for f in files:
        data = f.read()
        suffix = ".pdf" if f.name.lower().endswith(".pdf") else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(data)
            tmp_path = tf.name
        text = extract_text_from_file(tmp_path)
        rid = f.name.rsplit(".", 1)[0]
        name = extract_candidate_name(text, fallback_id=rid)
        resumes.append({"id": name, "text": text, "filename": f.name})

    # Rank
    results, timings = rank_candidates(
        jd_text, resumes,
        cosine_only=cosine_only,
        use_bm25=use_bm25, alpha=alpha,
        chunk_tokens=chunk_tokens, overlap=overlap,
        rel_threshold=0.30, coverage_boost=0.03, section_boost_val=section_boost_amt
    )

    st.subheader("Top candidates")
    chip_line = f"**Cosine-only:** {'ON' if cosine_only else 'OFF'}  •  **Top {topk}**  •  **Device:** {actual_device.upper()}"
    if not cosine_only:
        chip_line += f"  •  **Hybrid:** {'ON' if use_bm25 else 'OFF'} (α={alpha:.2f})"
    st.caption(chip_line)

    # ===== Build CSV rows for ALL ranked candidates (no AI yet) =====
    rows_all = []
    for r in results:
        fit_all = build_fit_card(jd_text, {
            "id": r["id"],
            "cosine_max": r["cosine_max"],
            "coverage": r["coverage"],
            "support_snippets": r["support_snippets"],
            "top2_cosine": r.get("cosine_top2", 0.0),
        })
        rows_all.append({
            "raw_index": r.get("raw_index", None),  # kept internally so we can map AI summaries later
            "filename": resumes[r["raw_index"]].get("filename", "") if r.get("raw_index") is not None else "",
            "id": r["id"],
            "match_score_cosine": r["cosine_max"],
            "coverage_snippets": r["coverage"],
            "hybrid_score": r["final_score"],
            "verdict": fit_all["verdict"],
            "confidence": fit_all["confidence"],
            "recommendation": fit_all["recommendation"],
            "matched_skills": "; ".join(m["jd_skill"] for m in fit_all["matched"]),
            "gaps": "; ".join(fit_all["gaps"]),
            "ai_summary": "",  # will fill for top-k only if AI toggle is ON
        })
    # ================================================================

    # Progress bar for AI summaries (top-k only)
    if ai_summary_on:
        ai_bar = st.progress(0, text=f"AI Summaries: 0/{topk}")

    # UI cards: only top-k
    rows_display = []
    for idx, r in enumerate(results[:topk], start=1):
        with st.container(border=True):
            header, donut = st.columns([1.6, 0.4])
            with header:
                score_pct = r["cosine_max"] * 100.0
                st.markdown(f"### {idx}. {r['id']} — **{score_pct:.1f}% Match**")

                # Always show original uploaded filename (small/muted)
                file_label = resumes[r["raw_index"]]["filename"]
                st.caption(f"File: {file_label}")

                fit = build_fit_card(jd_text, {
                    "id": r["id"],
                    "cosine_max": r["cosine_max"],
                    "coverage": r["coverage"],
                    "support_snippets": r["support_snippets"],
                    "top2_cosine": r.get("cosine_top2", 0.0),
                })
                verdict_color = {"Strong Fit": BRAND["success"], "Partial Fit": BRAND["accent"], "Weak Fit": BRAND["danger"]}[fit["verdict"]]
                st.markdown(
                    f"<span style='background-color:{verdict_color}; color:white; padding:4px 8px; border-radius:999px;'>"
                    f"{fit['verdict']} • Confidence: {fit['confidence']}</span>",
                    unsafe_allow_html=True
                )

                st.write(f"**Match Score:** {score_pct:.1f}%  •  **Coverage:** {r['coverage']} snippet(s)")
                if not cosine_only:
                    st.write(f"**Hybrid Score:** {r['final_score']:.3f}  (α={alpha:.2f}{', BM25 on' if use_bm25 else ', BM25 off'})")

                if ai_summary_on:
                    # update progress bar before the call
                    ai_bar.progress(min(idx / topk, 1.0), text=f"AI Summaries: {idx}/{topk}")

                    # Use the original resume index attached by the ranker
                    ai_text = generate_ai_fit_summary(
                        verdict=fit["verdict"],
                        jd_text=jd_text,
                        resume_text=resumes[r["raw_index"]]["text"],
                        matched_skills=fit["matched"],
                        gaps=fit["gaps"],
                        out_tok=220
                    )
                    st.markdown("**Why this fit:** " + ai_text)

                    # Write AI summary back into rows_all for this candidate
                    for row in rows_all:
                        if row["raw_index"] == r.get("raw_index"):
                            row["ai_summary"] = ai_text
                            break
                else:
                    st.caption("Minimal view (AI off)")


            with donut:
                donut_plotly(score_pct, key=f"donut-{idx}")  # unique key per chart

        rows_display.append({
            "rank": idx,
            "id": r["id"],
            "match_score_cosine": r["cosine_max"],
            "coverage_snippets": r["coverage"],
            "hybrid_score": r["final_score"],
            "verdict": fit["verdict"],
            "confidence": fit["confidence"],
            "recommendation": fit["recommendation"],
            "matched_skills": "; ".join(m["jd_skill"] for m in fit["matched"]),
            "gaps": "; ".join(fit["gaps"]),
        })

    if ai_summary_on:
        ai_bar.progress(1.0, text=f"AI Summaries: {min(topk, len(results))}/{topk}")

    # Download CSV for ALL rows (not just top-k)
    if dl_ready and rows_all:
        df = pd.DataFrame(rows_all)
        if "raw_index" in df.columns:
            df = df.drop(columns=["raw_index"])
        st.download_button(
            "Download results (.csv)",
            df.to_csv(index=False).encode("utf-8"),
            "ranked_candidates_v8_1.csv",
            mime="text/csv"
        )

elif run and (not jd_text or not files):
    st.error("Please paste/upload a JD and upload at least one resume, then click **Rank Candidates**.")
else:
    st.info("Paste/upload a JD and upload resumes, then click **Rank Candidates**.")
