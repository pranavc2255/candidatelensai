import re
from collections import Counter

# a small alias map so common synonyms count as the same thing.
# definitely not exhaustive, just the ones i kept seeing.
ALIASES = {
    "pytorch": ["torch", "py torch"],
    "tensorflow": ["tf"],
    "nlp": ["natural language processing"],
    "transformers": ["bert", "roberta", "gpt"],
    "semantic search": ["vector search", "retrieval", "faiss"],
    "faiss": ["vector db", "ann index"],
    "aws": ["amazon web services"],
    "gcp": ["google cloud"],
    "kubernetes": ["k8s"],
    "javascript": ["js"],
    "typescript": ["ts"],
    "react": ["react.js", "reactjs"],
    "next.js": ["nextjs", "next"],
    "tableau": [],
    "power bi": ["powerbi"],
    "looker": [],
    "dbt": [],
    "grpc": ["g rpc"],
    "postgresql": ["postgres", "psql"],
    "redis": [],
    "terraform": [],
    "prometheus": ["prom"],
    "grafana": [],
    "airflow": [],
    "fastapi": [],
    "huggingface": ["hugging face"],
    "opencv": [],
    "docker": [],
    "ci/cd": ["cicd", "continuous integration", "continuous delivery"],
    "etl": ["elt"],
    "ab testing": ["a/b testing", "experimentation"],
    "accessibility": ["a11y"],
}

# super tiny stopword list, just to avoid counting fluff when guessing skills
STOP = set("""a an the for to and or of in with on at by from into over after before about than as is are be been being this that those these your our their we you i""".split())

def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s.lower()).strip()

# pull a small set of "skills" like terms from the JD. not ML, just counts.
def extract_skills_from_jd(jd_text: str, top_k: int = 15):
    text = normalize(jd_text)
    tokens = re.findall(r'[a-z0-9\+\#\.\-]{2,}', text)
    tok_counts = Counter(t for t in tokens if t not in STOP and not t.isdigit())
    for k in ALIASES.keys():
        if k in text:
            tok_counts[k] += 5  # tiny boost if alias appears in JD
    skills = []
    seen = set()
    for term, _ in tok_counts.most_common(80):
        if term in seen or len(term) <= 2 or term in STOP: 
            continue
        skills.append(term); seen.add(term)
        if len(skills) >= top_k: break
    curated, seen2 = [], set()
    for s in skills:
        if s in ALIASES or re.search(r'[a-z]+\d+|[+#.]|sql|aws|gcp|azure|js|ts|ml|cv|nlp|db|api', s):
            if s not in seen2:
                curated.append(s); seen2.add(s)
    return curated[:top_k]

def sentence_split(text: str):
    parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

# try to map JD skills to the strongest resume snippets, also collect gaps
def find_matches(skills, snippets):
    found, missing = [], []
    joined = " ".join(sn["snippet"] for sn in snippets).lower()
    for sk in skills:
        alts = [sk] + ALIASES.get(sk, [])
        hit = None
        for alt in alts:
            if alt and alt.lower() in joined:
                for sn in snippets:
                    sent_hits = [s for s in sentence_split(sn["snippet"]) if alt.lower() in s.lower()]
                    if sent_hits:
                        hit = (sk, sn, sent_hits[0][:220]); break
                if hit: break
        if hit:
            found.append({"jd_skill": sk, "snippet_id": snippets.index(hit[1])+1, "quote": hit[2]})
        else:
            missing.append(sk)
    return found[:5], missing[:3]

# some simple rules for verdict/confidence. this is meant to be readable.
def verdict_from_scores(cosine_max: float, coverage: int):
    if cosine_max >= 0.72 and coverage >= 2:
        return "Strong Fit"
    if cosine_max >= 0.58:
        return "Partial Fit"
    return "Weak Fit"

def confidence(cosine_max: float, coverage: int, margin: float):
    if cosine_max >= 0.78 and coverage >= 2 and margin >= 0.05:
        return "High"
    if cosine_max >= 0.62 and coverage >= 1:
        return "Medium"
    return "Low"

def recommendation(verdict: str):
    if verdict == "Strong Fit":
        return "Invite to interview"
    if verdict == "Partial Fit":
        return "Consider phone screen"
    return "Skip for this role"

# assemble the card-ish dict that the UI uses
def build_fit_card(jd_text: str, candidate: dict):
    skills = extract_skills_from_jd(jd_text)
    matched, gaps = find_matches(skills, candidate.get("support_snippets", []))
    cosine_max = float(candidate.get("cosine_max", 0.0))
    coverage = int(candidate.get("coverage", 0))
    margin = max(0.0, cosine_max - float(candidate.get("top2_cosine", 0.0)))
    verdict = verdict_from_scores(cosine_max, coverage)
    conf = confidence(cosine_max, coverage, margin)
    rec = recommendation(verdict)
    return {
        "verdict": verdict,
        "confidence": conf,
        "recommendation": rec,
        "similarity": cosine_max,
        "coverage": coverage,
        "matched": matched,
        "gaps": gaps,
    }
