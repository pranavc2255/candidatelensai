import re

# quick and dirty way i pull a name from the top of a resume text.
# not perfect, but usually the first non-empty line with 2-4 words looks like a name.
def extract_candidate_name(text: str, fallback_id: str = "Candidate") -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines[:20]:
        if 2 <= len(l.split()) <= 4 and re.search(r'[A-Za-z]', l):
            if len(l) <= 60:
                return l  # if this looks reasonable, just use it
    return fallback_id  # if nothing matched, use the file id as a backup
