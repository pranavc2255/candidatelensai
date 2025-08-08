import fitz  # PyMuPDF

# pdf reader using pymupdf. works for native text pdfs (not doing ocr here).
def extract_text_from_pdf(path: str) -> str:
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text("text"))
    return "\n".join(text)

# switcher that decides pdf vs plain txt. keep it tiny.
def extract_text_from_file(path: str) -> str:
    lp = path.lower()
    if lp.endswith(".pdf"):
        return extract_text_from_pdf(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
