# picks a device based on user pref and whether cuda actually exists
def choose_device(pref: str = "cpu") -> str:
    try:
        import torch  # sentence-transformers brings torch
        has_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        has_cuda = False

    if pref == "auto":
        return "cuda" if has_cuda else "cpu"
    if pref == "cuda":
        return "cuda" if has_cuda else "cpu"
    return "cpu"
