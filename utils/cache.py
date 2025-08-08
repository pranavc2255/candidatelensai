# tiny file cache. i just use pickle on disk. simple but does the job.
import os, hashlib, pickle

class DiskCache:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _path(self, ns: str, key: str) -> str:
        safe = "".join(c for c in key if c.isalnum())
        return os.path.join(self.root, f"{ns}_{safe}.pkl")

    def get(self, ns: str, key: str):
        try:
            with open(self._path(ns, key), "rb") as f:
                return pickle.load(f)
        except Exception:
            return None  # cache miss or read error, just ignore

    def set(self, ns: str, key: str, value):
        try:
            with open(self._path(ns, key), "wb") as f:
                pickle.dump(value, f)
        except Exception:
            pass  # don't crash if cache can't be written

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
