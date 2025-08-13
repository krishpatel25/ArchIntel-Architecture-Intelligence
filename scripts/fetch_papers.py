# scripts/fetch_papers.py
import os
import re
import sys
import time
import yaml
import hashlib
import requests
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PAPERS_DIR = DATA_DIR / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

def arxiv_pdf_url(url_or_id: str) -> str:
    """Return a direct arXiv PDF URL from an arXiv abs or id if possible; else return original."""
    s = url_or_id.strip()
    # if it's already a PDF, return as is
    if s.endswith(".pdf"):
        return s
    # arXiv abs -> pdf
    m = re.match(r"https?://arxiv\.org/abs/([\w\.\-]+)", s)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
    # bare arXiv id
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", s):
        return f"https://arxiv.org/pdf/{s}.pdf"
    # otherwise, return original; might still be a direct PDF elsewhere
    return s

def safe_slug(s: str) -> str:
    # make a stable filename from URL/id
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
    parsed = urlparse(s)
    base = os.path.basename(parsed.path) or "paper"
    base = base.replace(".pdf","")
    return f"{base}_{h}"

def download_pdf(url: str, out_path: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(out_path, "wb") as f, tqdm(
                total=total if total else None,
                unit="B", unit_scale=True, desc=out_path.name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        if total: pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return False

def main(seeds_file: str = "seeds/papers.yaml"):
    with open(seeds_file, "r") as f:
        cfg = yaml.safe_load(f)
    papers = cfg.get("papers", [])
    if not papers:
        print("No papers in seeds file."); sys.exit(1)

    print(f"[INFO] Found {len(papers)} seed entries.")
    for entry in papers:
        url = arxiv_pdf_url(entry)
        slug = safe_slug(url)
        out_pdf = PAPERS_DIR / f"{slug}.pdf"
        if out_pdf.exists():
            print(f"[SKIP] {out_pdf.name} already exists.")
            continue
        ok = download_pdf(url, out_pdf)
        time.sleep(0.5)  # be polite
        if ok:
            print(f"[OK] Saved to {out_pdf}")
        else:
            print(f"[FAIL] Could not download {url}")

if __name__ == "__main__":
    seeds = sys.argv[1] if len(sys.argv) > 1 else "seeds/papers.yaml"
    main(seeds)