# scripts/parse_pdf.py
import json
from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PAPERS_DIR = DATA_DIR / "papers"
EXTRACT_DIR = DATA_DIR / "extract"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        pages.append({"page": i+1, "text": text})
    full_text = "\n".join(p["text"] for p in pages)
    return {"filename": pdf_path.name, "num_pages": len(doc), "pages": pages, "full_text": full_text}

def main():
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in data/papers."); return
    for pdf in tqdm(pdfs, desc="Parsing PDFs"):
        out_json = EXTRACT_DIR / (pdf.stem + ".json")
        out_txt  = EXTRACT_DIR / (pdf.stem + ".txt")
        if out_json.exists() and out_txt.exists():
            continue
        data = extract_text(pdf)
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        out_txt.write_text(data["full_text"])
    print("[OK] Extracted text to data/extract/")

if __name__ == "__main__":
    main()