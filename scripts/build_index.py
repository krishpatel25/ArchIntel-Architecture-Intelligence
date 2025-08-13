# scripts/build_index.py
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXTRACT_DIR = DATA_DIR / "extract"
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

class ArchIntelIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the indexer with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.chunks = []
        self.metadata = []
        
    def load_extracted_papers(self) -> List[Dict[str, Any]]:
        """Load all extracted JSON files from data/extract/."""
        papers = []
        for json_file in EXTRACT_DIR.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                papers.append(json.load(f))
        return papers
    
    def chunk_paper(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a paper into chunks and return with metadata."""
        paper_chunks = []
        full_text = paper.get("full_text", "")
        
        if not full_text.strip():
            return paper_chunks
            
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(full_text)
        
        for i, chunk in enumerate(text_chunks):
            chunk_data = {
                "text": chunk,
                "paper_title": paper.get("filename", "Unknown"),
                "paper_id": paper.get("filename", "").replace(".pdf", ""),
                "chunk_id": i,
                "total_chunks": len(text_chunks),
                "num_pages": paper.get("num_pages", 0)
            }
            paper_chunks.append(chunk_data)
            
        return paper_chunks
    
    def build_index(self):
        """Build the FAISS index from all extracted papers."""
        print("[INFO] Loading extracted papers...")
        papers = self.load_extracted_papers()
        
        if not papers:
            print("[ERROR] No extracted papers found. Run parse_pdf.py first.")
            return
            
        print(f"[INFO] Found {len(papers)} papers to index.")
        
        # Process each paper
        for paper in tqdm(papers, desc="Processing papers"):
            chunks = self.chunk_paper(paper)
            self.chunks.extend(chunks)
            self.metadata.extend([chunk for chunk in chunks])
        
        if not self.chunks:
            print("[ERROR] No text chunks found.")
            return
            
        print(f"[INFO] Created {len(self.chunks)} text chunks.")
        
        # Generate embeddings
        print("[INFO] Generating embeddings...")
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        print("[INFO] Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        # Save index and metadata
        print("[INFO] Saving index and metadata...")
        faiss.write_index(index, str(INDEX_DIR / "papers.index"))
        
        with open(INDEX_DIR / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
            
        # Save some statistics
        stats = {
            "num_papers": len(papers),
            "num_chunks": len(self.chunks),
            "embedding_dimension": dimension,
            "model_name": self.model.get_sentence_embedding_dimension()
        }
        
        with open(INDEX_DIR / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"[OK] Index built successfully!")
        print(f"  - Papers: {stats['num_papers']}")
        print(f"  - Chunks: {stats['num_chunks']}")
        print(f"  - Dimension: {stats['embedding_dimension']}")

def main():
    indexer = ArchIntelIndexer()
    indexer.build_index()

if __name__ == "__main__":
    main()
