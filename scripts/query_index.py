# scripts/query_index.py
import json
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INDEX_DIR = DATA_DIR / "index"

class ArchIntelRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the retriever with the same model used for indexing."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = None
        self.load_index()
        
    def load_index(self):
        """Load the FAISS index and metadata."""
        index_path = INDEX_DIR / "papers.index"
        metadata_path = INDEX_DIR / "metadata.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Index not found. Run build_index.py first.")
            
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
        print(f"[INFO] Loaded index with {len(self.metadata)} chunks")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks given a query."""
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return results with metadata
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(score)
                result['rank'] = i + 1
                results.append(result)
                
        return results
    
    def format_results(self, results: List[Dict[str, Any]], include_text: bool = True) -> str:
        """Format search results for display."""
        output = []
        for result in results:
            paper_title = result['paper_title']
            score = result['score']
            rank = result['rank']
            
            line = f"[{rank}] {paper_title} (score: {score:.3f})"
            output.append(line)
            
            if include_text:
                # Truncate text for display
                text = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
                output.append(f"    {text}")
                output.append("")
                
        return "\n".join(output)

class ArchIntelLLM:
    def __init__(self, provider: str = "openai"):
        """Initialize LLM for synthesis."""
        self.provider = provider
        self.client = None
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package not installed")
                
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package not installed")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def synthesize_answer(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Synthesize an answer from search results using LLM."""
        if not self.client:
            return "LLM not available. Showing raw results only."
            
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(results[:5]):  # Use top 5 results
            paper_title = result['paper_title']
            text = result['text']
            context_parts.append(f"Source {i+1} ({paper_title}):\n{text}\n")
        
        context = "\n".join(context_parts)
        
        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": """You are an expert research assistant specializing in computer architecture and AI hardware. 
                        Answer questions based on the provided research papers. Always cite the specific papers you reference using the format "Source X (Paper Title)".
                        Be concise but thorough, and focus on the most relevant information from the papers."""},
                        {"role": "user", "content": f"Question: {query}\n\nContext from research papers:\n{context}\n\nPlease provide a comprehensive answer with citations:"}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error calling OpenAI API: {e}"
                
        elif self.provider == "anthropic":
            try:
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    temperature=0.3,
                    system="You are an expert research assistant specializing in computer architecture and AI hardware. Answer questions based on the provided research papers. Always cite the specific papers you reference using the format 'Source X (Paper Title)'. Be concise but thorough, and focus on the most relevant information from the papers.",
                    messages=[
                        {"role": "user", "content": f"Question: {query}\n\nContext from research papers:\n{context}\n\nPlease provide a comprehensive answer with citations:"}
                    ]
                )
                return response.content[0].text
            except Exception as e:
                return f"Error calling Anthropic API: {e}"

def main():
    """Interactive query interface."""
    try:
        retriever = ArchIntelRetriever()
        llm = None
        
        # Try to initialize LLM if API keys are available
        if os.getenv("OPENAI_API_KEY"):
            try:
                llm = ArchIntelLLM("openai")
                print("[INFO] OpenAI LLM available for synthesis")
            except:
                pass
        elif os.getenv("ANTHROPIC_API_KEY"):
            try:
                llm = ArchIntelLLM("anthropic")
                print("[INFO] Anthropic LLM available for synthesis")
            except:
                pass
                
        if not llm:
            print("[INFO] No LLM available. Will show raw search results only.")
            print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable for synthesis.")
        
        print("\n" + "="*60)
        print("ArchIntel - Architecture Intelligence Query Interface")
        print("="*60)
        print("Type your query about computer architecture/AI hardware papers.")
        print("Type 'quit' to exit.")
        print("-"*60)
        
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query:
                continue
                
            print(f"\nSearching for: {query}")
            print("-"*40)
            
            # Search
            results = retriever.search(query, top_k=5)
            
            if not results:
                print("No relevant results found.")
                continue
                
            # Show raw results
            print("Top search results:")
            print(retriever.format_results(results, include_text=True))
            
            # Synthesize answer if LLM available
            if llm:
                print("\n" + "="*40)
                print("Synthesized Answer:")
                print("="*40)
                answer = llm.synthesize_answer(query, results)
                print(answer)
            
            print("\n" + "-"*60)
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
