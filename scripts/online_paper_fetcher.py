#!/usr/bin/env python3
"""
Online Paper Fetcher for ArchIntel
Fetches papers from Google Scholar, arXiv, and other online sources in real-time.
"""

import requests
import json
import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse, quote
from typing import List, Dict, Any, Optional
import re
from tqdm import tqdm

class OnlinePaperFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arXiv for papers."""
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:"{query}"',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            response = self.session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response (simplified)
            papers = []
            content = response.text
            
            # Extract paper IDs and titles
            paper_ids = re.findall(r'<id>http://arxiv.org/abs/([^<]+)</id>', content)
            titles = re.findall(r'<title>([^<]+)</title>', content)
            summaries = re.findall(r'<summary>([^<]+)</summary>', content)
            
            for i, paper_id in enumerate(paper_ids[:max_results]):
                if i < len(titles) and i < len(summaries):
                    papers.append({
                        'id': paper_id,
                        'title': titles[i].strip(),
                        'summary': summaries[i].strip(),
                        'pdf_url': f'https://arxiv.org/pdf/{paper_id}.pdf',
                        'source': 'arxiv'
                    })
            
            return papers
            
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []
    
    def search_semantic_scholar(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for papers."""
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': max_results,
            'fields': 'title,abstract,url,year,authors.name'
        }
        
        try:
            response = self.session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for paper in data.get('data', []):
                papers.append({
                    'id': paper.get('paperId', ''),
                    'title': paper.get('title', ''),
                    'summary': paper.get('abstract', ''),
                    'pdf_url': paper.get('url', ''),
                    'year': paper.get('year', ''),
                    'authors': [author.get('name', '') for author in paper.get('authors', [])],
                    'source': 'semantic_scholar'
                })
            
            return papers
            
        except Exception as e:
            print(f"Error searching Semantic Scholar: {e}")
            return []
    
    def search_google_scholar(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Google Scholar for papers (simplified)."""
        # Note: Google Scholar doesn't have a public API, so this is a simplified approach
        # In practice, you'd need to use a service like SerpAPI or similar
        
        base_url = "https://scholar.google.com/scholar"
        params = {
            'q': query,
            'hl': 'en',
            'num': max_results
        }
        
        try:
            response = self.session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # This is a simplified parser - in practice you'd need more sophisticated parsing
            papers = []
            content = response.text
            
            # Extract titles and URLs (simplified)
            titles = re.findall(r'<h3[^>]*><a[^>]*>([^<]+)</a></h3>', content)
            urls = re.findall(r'<h3[^>]*><a[^>]*href="([^"]+)"[^>]*>', content)
            
            for i, title in enumerate(titles[:max_results]):
                papers.append({
                    'id': f'gs_{hashlib.md5(title.encode()).hexdigest()[:8]}',
                    'title': title.strip(),
                    'summary': '',
                    'pdf_url': urls[i] if i < len(urls) else '',
                    'source': 'google_scholar'
                })
            
            return papers
            
        except Exception as e:
            print(f"Error searching Google Scholar: {e}")
            return []
    
    def fetch_paper_text(self, paper: Dict[str, Any]) -> Optional[str]:
        """Fetch the full text of a paper if available."""
        if not paper.get('pdf_url'):
            return None
            
        try:
            response = self.session.get(paper['pdf_url'], timeout=30)
            response.raise_for_status()
            
            # For now, return the abstract/summary
            # In a full implementation, you'd use PDF parsing libraries
            return paper.get('summary', '')
            
        except Exception as e:
            print(f"Error fetching paper text: {e}")
            return None
    
    def search_all_sources(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search all available sources for papers."""
        all_papers = []
        
        print(f"Searching for: {query}")
        
        # Search arXiv
        print("Searching arXiv...")
        arxiv_papers = self.search_arxiv(query, max_results)
        all_papers.extend(arxiv_papers)
        
        # Search Semantic Scholar
        print("Searching Semantic Scholar...")
        ss_papers = self.search_semantic_scholar(query, max_results)
        all_papers.extend(ss_papers)
        
        # Remove duplicates based on title similarity
        unique_papers = self.remove_duplicates(all_papers)
        
        return unique_papers[:max_results]
    
    def remove_duplicates(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title similarity."""
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            title_lower = paper['title'].lower()
            # Simple duplicate detection - in practice you'd use more sophisticated methods
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(paper)
        
        return unique_papers

class OnlineArchIntel:
    def __init__(self):
        self.fetcher = OnlinePaperFetcher()
        self.cache_dir = Path("data/online_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for papers online and return results with text."""
        # Check cache first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if time.time() - cached_data.get('timestamp', 0) < 3600:  # 1 hour cache
                    return cached_data.get('papers', [])
        
        # Search online
        papers = self.fetcher.search_all_sources(query, max_results)
        
        # Fetch text for each paper
        for paper in papers:
            paper['text'] = self.fetcher.fetch_paper_text(paper)
        
        # Cache results
        cache_data = {
            'timestamp': time.time(),
            'papers': papers
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        return papers
    
    def get_paper_chunks(self, papers: List[Dict[str, Any]], chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """Split papers into chunks for indexing."""
        chunks = []
        
        for paper in papers:
            text = paper.get('text', paper.get('summary', ''))
            if not text:
                continue
            
            # Simple chunking - in practice you'd use LangChain's text splitter
            words = text.split()
            for i in range(0, len(words), chunk_size // 5):  # Approximate word count
                chunk_text = ' '.join(words[i:i + chunk_size // 5])
                if len(chunk_text) > 100:  # Minimum chunk size
                    chunks.append({
                        'text': chunk_text,
                        'paper_title': paper['title'],
                        'paper_id': paper['id'],
                        'source': paper['source'],
                        'chunk_id': len(chunks),
                        'score': 0.0  # Will be set during search
                    })
        
        return chunks

def main():
    """Test the online paper fetcher."""
    archintel = OnlineArchIntel()
    
    # Test queries
    test_queries = [
        "cache optimization techniques computer architecture",
        "neural network accelerator hardware",
        "memory hierarchy optimization",
        "RISC-V vector extensions",
        "quantum computing architecture"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Searching for: {query}")
        print(f"{'='*60}")
        
        papers = archintel.search_papers(query, max_results=5)
        
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Source: {paper['source']}")
            print(f"   URL: {paper.get('pdf_url', 'N/A')}")
            if paper.get('summary'):
                print(f"   Summary: {paper['summary'][:200]}...")
            print()

if __name__ == "__main__":
    main()
