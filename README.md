# 🏗️ ArchIntel - Architecture Intelligence

**AI-Powered Research Assistant for Computer Architecture & AI Hardware Papers**

ArchIntel is a sophisticated research assistant that combines local paper indexing with real-time online search capabilities. It uses retrieval-augmented generation (RAG) to help you search, summarize, and compare research papers in computer architecture and AI hardware.

## 🌟 Key Features

### 🔍 **Multi-Source Search**
- **Local Papers**: Search through your downloaded PDF collection
- **Online Papers**: Real-time search from arXiv, Semantic Scholar, and Google Scholar
- **Hybrid Mode**: Combine local and online results for comprehensive research

### 🤖 **AI-Powered Analysis**
- **RAG Integration**: Retrieval-augmented generation for intelligent answers
- **LLM Synthesis**: OpenAI GPT-3.5/4 and Anthropic Claude support
- **Cited Answers**: Get coherent responses with proper paper citations

### 📊 **Smart Indexing**
- **FAISS Vector Store**: Fast similarity search using embeddings
- **Text Chunking**: Intelligent document segmentation for better retrieval
- **Metadata Tracking**: Paper titles, sources, scores, and chunk information

### 🎯 **User-Friendly Interface**
- **Streamlit Web App**: Beautiful, responsive web interface
- **CLI Interface**: Command-line tool for quick queries
- **Real-time Results**: Instant search across multiple sources

## 🚀 Quick Start

### 1. **Setup Environment**

```bash
# Clone the repository
git clone <repository-url>
cd ArchIntel-Architecture-Intelligence

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Configure API Keys (Optional)**

Create a `.env` file for LLM synthesis:

```bash
# For OpenAI GPT-4/3.5 synthesis
OPENAI_API_KEY=your-openai-api-key-here

# For Anthropic Claude synthesis
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

### 3. **Launch the App**

```bash
# Start the web interface
streamlit run app.py

# Or use CLI interface
python3 scripts/query_index.py
```

## 📁 Project Structure

```
ArchIntel-Architecture-Intelligence/
├── app.py                    # Main Streamlit web application
├── scripts/
│   ├── fetch_papers.py       # Download papers from URLs
│   ├── parse_pdf.py          # Extract text from PDFs
│   ├── build_index.py        # Create embeddings and FAISS index
│   ├── query_index.py        # CLI query interface
│   └── online_paper_fetcher.py # Real-time online paper search
├── data/
│   ├── papers/               # Downloaded PDF files
│   ├── extract/              # Extracted text (JSON + TXT)
│   ├── index/                # FAISS index and metadata
│   └── online_cache/         # Cached online search results
├── seeds/
│   └── papers.yaml           # Paper URLs for local collection
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔧 Usage Guide

### **Web Interface**

1. **Open the App**: Navigate to `http://localhost:8501`
2. **Choose Search Mode**:
   - **Local Papers**: Search your downloaded collection
   - **Online Papers**: Search live from academic sources
   - **Both**: Combine local and online results
3. **Enter Queries**: Use natural language or click quick search buttons
4. **View Results**: Get synthesized answers and raw search results
5. **Access Papers**: Click links to view/download papers

### **Quick Search Queries**

The app includes 10 pre-defined search terms:
- Cache optimization techniques computer architecture
- Neural network accelerator hardware design
- Memory hierarchy optimization strategies
- RISC-V vector extensions implementation
- Quantum computing architecture principles
- GPU memory management techniques
- AI hardware security vulnerabilities
- Energy-efficient processor design
- Parallel computing architectures
- Emerging memory technologies

### **CLI Interface**

```bash
# Interactive query mode
python3 scripts/query_index.py

# Example queries:
# - "Summarize cache optimization techniques from these papers"
# - "Compare prefetching strategies discussed in the literature"
# - "What are the latest developments in in-memory computing?"
```

## 🛠️ Advanced Configuration

### **Adding Local Papers**

1. **Edit `seeds/papers.yaml`**:
```yaml
papers:
  - https://arxiv.org/pdf/2405.00458.pdf  # Memory-Centric Computing
  - https://arxiv.org/pdf/2401.10301.pdf  # RISC-V Vector Processor
  # Add more papers here...
```

2. **Run the pipeline**:
```bash
python3 run_pipeline.py
```

### **Customizing Search Sources**

Modify `scripts/online_paper_fetcher.py` to add new sources:
- arXiv API
- Semantic Scholar API
- Google Scholar (simplified)
- Custom academic repositories

### **Adjusting Index Parameters**

In `scripts/build_index.py`:
- **Chunk Size**: Default 1000 characters
- **Chunk Overlap**: Default 200 characters
- **Embedding Model**: Default "all-MiniLM-L6-v2"

## 📊 Performance

### **Indexing Performance**
- **Local Papers**: ~1-2 seconds per paper
- **Online Search**: ~2-5 seconds per query
- **Memory Usage**: ~50-100MB per 1000 chunks

### **Search Performance**
- **Query Response**: Sub-second retrieval
- **LLM Synthesis**: 5-15 seconds (depending on model)
- **Cache Hit Rate**: ~80% for repeated queries

## 🔍 Search Capabilities

### **Local Search**
- **Vector Similarity**: FAISS-based semantic search
- **Metadata Filtering**: Filter by paper title, date, etc.
- **Chunk-level Retrieval**: Precise text segment matching

### **Online Search**
- **Multi-source**: arXiv, Semantic Scholar, Google Scholar
- **Real-time**: Latest papers from academic repositories
- **Caching**: 1-hour cache to reduce API calls

### **Hybrid Search**
- **Combined Results**: Merge local and online findings
- **Score Normalization**: Unified ranking across sources
- **Deduplication**: Remove duplicate papers automatically

## 🤖 AI Integration

### **LLM Providers**
- **OpenAI**: GPT-3.5-turbo, GPT-4
- **Anthropic**: Claude-3-sonnet
- **Fallback**: Raw search results if no API key

### **Synthesis Features**
- **Context-aware**: Uses retrieved paper chunks as context
- **Citation-aware**: References specific papers in answers
- **Domain-specific**: Specialized for computer architecture

## 🐛 Troubleshooting

### **Common Issues**

**"Index not found"**
```bash
# Rebuild the index
python3 scripts/build_index.py
```

**"No papers found"**
```bash
# Download papers first
python3 scripts/fetch_papers.py
```

**"LLM not available"**
```bash
# Set API keys in .env file
echo "OPENAI_API_KEY=your-key" > .env
```

**"Online search failed"**
- Check internet connection
- Verify API rate limits
- Try different search terms

### **Debug Mode**

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Adding New Features

### **New Search Sources**
1. Add source class to `online_paper_fetcher.py`
2. Implement search method
3. Add to `search_all_sources()` method

### **New LLM Providers**
1. Add provider class to `query_index.py`
2. Implement synthesis method
3. Update provider selection logic

### **Custom Embeddings**
1. Change model in `build_index.py`
2. Update embedding dimension
3. Rebuild index

## 📈 Future Enhancements

### **Planned Features**
- **Citation Network**: Visualize paper relationships
- **Trend Analysis**: Track research trends over time
- **Collaborative Filtering**: Paper recommendations
- **Advanced Filtering**: Date, venue, author filters
- **Export Results**: Save searches and results

### **Performance Improvements**
- **Distributed Indexing**: Multi-node FAISS clusters
- **Streaming Search**: Real-time result updates
- **Advanced Caching**: Redis-based result caching
- **GPU Acceleration**: CUDA-enabled embeddings

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Add** tests if applicable
5. **Submit** a pull request

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black scripts/ app.py

# Type checking
mypy scripts/ app.py
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: RAG framework and text processing
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings
- **Streamlit**: Web application framework
- **arXiv API**: Academic paper access
- **Semantic Scholar API**: Research paper metadata

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: your-email@example.com

---

**Happy researching! 🎓**

*ArchIntel - Making computer architecture research smarter, faster, and more accessible.*