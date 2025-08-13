# app.py - Streamlit web interface for ArchIntel
import streamlit as st
import os
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.append(str(scripts_dir))

from query_index import ArchIntelRetriever, ArchIntelLLM
from scripts.online_paper_fetcher import OnlineArchIntel

# Page configuration
st.set_page_config(
    page_title="ArchIntel - Architecture Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #1f77b4;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .paper-title {
        font-weight: bold;
        color: #1f77b4;
    }
    .score {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_retriever():
    """Load the retriever with caching."""
    try:
        return ArchIntelRetriever()
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        st.info("Please run the indexing scripts first:")
        st.code("python scripts/build_index.py")
        return None

@st.cache_resource
def load_online_fetcher():
    """Load the online paper fetcher with caching."""
    return OnlineArchIntel()

@st.cache_resource
def load_llm():
    """Load LLM with caching."""
    try:
        if os.getenv("OPENAI_API_KEY"):
            return ArchIntelLLM("openai")
        elif os.getenv("ANTHROPIC_API_KEY"):
            return ArchIntelLLM("anthropic")
        else:
            return None
    except Exception as e:
        st.warning(f"LLM not available: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏗️ ArchIntel</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Research Assistant for Computer Architecture & AI Hardware</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # LLM provider selection
        llm_provider = st.selectbox(
            "LLM Provider (for synthesis)",
            ["None", "OpenAI", "Anthropic"],
            help="Select LLM provider for answer synthesis. Requires API key."
        )
        
        # Number of results
        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of top results to retrieve"
        )
        
        # Show raw results toggle
        show_raw = st.checkbox(
            "Show raw search results",
            value=True,
            help="Display the raw text chunks from papers"
        )
        
        # API key setup
        st.header("API Setup")
        if llm_provider == "OpenAI":
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key for answer synthesis"
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                
        elif llm_provider == "Anthropic":
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                help="Enter your Anthropic API key for answer synthesis"
            )
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
        
        # System info
        st.header("System Info")
        retriever = load_retriever()
        if retriever:
            st.success("✅ Index loaded")
            st.info(f"Papers indexed: {len(retriever.metadata)} chunks")
        else:
            st.error("❌ Index not found")
            
        llm = load_llm()
        if llm:
            st.success("✅ LLM available")
        else:
            st.warning("⚠️ LLM not available")
    
    # Main content
    online_fetcher = load_online_fetcher()
    
    # Search mode selection
    search_mode = st.radio(
        "Search Mode",
        ["Local Papers", "Online Papers", "Both"],
        horizontal=True,
        help="Choose whether to search local papers, online sources, or both"
    )
    
    if search_mode == "Local Papers" and retriever is None:
        st.error("Please set up the local index first by running the indexing scripts.")
        st.code("""
# First, fetch papers
python scripts/fetch_papers.py

# Then parse PDFs
python scripts/parse_pdf.py

# Finally, build the index
python scripts/build_index.py
        """)
        return
    
    # Query input
    st.header("🔍 Query Your Research Papers")
    
    # Quick search queries
    with st.expander("🔍 Quick Search Queries"):
        quick_queries = [
            "cache optimization techniques computer architecture",
            "neural network accelerator hardware design",
            "memory hierarchy optimization strategies",
            "RISC-V vector extensions implementation",
            "quantum computing architecture principles",
            "GPU memory management techniques",
            "AI hardware security vulnerabilities",
            "energy-efficient processor design",
            "parallel computing architectures",
            "emerging memory technologies"
        ]
        
        cols = st.columns(2)
        for i, query in enumerate(quick_queries):
            col_idx = i % 2
            if cols[col_idx].button(f"🔍 {query[:30]}...", key=f"quick_{i}"):
                st.session_state.query = query
    
    # Query input
    query = st.text_area(
        "Enter your query:",
        value=st.session_state.get("query", ""),
        height=100,
        placeholder="e.g., 'Summarize cache optimization techniques from these papers'"
    )
    
    # Search button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("🔍 Search Papers", type="primary", use_container_width=True)
    
    if search_button and query.strip():
        st.session_state.query = query
        
        with st.spinner("Searching papers..."):
            all_results = []
            
            # Search local papers if available
            if search_mode in ["Local Papers", "Both"] and retriever:
                local_results = retriever.search(query.strip(), top_k=top_k)
                all_results.extend(local_results)
            
            # Search online papers
            if search_mode in ["Online Papers", "Both"]:
                with st.spinner("Searching online sources..."):
                    online_papers = online_fetcher.search_papers(query.strip(), max_results=top_k)
                    online_chunks = online_fetcher.get_paper_chunks(online_papers)
                    
                    # Convert to same format as local results
                    for i, chunk in enumerate(online_chunks):
                        chunk['rank'] = i + 1
                        chunk['total_chunks'] = len(online_chunks)
                        chunk['num_pages'] = 1  # Online papers don't have page info
                        all_results.append(chunk)
            
            if not all_results:
                st.warning("No relevant results found. Try a different query.")
                return
            
            # Sort by score and take top results
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            results = all_results[:top_k]
            
            # Display results
            st.header("📄 Search Results")
            
            # Synthesized answer
            if llm_provider != "None" and llm:
                with st.spinner("Generating synthesized answer..."):
                    answer = llm.synthesize_answer(query, results)
                    
                st.subheader("🤖 Synthesized Answer")
                st.markdown(answer)
                st.divider()
            
            # Raw results
            if show_raw:
                st.subheader("📋 Raw Search Results")
                
                for i, result in enumerate(results):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            paper_title = result['paper_title']
                            score = result.get('score', 0)
                            
                            # Check if it's an online paper
                            is_online = 'source' in result
                            
                            if is_online:
                                # Online paper display
                                st.markdown(f"""
                                <div class="result-box">
                                    <div class="paper-title">
                                        🌐 <strong>{paper_title}</strong>
                                        <span class="score" style="float: right; color: #28a745; font-weight: bold;">
                                            Score: {score:.3f}
                                        </span>
                                    </div>
                                    <div style="color: #666; font-size: 0.8rem;">
                                        Source: {result.get('source', 'Unknown')} | ID: {result.get('paper_id', 'N/A')}
                                    </div>
                                    <br>
                                    <strong>Text:</strong><br>
                                    {result['text'][:300]}{'...' if len(result['text']) > 300 else ''}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Add link to paper
                                if result.get('pdf_url'):
                                    st.link_button(f"🔗 View Paper", result['pdf_url'], key=f"link_{i}")
                            else:
                                # Local paper display
                                pdf_filename = paper_title
                                pdf_path = f"data/papers/{pdf_filename}"
                                
                                st.markdown(f"""
                                <div class="result-box">
                                    <div class="paper-title">
                                        📄 <strong>{pdf_filename}</strong>
                                        <span class="score" style="float: right; color: #28a745; font-weight: bold;">
                                            Score: {score:.3f}
                                        </span>
                                    </div>
                                    <br>
                                    <strong>Text:</strong><br>
                                    {result['text'][:300]}{'...' if len(result['text']) > 300 else ''}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Add a button to open PDF
                                if st.button(f"📖 Open PDF: {pdf_filename}", key=f"pdf_{i}"):
                                    try:
                                        with open(pdf_path, "rb") as pdf_file:
                                            st.download_button(
                                                label="📥 Download PDF",
                                                data=pdf_file.read(),
                                                file_name=pdf_filename,
                                                mime="application/pdf",
                                                key=f"download_{i}"
                                            )
                                    except FileNotFoundError:
                                        st.error(f"PDF file not found: {pdf_filename}")
                        
                        with col2:
                            st.metric("Rank", result.get('rank', i+1))
                            if not is_online:
                                st.metric("Chunk", f"{result.get('chunk_id', 0)}/{result.get('total_chunks', 1)}")
                                st.metric("Pages", result.get('num_pages', 1))
                            else:
                                st.metric("Source", result.get('source', 'Unknown'))
                                st.metric("Type", "Online")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>ArchIntel - Architecture Intelligence Research Assistant</p>
        <p>Built with Streamlit, FAISS, and Sentence Transformers</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
