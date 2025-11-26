import os
import re
import time
import json
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

try:
    import chromadb
except Exception as e:
    chromadb = None

try:
    import google.generativeai as genai
except Exception as e:
    genai = None

try:
    from rp import (
        ResearchPaperSummarizer,
        SemanticScholarSearch,
        TopicExtractor,
        classify_research_paper
    )
except Exception as e:
    st.error(f"❌ Failed to import rp.py analyzer: {e}")
    st.stop()


st.set_page_config(
    page_title="AI Research Paper Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom Card Style */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 20px 0;
    }

    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 2px 2px 8px rgba(255,255,255,0.3), 0 0 20px rgba(255,255,255,0.2); }
        to { text-shadow: 2px 2px 12px rgba(255,255,255,0.5), 0 0 30px rgba(255,255,255,0.3); }
    }
    .sub-title { font-size: 1.3rem; color: #ffffff !important; text-align: center; margin-bottom: 30px; font-weight: 300; text-shadow: 1px 1px 4px rgba(0,0,0,0.2); }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3); }
    .metric-value { font-size: 2.5rem; font-weight: bold; margin: 10px 0; }
    .metric-label { font-size: 1rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; }

    .upload-zone {
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 60px;
        text-align: center;
        background: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
    }
    .upload-zone:hover { border-color: #764ba2; background: rgba(255, 255, 255, 1); transform: scale(1.02); }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6); }

    .stTabs [data-baseweb="tab-list"] { gap: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 15px; padding: 10px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 10px 20px; background: rgba(255, 255, 255, 0.2); color: white; font-weight: 600; }
    .stTabs [aria-selected="true"] { background: white; color: #667eea; }

    .stProgress > div > div > div > div { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }
    .stSuccess { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; border-radius: 10px; padding: 15px; }
    .stInfo { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; padding: 15px; }

    .dataframe { border-radius: 10px; overflow: hidden; }
    .element-container:empty { display: none; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .fade-in { animation: fadeIn 0.6s ease-out; }

    .badge { display: inline-block; padding: 8px 15px; border-radius: 20px; font-weight: 600; font-size: 0.9rem; margin: 5px; }
    .badge-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .badge-success { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; }
    .badge-info { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }

    [data-testid="stFileUploader"] { background: rgba(255, 255, 255, 0.1); border-radius: 15px; padding: 20px; }
    [data-testid="stFileUploader"] label { color: white !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'paper_text' not in st.session_state:
    st.session_state.paper_text = None
if 'similar_papers' not in st.session_state:
    st.session_state.similar_papers = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'vector_collections' not in st.session_state:
    st.session_state.vector_collections = {} 


embedder = None
chroma_client = None
if SentenceTransformer is not None:
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"⚠️ Embedding model failed to load: {e}")
        embedder = None
else:
    st.warning("⚠️ sentence-transformers not installed. Chat feature disabled.")

if chromadb is not None:
    try:
        chroma_client = chromadb.Client()
    except Exception as e:
        st.warning(f"⚠️ Chroma client init failed: {e}")
        chroma_client = None
else:
    st.warning("⚠️ chromadb not installed. Chat feature disabled.")


if genai is None:
    st.warning("⚠️ google-generativeai is not installed. Chat feature disabled.")
else:
    try:
        gemini_key = st.secrets["gemini"]["api_key"]
        if gemini_key:
            try:
                genai.configure(api_key=gemini_key)
            except Exception as e:
                st.warning(f"⚠️ Failed to configure Gemini with provided key: {e}")
        else:
            st.warning("⚠️ Gemini API key not found in secrets. Chat requires key in .streamlit/secrets.toml")
    except Exception:
        st.warning("⚠️ Gemini API key not found in secrets. Please add [gemini] api_key = '...' in .streamlit/secrets.toml")


with st.sidebar:
    st.markdown("<h2 style='color: white;'>⚙️ Settings</h2>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<h3 style='color: white; font-size: 1.2rem;'>🔍 Search Settings</h3>", unsafe_allow_html=True)
    max_papers = st.slider("Similar papers to find", 10, 30, 20, help="Number of similar papers to search")
    year_range = st.slider("Publication year range", 2015, 2025, (2018, 2025))

    st.markdown("---")
    st.markdown("<h3 style='color: white; font-size: 1.2rem;'>ℹ️ About</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color: white; font-size: 0.9rem;'>
    This AI-powered tool analyzes research papers using:
    <br>• <b>BART</b> - Abstractive summarization
    <br>• <b>ML Classification</b> - Domain prediction
    <br>• <b>NLP</b> - Keyword extraction
    <br>• <b>Semantic Scholar</b> - Paper search
    <br><br>
    <b>Version:</b> 3.2 Beta<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Reset Analysis", use_container_width=True):
        st.session_state.analysis_results = None
        st.session_state.paper_text = None
        st.session_state.similar_papers = None
        st.session_state.vector_collections = {}
        st.session_state.analysis_history = []
        st.rerun()

    if st.session_state.analysis_results:
        if st.button("📥 Download Report", use_container_width=True):
            st.info("Report download feature coming soon!")


st.markdown("<h1 class='main-title'>🔬 AI Research Paper Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Upload, Analyze, Discover - Powered by Advanced AI & Machine Learning</p>", unsafe_allow_html=True)


def split_text_into_chunks(text: str, chunk_size_words: int = 700, overlap: int = 50):
    """
    Splits text into chunks of approximately chunk_size_words with slight overlap.
    Returns list of chunks.
    """
    if not text:
        return []
    words = text.split()
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        chunk = words[i:i + chunk_size_words]
        chunks.append(" ".join(chunk))
        i += chunk_size_words - overlap
    return chunks

def safe_collection_name_for_file(filename: str):
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
    return f"paper_vectors_{name}"

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("<div class='custom-card fade-in'>", unsafe_allow_html=True)
    st.markdown("### 📄 Upload Research Paper")
    st.markdown("Drag and drop your PDF file or click to browse")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a research paper in PDF format (max 10MB)",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        
        file_name_display = uploaded_file.name if len(uploaded_file.name) <= 40 else uploaded_file.name[:37] + "..."
        col1, col2, col3 = st.columns(3)
        col1.metric("📁 Filename", file_name_display)
        col2.metric("💾 Size", f"{uploaded_file.size / 1024:.2f} KB")
        col3.metric("🕐 Status", "Ready")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 Analyze Paper", use_container_width=True, type="primary"):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                
                if st.session_state.analyzer is None:
                    status_text.text("🔄 Loading AI models...")
                    progress_bar.progress(10)
                    st.session_state.analyzer = ResearchPaperSummarizer()
                    progress_bar.progress(30)

                
                status_text.text("📄 Extracting text from PDF...")
                progress_bar.progress(40)
                text = st.session_state.analyzer.extract_text_from_pdf(tmp_path)
                st.session_state.paper_text = text
                progress_bar.progress(50)

                
                status_text.text("🧠 Analyzing paper structure...")
                sections = st.session_state.analyzer.extract_sections(text)
                progress_bar.progress(60)

                
                status_text.text("🔍 Extracting key information...")
                headings = st.session_state.analyzer.extract_section_headings(text)
                methods = st.session_state.analyzer.extract_ml_methods_and_formulas(text)
                key_terms = st.session_state.analyzer.extract_key_terms(text)
                scientific_terms = st.session_state.analyzer.identify_scientific_terms(text)
                progress_bar.progress(70)

                
                status_text.text("📝 Generating AI summary with BART...")
                summary = st.session_state.analyzer.generate_summary(sections)
                progress_bar.progress(80)

                
                status_text.text("📊 Calculating quality scores...")
                quality_scores = st.session_state.analyzer.calculate_quality_scores(text)
                progress_bar.progress(85)

                
                status_text.text("🤖 Classifying with ML model...")
                title = st.session_state.analyzer.topic_extractor.extract_title(text) if st.session_state.analyzer.topic_extractor else ""
                abstract = sections.get('abstract', '')
                ml_category = classify_research_paper(title, abstract)
                progress_bar.progress(90)

                
                status_text.text("💡 Extracting key insights...")
                insights = st.session_state.analyzer.extract_key_insights(sections, text)
                progress_bar.progress(95)

                
                status_text.text("🔍 Finding similar papers on Semantic Scholar...")
                topic_extractor = TopicExtractor()
                main_topic, all_topics = topic_extractor.extract_main_topic(text)

                searcher = SemanticScholarSearch()
                similar_papers = searcher.search_papers(
                    keyword=main_topic,
                    max_pages=2,
                    min_year=year_range[0],
                    max_year=year_range[1],
                    api_wait=2
                )
                st.session_state.similar_papers = similar_papers
                progress_bar.progress(100)

                
                st.session_state.analysis_results = {
                    'summary': summary,
                    'quality_scores': quality_scores,
                    'key_terms': key_terms,
                    'scientific_terms': scientific_terms,
                    'headings': headings,
                    'methods': methods,
                    'insights': insights,
                    'ml_category': ml_category,
                    'main_topic': main_topic,
                    'sections': sections,
                    'file_name': uploaded_file.name,
                    'analysis_time': datetime.now()
                }

                
                st.session_state.analysis_history.append({
                    'filename': uploaded_file.name,
                    'timestamp': datetime.now(),
                    'category': ml_category
                })

                
                if embedder is not None and chroma_client is not None and st.session_state.paper_text:
                    try:
                        collection_name = safe_collection_name_for_file(uploaded_file.name)
                        collection = chroma_client.get_or_create_collection(name=collection_name)

                        chunks = split_text_into_chunks(st.session_state.paper_text, chunk_size_words=700, overlap=80)
                        
                        ts = int(time.time())
                        for i, chunk in enumerate(chunks):
                            emb = embedder.encode(chunk).tolist()
                            cid = f"{i}_{ts}"
                            try:
                                collection.add(ids=[cid], embeddings=[emb], documents=[chunk])
                            except Exception:
                                
                                pass

                        st.session_state.vector_collections[collection_name] = {
                            'collection': collection,
                            'file_name': uploaded_file.name
                        }
                    except Exception as e:
                        st.warning(f"⚠️ Vector store creation failed: {e}")

                status_text.empty()
                progress_bar.empty()
                st.success("✅ Analysis completed successfully!")
                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                progress_bar.empty()
                status_text.empty()
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    else:
        
        st.markdown("""
        <div class='upload-zone'>
            <h2>📤 Drop your PDF here</h2>
            <p style='color: #666; margin-top: 10px;'>
                or click to browse from your computer
            </p>
            <p style='color: #999; font-size: 0.9rem; margin-top: 20px;'>
                Supported format: PDF • Max size: 10MB
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='custom-card fade-in'>", unsafe_allow_html=True)
    st.markdown("### 📊 Quick Stats")

    if st.session_state.analysis_results:
        results = st.session_state.analysis_results

        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Overall Quality</div>
            <div class='metric-value'>{results['quality_scores']['overall']}/100</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);'>
                <div class='metric-label'>Readability</div>
                <div class='metric-value'>{results['quality_scores']['readability']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
                <div class='metric-label'>Structure</div>
                <div class='metric-value'>{results['quality_scores']['structure']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='text-align: center;'>
            <span class='badge badge-primary'>
                🏷️ {results['ml_category']}
            </span>
        </div>
        """, unsafe_allow_html=True)

        
        if st.session_state.analysis_history:
            st.markdown("---")
            st.markdown("### 📚 Recent Analyses")
            for item in st.session_state.analysis_history[-3:]:
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px; margin: 5px 0;'>
                    <small style='color: white;'>{item['filename'][:30]}...</small><br>
                    <small style='color: rgba(255,255,255,0.7);'>{item['timestamp'].strftime('%H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📊 Upload and analyze a paper to see statistics")
        st.markdown("---")
        st.markdown("### ✨ Features")
        features = [
            "🤖 AI-Powered Summarization",
            "🏷️ ML Classification",
            "🔍 Keyword Extraction",
            "📈 Quality Scoring",
            "🔗 Similar Paper Search",
            "💡 Key Insights Detection"
            "📊 Interactive Visualizations",
            "💬 Chat with Paper Content"
        ]
        for feature in features:
            st.markdown(f"• {feature}")

    st.markdown("</div>", unsafe_allow_html=True)


if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    st.markdown("<br><br>", unsafe_allow_html=True)

    tabs = st.tabs([
        "📝 Summary",
        "📊 Analysis",
        "🔑 Keywords",
        "🔍 Similar Papers",
        "💡 Insights",
        "📈 Visualizations",
        "💬 Chat with Paper"
    ])

    
    with tabs[0]:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("## 📝 AI-Generated Summary")
        st.markdown("*Generated using BART transformer model*")
        st.markdown("---")
        if results['summary']:
            summary_parts = results['summary'].split('\n\n')
            for part in summary_parts:
                if part.strip():
                    if part.startswith('ABSTRACT:'):
                        st.markdown("### 📄 Abstract")
                        st.markdown(part.replace('ABSTRACT:', '').strip())
                    elif part.startswith('INTRODUCTION:'):
                        st.markdown("### 🎯 Introduction")
                        st.markdown(part.replace('INTRODUCTION:', '').strip())
                    elif part.startswith('METHODOLOGY:'):
                        st.markdown("### 🔬 Methodology")
                        st.markdown(part.replace('METHODOLOGY:', '').strip())
                    elif part.startswith('RESULTS:'):
                        st.markdown("### 📊 Results")
                        st.markdown(part.replace('RESULTS:', '').strip())
        else:
            st.warning("Summary not available")
        st.markdown("</div>", unsafe_allow_html=True)

    
    with tabs[1]:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("## 📊 Detailed Analysis")
        st.markdown("### 📈 Quality Scores")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=results['quality_scores']['overall'],
                title={'text': "Overall", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 50], 'color': "#f8d7da"},
                        {'range': [50, 75], 'color': "#fff3cd"},
                        {'range': [75, 100], 'color': "#d4edda"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=results['quality_scores']['readability'],
                title={'text': "Readability", 'font': {'size': 16}},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#11998e"}}
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=results['quality_scores']['structure'],
                title={'text': "Structure", 'font': {'size': 16}},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#4facfe"}}
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=results['quality_scores']['citation'],
                title={'text': "Citations", 'font': {'size': 16}},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#764ba2"}}
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📑 Paper Structure")
            if results['headings']:
                for i, heading in enumerate(results['headings'], 1):
                    st.markdown(f"{i}. **{heading}**")
            else:
                st.info("No headings detected")
        with c2:
            st.markdown("### 🤖 ML Methods Detected")
            if results['methods']:
                for i, method in enumerate(results['methods'][:10], 1):
                    st.markdown(f"{i}. `{method}`")
            else:
                st.info("No ML methods detected")
        st.markdown("</div>", unsafe_allow_html=True)

    
    with tabs[2]:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("## 🔑 Keywords & Terms")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 📊 Key Terms (Ranked by Relevance)")
            if results['key_terms']:
                terms_df = pd.DataFrame(results['key_terms'], columns=['Term', 'Score'])
                terms_df['Score'] = terms_df['Score'].round(3)
                st.dataframe(terms_df, use_container_width=True, height=400)
                fig = px.bar(terms_df.head(10), x='Score', y='Term', orientation='h', title="Top 10 Key Terms", color='Score', color_continuous_scale='Viridis')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No key terms extracted")
        with col2:
            st.markdown("### 🔬 Scientific Terms")
            if results['scientific_terms']:
                for i, term in enumerate(results['scientific_terms'], 1):
                    st.markdown(f"<span class='badge badge-info'>{term}</span>", unsafe_allow_html=True)
            else:
                st.info("No scientific terms detected")

            if st.session_state.paper_text:
                st.markdown("### ☁️ Word Cloud")
                try:
                    wc = WordCloud(width=400, height=300, background_color='white', colormap='viridis').generate(st.session_state.paper_text[:5000])
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception:
                    st.info("Word cloud generation failed")
        st.markdown("</div>", unsafe_allow_html=True)

    
    with tabs[3]:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("## 🔍 Similar Papers from Semantic Scholar")
        if st.session_state.similar_papers is not None and not st.session_state.similar_papers.empty:
            df = st.session_state.similar_papers
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; border-radius: 10px; color: white; margin-bottom: 20px;'>
                <b>📌 Search Topic:</b> {results['main_topic']}<br>
                <b>📚 Papers Found:</b> {len(df)}
            </div>
            """, unsafe_allow_html=True)
            for idx, row in df.iterrows():
                year_badge = f"<span class='badge badge-info'>{row['year']}</span>" if row['year'] != "N/A" else ""
                st.markdown(f"""
                <div style='background: white; padding: 20px; border-radius: 10px; 
                            margin: 10px 0; border-left: 4px solid #667eea;'>
                    <h4 style='color: #333; margin-bottom: 10px;'>{idx + 1}. {row['title']}</h4>
                    {year_badge}
                    <br><br>
                    <a href='{row['link']}' target='_blank' style='color: #667eea; text-decoration: none;'>
                        🔗 View Paper →
                    </a>
                </div>
                """, unsafe_allow_html=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Similar Papers (CSV)", data=csv, file_name=f"similar_papers_{results['main_topic'].replace(' ', '_')}.csv", mime="text/csv")
        else:
            st.info("No similar papers found. Try analyzing another paper!")
        st.markdown("</div>", unsafe_allow_html=True)

    
    with tabs[4]:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("## 💡 Key Insights & Conclusions")
        if results['insights']:
            for i, insight in enumerate(results['insights'], 1):
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 15px; border-radius: 10px; color: white; margin: 10px 0;'>
                    <b>{i}.</b> {insight}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No key insights detected")
        st.markdown("</div>", unsafe_allow_html=True)

    
    with tabs[5]:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("## 📈 Advanced Visualizations")
        st.markdown("### 📊 Quality Score Breakdown")
        scores_df = pd.DataFrame({
            'Metric': ['Readability', 'Structure', 'Citation', 'Overall'],
            'Score': [
                results['quality_scores']['readability'],
                results['quality_scores']['structure'],
                results['quality_scores']['citation'],
                results['quality_scores']['overall']
            ]
        })
        fig = px.bar(scores_df, x='Metric', y='Score', color='Score', color_continuous_scale='Viridis', title="Quality Metrics Comparison")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🎯 Multi-Dimensional Analysis")
        categories = ['Readability', 'Structure', 'Citations', 'Key Terms', 'Insights']
        values = [
            results['quality_scores']['readability'],
            results['quality_scores']['structure'],
            results['quality_scores']['citation'],
            min(len(results['key_terms']) * 5, 100),
            min(len(results['insights']) * 7, 100)
        ]
        radar = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', line=dict(color='#667eea', width=2), fillcolor='rgba(102, 126, 234, 0.3)'))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Paper Analysis Radar")
        st.plotly_chart(radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    
    with tabs[6]:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("## 💬 Chat with the Research Paper")
        st.markdown("Ask anything about the paper. The model will use retrieved paper context as primary source but is allowed to use general knowledge and inference to answer clearly.")

        if not st.session_state.vector_collections:
            st.info("No analyzed paper vector store available. Upload and analyze a paper first (Upload → Analyze).")
        else:
            
            collection_names = list(st.session_state.vector_collections.keys())
            display_names = [st.session_state.vector_collections[n]['file_name'] for n in collection_names]
            selected_idx = st.selectbox("Choose analyzed paper to chat with:", options=list(range(len(display_names))), format_func=lambda i: display_names[i])

            selected_collection_name = collection_names[selected_idx]
            selected_collection = st.session_state.vector_collections[selected_collection_name]['collection']

            user_query = st.text_input("Ask anything about the paper:")

            
            col_a, col_b = st.columns([1, 3])
            with col_a:
                top_k = st.selectbox("Chunks to retrieve", options=[1, 2, 3, 4], index=2)
            with col_b:
                answer_style = st.selectbox("Answer style", options=["Concise", "Detailed", "Explain like I'm 10"], index=1)

            if user_query:
                if embedder is None or chroma_client is None or genai is None:
                    st.error("Chat components not fully available. Ensure sentence-transformers, chromadb, and google-generativeai are installed and configured.")
                else:
                    try:
                        
                        q_emb = embedder.encode(user_query).tolist()
                        query_results = selected_collection.query(query_embeddings=[q_emb], n_results=top_k)
                        docs = query_results.get("documents", [[]])[0]
                        if not docs:
                            st.info("No relevant text chunks found in the paper for this query.")
                        else:
                            context = "\n\n".join(docs)

                            
                            style_instr = ""
                            if answer_style == "Concise":
                                style_instr = "Answer concisely in 2-4 sentences."
                            elif answer_style == "Detailed":
                                style_instr = "Provide a detailed explanation with key points and steps if relevant."
                            elif answer_style == "Explain like I'm 10":
                                style_instr = "Explain in simple language as if speaking to a curious 10-year-old."

                            prompt = f"""
You are an expert research assistant. Use the PAPER CONTENT (below) as a primary source of facts. 
You may use general domain knowledge and logical inference to explain or fill in gaps, but do not invent specific facts that contradict the paper.
When possible, cite (briefly) which retrieved chunk the answer uses by referencing 'context snippet #' for clarity.

{style_instr}

PAPER CONTENT (retrieved snippets):
{context}

QUESTION:
{user_query}

Provide a well-structured answer.
"""

                            
                            try:
                                model = genai.GenerativeModel("models/gemini-2.5-flash")
                                resp = model.generate_content(prompt)
                                answer_text = resp.text if hasattr(resp, "text") else str(resp)
                                st.markdown("### 🧠 Answer:")
                                st.write(answer_text)
                            except Exception as e:
                                st.error(f"Error generating answer: {e}")
                    except Exception as e:
                        st.error(f"Retrieval or generation error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p>Made with ❤️ using Streamlit | Powered by BART, Semantic Scholar & Advanced ML</p>
    <p style='font-size: 0.8rem; opacity: 0.8;'>© 2025 AI Research Paper Analyzer | Version 3.2</p>
</div>
""", unsafe_allow_html=True)
