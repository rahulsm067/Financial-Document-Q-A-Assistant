
"""app.py
Entry point for the Financial Document Q&A Streamlit application.
Run with: streamlit run app.py
"""
from ui.components import render_sidebar, render_chat_and_results
from core.document_parser import DocumentParser
from core.embeddings_retrieval import EmbedderRetriever
from core.ollama_client import OllamaClient

import streamlit as st

st.set_page_config(page_title="Financial Doc Q&A Assistant", layout="wide")

# Initialize singletons / resources
@st.cache_resource
def get_doc_parser():
    return DocumentParser()

@st.cache_resource
def get_embedder_retriever(embed_model_name=None):
    return EmbedderRetriever(embed_model_name=embed_model_name)

@st.cache_resource
def get_ollama_client(base_url=None, model_name=None):
    return OllamaClient(base_url=base_url, model_name=model_name)

doc_parser = get_doc_parser()
embedder_retriever = get_embedder_retriever()
ollama_client = get_ollama_client()

# UI: Sidebar (uploads + settings)
settings = render_sidebar()

# If documents uploaded and processed, handle processing and chat
if settings.get('process_docs_btn'):
    uploaded_files = settings.get('uploaded_files', [])
    if not uploaded_files:
        st.warning('Please upload at least one PDF or Excel file.')
    else:
        try:
            with st.spinner('Parsing documents...'):
                texts, tables, file_sources = doc_parser.parse_files_with_sources(uploaded_files)
            st.session_state['docs_text'] = texts
            st.session_state['tables'] = tables
            st.session_state['file_sources'] = file_sources

            # Chunking + embeddings + index build
            with st.spinner('Creating embeddings and index...'):
                chunks = embedder_retriever.chunk_texts(texts, sources=file_sources)
                embedder_retriever.build_index_from_texts(chunks)

            st.success('Documents processed. You can now ask questions in the chat.')
        except Exception as e:
            st.error(f'Processing failed: {e}')

# Render chat and results area
render_chat_and_results(
    embedder_retriever=embedder_retriever,
    ollama_client=ollama_client
)
