import streamlit as st
from typing import Dict
import pandas as pd

def render_sidebar() -> Dict:
    st.sidebar.title("Upload & Settings")
    uploaded_files = st.sidebar.file_uploader("Upload PDF or Excel files", type=['pdf','xls','xlsx'], accept_multiple_files=True)
    process_docs_btn = st.sidebar.button("Process Documents")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Retrieval & Model")
    top_k = st.sidebar.number_input("Top-k retrieval", min_value=1, max_value=20, value=5, step=1)
    temperature = st.sidebar.slider("Model temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    model_name = st.sidebar.text_input("Ollama model name", value=st.session_state.get('ollama_model', 'llama2'))

    # store certain settings to session_state for cross-component access
    st.session_state['top_k'] = top_k
    st.session_state['temperature'] = temperature
    st.session_state['ollama_model'] = model_name

    return {
        'uploaded_files': uploaded_files,
        'process_docs_btn': process_docs_btn,
        'top_k': top_k,
        'temperature': temperature,
        'model_name': model_name
    }

def _download_button_df(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label, data=csv, file_name=filename, mime='text/csv')

def render_chat_and_results(embedder_retriever, ollama_client):
    st.header("📊 Financial Document Q&A Assistant")
    cols = st.columns([1,2])
    with cols[0]:
        st.subheader("Extracted Tables & Downloads")
        tables = st.session_state.get('tables', [])
        if tables:
            for i, tbl in enumerate(tables):
                st.markdown(f"**Table {i+1}** - {tbl.shape[0]} rows x {tbl.shape[1]} cols")
                st.dataframe(tbl.head(10))
                _download_button_df(tbl, f"extracted_table_{i+1}.csv", f"Download Table {i+1} as CSV")
        else:
            st.info("No tables to preview.")
        st.markdown('---')
        st.subheader('Document Metrics (naive extraction)')
        docs_text = st.session_state.get('docs_text', '')
        if docs_text:
            from core.processing_utils import extract_metrics_from_text
            metrics = extract_metrics_from_text(docs_text)
            if metrics:
                st.table(pd.DataFrame(list(metrics.items()), columns=['Metric','Value']))
            else:
                st.info('No obvious metrics found via heuristics.')
        else:
            st.info('No document processed yet.')

    with cols[1]:
        st.subheader('Chat (Conversations are preserved during the session)')
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        query = st.text_input('Ask a question about the uploaded documents', key='user_query')
        if st.button('Ask'):
            if not st.session_state.get('docs_text'):
                st.warning('Upload and process documents first.')
            elif not query:
                st.warning('Enter a question.')
            else:
                st.session_state['chat_history'].append(('user', query))
                # retrieval
                top_k = st.session_state.get('top_k', 5)
                try:
                    results = embedder_retriever.retrieve(query, top_k=top_k)
                except Exception as e:
                    st.error(f'Retrieval failed: {e}')
                    results = []
                context_pieces = []
                for score, item in results:
                    src = item.get('source', {})
                    src_str = f"[file={src.get('file', 'unknown')} page={src.get('page', '')} types={src.get('types', [])}]"
                    context_pieces.append(src_str + "\\n" + item.get('text',''))
                context = "\\n\\n---\\n\\n".join(context_pieces)
                system_prompt = ("You are a helpful financial document assistant. Answer user questions using ONLY the provided context. " 
                                 "If the answer is not present, say you could not find the information and suggest what to check in the document.")
                conversation = "\\n".join([f"User: {m[1]}" if m[0]=='user' else f"Assistant: {m[1]}" for m in st.session_state['chat_history'] if isinstance(m, tuple)])
                prompt = f"SYSTEM:\\n{system_prompt}\\n\\nCONTEXT:\\n{context}\\n\\nCONVERSATION:\\n{conversation}\\n\\nUSER QUESTION:\\n{query}\\n\\nProvide an accurate concise answer with references to which chunk/page the information came from when possible."
                # call Ollama
                temperature = st.session_state.get('temperature', 0.0)
                ollama_client.model_name = st.session_state.get('ollama_model', ollama_client.model_name)
                with st.spinner('Generating answer with local Ollama model...'):
                    answer = ollama_client.generate(prompt, temperature=temperature)
                st.session_state['chat_history'].append(('assistant', answer))

        # show chat history with simple styling
        for role, text in st.session_state.get('chat_history', []):
            if role == 'user':
                st.markdown(f"**You:** {text}")
            else:
                st.markdown(f"**Assistant:** {text}")
