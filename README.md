
# Financial Document Q&A Assistant

This repository is a cleaned, modular implementation of the Financial Document Q&A Assistant assignment.
It accepts PDF and Excel financial documents, extracts text and tables, builds embeddings, uses FAISS for retrieval,
and queries a local Ollama Small Language Model for context-aware answers.

## Features
- Accepts PDF and Excel uploads (Streamlit).
- Extracts text and tables; provides CSV downloads for extracted tables.
- Heuristics to detect Income Statement / Balance Sheet / Cash Flow statements.
- Optional OCR fallback using pdf2image + pytesseract (requires system deps).
- Embeddings with sentence-transformers + FAISS retrieval.
- Ollama client integration for local SLM generation.
- Chat interface that preserves session history for follow-up questions.
- Error handling and clear feedback in the UI.

## Quick start
1. Extract the zip and open a terminal in the project folder.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure Ollama is installed and a model is pulled (e.g. llama2).
5. Run the app:
   ```bash
   streamlit run app.py
   ```


