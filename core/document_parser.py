
"""document_parser.py
Parse PDF and Excel documents, extract text and structured numerical data (tables).
Includes heuristics for recognizing common financial statements and optional OCR fallback.
"""
from typing import List, Tuple, Dict
import io
import pandas as pd
import pdfplumber
import streamlit as st

# Optional OCR dependencies (only used if available)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

STATEMENT_KEYWORDS = {
    'income_statement': ['income statement', 'statement of profit', 'profit and loss', 'revenue', 'net income'],
    'balance_sheet': ['balance sheet', 'assets', 'liabilities', 'equity', 'total assets'],
    'cash_flow': ['cash flow', 'statement of cash flows', 'operating activities', 'investing activities']
}

class DocumentParser:
    def __init__(self):
        pass

    def _detect_statement_type(self, text: str) -> List[str]:
        found = set()
        t = text.lower()
        for k, kws in STATEMENT_KEYWORDS.items():
            for kw in kws:
                if kw in t:
                    found.add(k)
        return list(found)

    def _extract_tables_from_pdf_page(self, page):
        tables = []
        try:
            raw_tables = page.extract_tables() or []
            for t in raw_tables:
                if len(t) > 1:
                    df = pd.DataFrame(t[1:], columns=t[0])
                    tables.append(df)
        except Exception:
            pass
        return tables

    def parse_pdf_bytes(self, file_bytes: bytes, filename: str = '') -> Tuple[str, List[pd.DataFrame], List[Dict]]:
        text_chunks = []
        tables = []
        sources = []
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ''
                    header = f'[FILE: {filename}] [PAGE: {i+1}]\\n'
                    full_page_text = header + page_text
                    statement_types = self._detect_statement_type(full_page_text)
                    text_chunks.append({'text': full_page_text, 'source': {'file': filename, 'page': i+1, 'types': statement_types}})
                    sources.append({'file': filename, 'page': i+1, 'types': statement_types})
                    page_tables = self._extract_tables_from_pdf_page(page)
                    for tbl in page_tables:
                        tables.append(tbl)
        except Exception as e:
            st.warning(f'pdfplumber parsing error: {e}. OCR fallback will be attempted if available.')
            if OCR_AVAILABLE:
                try:
                    images = convert_from_bytes(file_bytes)
                    for idx, img in enumerate(images):
                        txt = pytesseract.image_to_string(img)
                        header = f'[FILE: {filename}] [PAGE_OCR: {idx+1}]\\n'
                        full = header + txt
                        statement_types = self._detect_statement_type(full)
                        text_chunks.append({'text': full, 'source': {'file': filename, 'page': idx+1, 'types': statement_types}})
                        sources.append({'file': filename, 'page': idx+1, 'types': statement_types})
                except Exception as e2:
                    st.error(f'OCR fallback failed: {e2}')
        return '\\n\\n'.join([c['text'] for c in text_chunks]), tables, sources

    def parse_excel_bytes(self, file_bytes: bytes, filename: str = '') -> Tuple[str, List[pd.DataFrame], List[Dict]]:
        texts = []
        tables = []
        sources = []
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet, engine='openpyxl')
                tables.append(df)
                header = f'[FILE: {filename}] [SHEET: {sheet}]\\n'
                texts.append(header + df.to_csv(index=False))
                stypes = self._detect_statement_type(df.to_csv(index=False))
                sources.append({'file': filename, 'sheet': sheet, 'types': stypes})
        except Exception as e:
            st.error(f'Error parsing Excel: {e}')
        return '\\n\\n'.join(texts), tables, sources

    def parse_files_with_sources(self, uploaded_files: List) -> Tuple[str, List[pd.DataFrame], List[Dict]]:
        all_texts = []
        all_tables = []
        all_sources = []
        for f in uploaded_files:
            name = f.name
            b = f.read()
            if name.lower().endswith('.pdf'):
                t, tbls, srcs = self.parse_pdf_bytes(b, filename=name)
            else:
                t, tbls, srcs = self.parse_excel_bytes(b, filename=name)
            if t:
                all_texts.append(t)
            all_tables.extend(tbls)
            all_sources.extend(srcs)
        return '\\n\\n---\\n\\n'.join(all_texts), all_tables, all_sources
