import re
import pandas as pd

"""
processing_utils.py
Helper functions for cleaning and chunking extracted text,
and extracting numeric/financial values.
"""

def chunk_text(text, max_chunk_size=500):
    """Split text into smaller chunks for embeddings/search."""
    words = text.split()
    chunks, current_chunk = [], []
    current_size = 0

    for word in words:
        if current_size + len(word) + 1 > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(word)
        current_size += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def extract_numbers(text):
    """Extract numeric values (currency, percentages, floats, ints) from text."""
    number_re = r"\(?-?[\d,]+(?:\.\d+)?\)?"
    return re.findall(number_re, text)


def df_to_chunks(df: pd.DataFrame):
    """Convert DataFrame rows into text chunks."""
    chunks = []
    for _, row in df.iterrows():
        row_text = " | ".join([str(v) for v in row.values])
        chunks.append(row_text)
    return chunks


def extract_metrics_from_text(text: str):
    """
    Extract key financial metrics (revenue, expenses, profit, income, cash flow) from text.
    Returns a dictionary {metric: [list of found numbers]}.
    """
    metrics = {
        "revenue": [],
        "expenses": [],
        "profit": [],
        "income": [],
        "cash flow": []
    }

    lowered = text.lower()

    for metric in metrics.keys():
        if metric in lowered:
            metrics[metric] = extract_numbers(text)

    # Keep only metrics that were actually found
    return {k: v for k, v in metrics.items() if v}


def clean_text(text: str) -> str:
    """
    Clean extracted text before sending to embeddings or Ollama.
    - Removes ₹ symbols
    - Normalizes commas in numbers (50,00,000 -> 5000000)
    - Strips weird unicode characters
    """
    text = text.replace("₹", "")
    text = text.replace(",", "")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # keep ASCII only
    return text.strip()