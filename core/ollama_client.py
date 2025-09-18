import os
import requests
import json
from typing import Optional
import streamlit as st

from .processing_utils import clean_text


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None, model_name: Optional[str] = None):
        self.base_url = base_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama2:latest")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Send a prompt to the Ollama API and return the generated text."""
        safe_prompt = clean_text(prompt)

        payload = {
            "model": self.model_name,
            "prompt": safe_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,  # non-streaming mode
        }

        try:
            resp = requests.post(self.base_url, json=payload, timeout=120)
            resp.raise_for_status()

            # Ollama sometimes returns multiple JSON objects (one per line)
            raw = resp.text.strip().splitlines()

            collected_text = []
            for line in raw:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        collected_text.append(data["response"])
                    elif "text" in data:
                        collected_text.append(data["text"])
                    elif "content" in data:
                        collected_text.append(data["content"])
                except json.JSONDecodeError:
                    # not valid JSON, just collect as raw text
                    collected_text.append(line)

            return " ".join(collected_text).strip()

        except Exception as e:
            st.error(f"Error calling Ollama: {e}")
            return f"Error calling Ollama: {e}"
