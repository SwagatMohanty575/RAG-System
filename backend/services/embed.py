import requests
from typing import List

from sentence_transformers import SentenceTransformer


class OllamaEmbedder:
    def __init__(self):
        # Use local sentence-transformers instead of Ollama API
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])
        return embedding[0].tolist()