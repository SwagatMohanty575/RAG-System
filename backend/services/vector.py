import chromadb
from chromadb.config import Settings
import json
from typing import List, Dict, Optional
from config import CHROMA_DIR


class ChromaVectorStore:
    def __init__(self, collection_name: str = "handbook", persist_directory: Optional[str] = None):
        path = persist_directory or str(CHROMA_DIR)

        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        ids = [f"{c['source_file']}_p{c['page_number']}_c{c['chunk_index']}" for c in chunks]
        texts = [c["text"] for c in chunks]
        
        # Store metadata in Chroma-compatible scalar form.
        metadatas = []
        for c in chunks:
            meta = {}
            for key, value in c.items():
                if key == "text" or value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    meta[key] = value
                elif isinstance(value, (list, dict)):
                    meta[key] = json.dumps(value, ensure_ascii=False)
                else:
                    meta[key] = str(value)
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, embedding: List[float], n_results: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "embedding": results["embeddings"][0][i],
            })
        return output

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        """Release Chroma references so Windows can delete session folders."""
        self.collection = None
        self.client = None


    def get_all_chunks(self) -> List[Dict]:
        """
        Return all stored document chunks.
        Used for:
        - global summaries
        - entity evidence
        - relationship evidence
        This is safe because every session has its own
        Chroma database.
        """
        results = self.collection.get(include=["documents","metadatas", "ids"])
        documents = (results.get("document") or [])
        metadatas = (results.get("metadatas") or [])
        ids = (results.get("ids") or [])
        output = []
        for i, text in enumerate(documents):
            output.append(
                {
                    "text": text,
                    "metadata": (
                        metadatas[i]
                        if i < len(metadatas)
                        else{}
                    ),
                    "id": (
                        ids[i]
                        if i < len(ids)
                        else None
                    ),
                }
            )
        return output
