"""Chroma vector store retriever for TTD-DR agent."""

import pickle
from pathlib import Path
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class ChromaRetriever:
    """Retrieves documents from persisted Chroma vector store."""
    
    def __init__(self, manifest_path: str = "data/vectorstores/chroma_manifest.pkl", persist_directory: str = "data/vectorstores/chroma_feasibility"):
        """Initialize Chroma retriever with manifest and persistence directory."""
        self.manifest_path = Path(manifest_path)
        self.persist_directory = Path(persist_directory)
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}. Run build_vector_db.ipynb to create it.")
        
        with self.manifest_path.open("rb") as f:
            self.manifest = pickle.load(f)
        
        self.embeddings = OpenAIEmbeddings(model=self.manifest["embedding_model"])
        self.vector_store = Chroma(
            collection_name=self.manifest["collection_name"],
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for query."""
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        return [{
            "content": doc.page_content,
            "metadata": {
                "source": doc.metadata.get("source", "Unknown"),
                "name": doc.metadata.get("name", "Unknown"),
                "provider": doc.metadata.get("provider", ""),
                "notes": doc.metadata.get("notes", ""),
                "score": float(score)
            }
        } for doc, score in results]
    
    def get_manifest_summary(self) -> Dict[str, Any]:
        """Return manifest metadata."""
        return {
            "collection_name": self.manifest["collection_name"],
            "embedding_model": self.manifest["embedding_model"],
            "chunk_size": self.manifest["chunk_size"],
            "chunk_overlap": self.manifest["chunk_overlap"],
            "generated_at": self.manifest["generated_at"],
            "num_documents": len(self.manifest.get("document_summary", []))
        }
