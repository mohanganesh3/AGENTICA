from langchain.tools import BaseTool
from typing import Optional, Type, List, Dict, Any
from pydantic import BaseModel, Field
import numpy as np
import faiss
import json
import os
import pickle
import hashlib
from sentence_transformers import SentenceTransformer

class VectorEmbeddingInput(BaseModel):
    """Input for vector embedding tool."""
    document_id: str = Field(..., description="Unique identifier for the document")
    document_text: str = Field(..., description="Text content to embed")
    document_metadata: dict = Field({}, description="Metadata associated with the document")
    index_name: str = Field("default", description="Name of the index to use")
    operation: str = Field("add", description="Operation: 'add', 'query', 'delete'")
    query_text: str = Field(None, description="Query text for search operations")
    top_k: int = Field(5, description="Number of results to return for queries")

class VectorEmbeddingTool(BaseTool):
    name = "vector_embedding_tool"
    description = "Create and search vector embeddings for document content"
    args_schema: Type[BaseModel] = VectorEmbeddingInput
    
    def __init__(self):
        super().__init__()
        # Initialize the embedding model
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Directory for storing indices
        self.index_dir = "vector_indices"
        os.makedirs(self.index_dir, exist_ok=True)
        
        # In-memory storage for indices
        self.indices = {}
        self.metadata = {}
        self.id_mapping = {}
        
        # Load existing indices
        self._load_indices()
    
    def _load_indices(self):
        """Load existing FAISS indices from disk."""
        for file in os.listdir(self.index_dir):
            if file.endswith('.index'):
                index_name = file[:-6]  # Remove .index
                self._load_index(index_name)
    
    def _load_index(self, index_name: str):
        """Load a specific index and its metadata."""
        index_path = os.path.join(self.index_dir, f"{index_name}.index")
        metadata_path = os.path.join(self.index_dir, f"{index_name}.metadata")
        mapping_path = os.path.join(self.index_dir, f"{index_name}.mapping")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path) and os.path.exists(mapping_path):
            try:
                # Load FAISS index
                self.indices[index_name] = faiss.read_index(index_path)
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    self.metadata[index_name] = pickle.load(f)
                
                # Load ID mapping
                with open(mapping_path, 'rb') as f:
                    self.id_mapping[index_name] = pickle.load(f)
                
                return True
            except Exception as e:
                print(f"Error loading index {index_name}: {str(e)}")
                return False
        
        return False
    
    def _save_index(self, index_name: str):
        """Save index and metadata to disk."""
        if index_name in self.indices:
            index_path = os.path.join(self.index_dir, f"{index_name}.index")
            metadata_path = os.path.join(self.index_dir, f"{index_name}.metadata")
            mapping_path = os.path.join(self.index_dir, f"{index_name}.mapping")
            
            try:
                # Save FAISS index
                faiss.write_index(self.indices[index_name], index_path)
                
                # Save metadata
                with open(metadata_path, 'wb') as f:
                    pickle.dump(self.metadata.get(index_name, {}), f)
                
                # Save ID mapping
                with open(mapping_path, 'wb') as f:
                    pickle.dump(self.id_mapping.get(index_name, {}), f)
                
                return True
            except Exception as e:
                print(f"Error saving index {index_name}: {str(e)}")
                return False
        
        return False
    
    def _ensure_index_exists(self, index_name: str):
        """Ensure the specified index exists."""
        if index_name not in self.indices:
            # Create a new index
            embedding_size = self.model.get_sentence_embedding_dimension()
            index = faiss.IndexFlatL2(embedding_size)
            self.indices[index_name] = index
            self.metadata[index_name] = {}
            self.id_mapping[index_name] = {}
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        # Truncate very long texts to avoid memory issues
        if len(text) > 10000:
            text = text[:10000]
        
        # Get embedding
        embedding = self.model.encode(text)
        return embedding
    
    def _add_document(self, index_name: str, document_id: str, document_text: str, metadata: dict):
        """Add document to the index."""
        self._ensure_index_exists(index_name)
        
        # Get embedding
        embedding = self._get_embedding(document_text)
        embedding_np = np.array([embedding]).astype('float32')
        
        # Add to index
        idx = self.indices[index_name].ntotal
        self.indices[index_name].add(embedding_np)
        
        # Store mapping from document_id to index
        self.id_mapping[index_name][document_id] = idx
        
        # Store metadata
        self.metadata[index_name][idx] = {
            "document_id": document_id,
            "metadata": metadata
        }
        
        # Save updated index
        self._save_index(index_name)
        
        return {
            "status": "success",
            "message": f"Document {document_id} added to index {index_name}"
        }
    
    def _query_index(self, index_name: str, query_text: str, top_k: int = 5):
        """Query the index for similar documents."""
        if index_name not in self.indices:
            return {
                "status": "error",
                "message": f"Index {index_name} does not exist"
            }
        
        # Get query embedding
        query_embedding = self._get_embedding(query_text)
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.indices[index_name].search(query_embedding_np, top_k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.metadata[index_name]:
                result = self.metadata[index_name][idx].copy()
                result["distance"] = float(distances[0][i])
                result["similarity"] = 1.0 / (1.0 + float(distances[0][i]))
                results.append(result)
        
        return {
            "status": "success",
            "query": query_text,
            "results": results
        }
    
    def _delete_document(self, index_name: str, document_id: str):
        """Delete document from index."""
        if index_name not in self.indices:
            return {
                "status": "error",
                "message": f"Index {index_name} does not exist"
            }
        
        if document_id not in self.id_mapping[index_name]:
            return {
                "status": "error",
                "message": f"Document {document_id} not found in index {index_name}"
            }
        
        # FAISS doesn't support direct deletion, so we need to rebuild the index
        # In a production environment, this would be handled differently
        # For now, we'll mark the document as deleted in metadata
        
        idx = self.id_mapping[index_name][document_id]
        if idx in self.metadata[index_name]:
            self.metadata[index_name][idx]["deleted"] = True
        
        # Save updated index
        self._save_index(index_name)
        
        return {
            "status": "success",
            "message": f"Document {document_id} marked as deleted in index {index_name}"
        }
    
    def _run(self, document_id: str, document_text: str, document_metadata: dict = {},
             index_name: str = "default", operation: str = "add", 
             query_text: str = None, top_k: int = 5):
        """Run the vector embedding tool."""
        
        if operation == "add":
            return self._add_document(index_name, document_id, document_text, document_metadata)
        
        elif operation == "query":
            if not query_text:
                return {
                    "status": "error",
                    "message": "Query text is required for query operations"
                }
            return self._query_index(index_name, query_text, top_k)
        
        elif operation == "delete":
            return self._delete_document(index_name, document_id)
        
        else:
            return {
                "status": "error",
                "message": f"Unsupported operation: {operation}"
            }

