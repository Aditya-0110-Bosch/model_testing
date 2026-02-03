"""
Vector Store Module with Azure OpenAI Embeddings and FAISS
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Azure OpenAI Embedding credentials from environment
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-large")


class VectorStoreManager:
    """Manage embeddings and FAISS vector store with persistence"""
    
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        max_workers: int = 4,
        index_type: str = "cosine"  # 'cosine', 'l2', 'ip' (inner product)
    ):
        self.max_workers = max_workers
        self.index_type = index_type
        
        # Initialize Azure OpenAI Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=EMBEDDING_DEPLOYMENT,
            openai_api_version="2023-05-15",
            azure_endpoint=EMBEDDING_ENDPOINT,
            api_key=EMBEDDING_API_KEY,
            model=embedding_model
        )
        
        self.vector_store: Optional[FAISS] = None
        self.dimension: Optional[int] = None
    
    def embed_documents_parallel(self, texts: List[str]) -> List[List[float]]:
        """Embed documents in parallel for better performance"""
        
        # For large batches, split into chunks
        batch_size = 100
        all_embeddings = []
        
        def embed_batch(batch_texts):
            return self.embeddings.embed_documents(batch_texts)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                futures.append(executor.submit(embed_batch, batch))
            
            for future in as_completed(futures):
                try:
                    embeddings = future.result()
                    all_embeddings.extend(embeddings)
                except Exception as e:
                    print(f"Error embedding batch: {e}")
                    # Add zero vectors for failed embeddings
                    if self.dimension:
                        all_embeddings.extend([[0.0] * self.dimension] * batch_size)
        
        return all_embeddings
    
    def create_vector_store(
        self,
        documents: List[Document],
        use_parallel: bool = True
    ) -> FAISS:
        """Create FAISS vector store from documents"""
        
        if not documents:
            raise ValueError("No documents provided")
        
        print(f"Creating embeddings for {len(documents)} documents...")
        
        # Extract texts
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        if use_parallel and len(documents) > 50:
            # Test embedding to get dimension
            test_embedding = self.embeddings.embed_query(texts[0])
            self.dimension = len(test_embedding)
            
            # Embed all documents in parallel
            embeddings = self.embed_documents_parallel(texts)
        else:
            # Use standard embedding
            embeddings = self.embeddings.embed_documents(texts)
            self.dimension = len(embeddings[0])
        
        # Normalize embeddings for cosine similarity
        if self.index_type == "cosine":
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            embeddings = embeddings_array.tolist()
        
        # Create the appropriate FAISS index
        if self.index_type == "cosine":
            # For cosine similarity, use inner product on normalized vectors
            index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "l2":
            # L2 distance (Euclidean)
            index = faiss.IndexFlatL2(self.dimension)
        else:  # ip (inner product/dot product)
            # Inner product
            index = faiss.IndexFlatIP(self.dimension)
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Add embeddings to index
        index.add(embeddings_array)
        
        # Create FAISS vector store manually
        from langchain_community.docstore.in_memory import InMemoryDocstore
        
        # Create docstore
        index_to_id = {i: str(i) for i in range(len(documents))}
        id_to_doc = {str(i): doc for i, doc in enumerate(documents)}
        docstore = InMemoryDocstore(id_to_doc)
        
        # Create FAISS instance
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_id
        )
        
        print(f"Vector store created with {len(documents)} documents")
        return self.vector_store
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to existing vector store"""
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        
        if not documents:
            return
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Add documents
        self.vector_store.add_documents(documents)
        
        print(f"Added {len(documents)} documents successfully")
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        return_scores: bool = True
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search"""
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        if return_scores:
            # Return documents with similarity scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_metadata
            )
            return results
        else:
            # Return only documents
            docs = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_metadata
            )
            return [(doc, 0.0) for doc in docs]
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 5,
        return_scores: bool = True
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search using embedding vector"""
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        if return_scores:
            results = self.vector_store.similarity_search_by_vector_with_score(
                embedding=embedding,
                k=k
            )
            return results
        else:
            docs = self.vector_store.similarity_search_by_vector(
                embedding=embedding,
                k=k
            )
            return [(doc, 0.0) for doc in docs]
    
    def save_vector_store(self, save_path: str) -> None:
        """Save vector store to disk"""
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        # Save FAISS index
        self.vector_store.save_local(save_path)
        
        # Save additional metadata
        metadata = {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'num_documents': len(self.vector_store.docstore._dict)
        }
        
        with open(os.path.join(save_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Vector store saved to {save_path}")
    
    def load_vector_store(self, load_path: str) -> FAISS:
        """Load vector store from disk"""
        
        if not os.path.exists(load_path):
            raise ValueError(f"Path {load_path} does not exist")
        
        # Load FAISS index
        self.vector_store = FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Load metadata
        metadata_path = os.path.join(load_path, 'metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.index_type = metadata.get('index_type', self.index_type)
                self.dimension = metadata.get('dimension')
        
        print(f"Vector store loaded from {load_path}")
        return self.vector_store
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        
        if not self.vector_store:
            return {
                'initialized': False,
                'num_documents': 0,
                'dimension': None,
                'index_type': self.index_type
            }
        
        return {
            'initialized': True,
            'num_documents': len(self.vector_store.docstore._dict),
            'dimension': self.dimension,
            'index_type': self.index_type
        }
    
    def delete_vector_store(self) -> None:
        """Delete the current vector store from memory"""
        self.vector_store = None
        self.dimension = None
        print("Vector store deleted from memory")


class RetrievalConfig:
    """Configuration for retrieval parameters"""
    
    def __init__(
        self,
        k: int = 5,
        index_type: str = "cosine",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitter_type: str = "recursive"
    ):
        self.k = k
        self.index_type = index_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter_type = splitter_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'k': self.k,
            'index_type': self.index_type,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'splitter_type': self.splitter_type
        }
