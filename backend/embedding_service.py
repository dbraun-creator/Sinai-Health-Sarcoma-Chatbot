"""
Embedding service module for generating embeddings using OpenAI API
"""
import numpy as np
from typing import List, Optional
from openai import OpenAI
from abc import ABC, abstractmethod
import time


class EmbeddingService(ABC):
    """Abstract base class for embedding services"""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        pass


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI implementation of embedding service"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding service
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use (default: text-embedding-3-small)
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_dimension = self._get_embedding_dimension()
    
    def _get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the model"""
        # Known dimensions for OpenAI models
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model, 1536)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of the embedding
        """
        try:
            # Clean the input text
            text = text.strip()
            if not text:
                raise ValueError("Input text cannot be empty")
            
            # Call OpenAI API
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            # Extract embedding
            embedding = response.data[0].embedding
            
            # Convert to numpy array
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of numpy arrays
        """
        try:
            # Clean texts
            cleaned_texts = [text.strip() for text in texts if text.strip()]
            
            if not cleaned_texts:
                return []
            
            # OpenAI API can handle batch requests
            response = self.client.embeddings.create(
                model=self.model,
                input=cleaned_texts
            )
            
            # Extract embeddings
            embeddings = [np.array(item.embedding, dtype=np.float32) 
                         for item in response.data]
            
            return embeddings
            
        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of tokens in text
        OpenAI's rule of thumb: ~4 characters per token
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens
        """
        return len(text) // 4
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            "model": self.model,
            "dimension": self.embedding_dimension,
            "provider": "OpenAI"
        }


class MockEmbeddingService(EmbeddingService):
    """
    Mock embedding service for testing without API calls
    Generates random embeddings of specified dimension
    """
    
    def __init__(self, dimension: int = 1536, seed: Optional[int] = 42):
        """
        Initialize mock embedding service
        
        Args:
            dimension: Dimension of embeddings to generate
            seed: Random seed for reproducibility
        """
        self.dimension = dimension
        if seed is not None:
            np.random.seed(seed)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate a mock embedding based on text hash"""
        # Use text hash as seed for consistent results
        text_hash = hash(text) % (2**32)
        rng = np.random.RandomState(text_hash)
        
        # Generate random embedding
        embedding = rng.randn(self.dimension).astype(np.float32)
        
        # Normalize to unit vector (common for embeddings)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Simulate API delay
        time.sleep(0.01)
        
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate mock embeddings for multiple texts"""
        return [self.generate_embedding(text) for text in texts]


class CachedEmbeddingService(EmbeddingService):
    """
    Wrapper service that adds caching to any embedding service
    Useful for reducing API calls during development/testing
    """
    
    def __init__(self, base_service: EmbeddingService):
        """
        Initialize cached embedding service
        
        Args:
            base_service: The underlying embedding service to wrap
        """
        self.base_service = base_service
        self.cache = {}
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding with caching"""
        # Check cache
        cache_key = text.strip().lower()
        
        if cache_key in self.cache:
            print(f"Cache hit for: {text[:50]}...")
            return self.cache[cache_key].copy()
        
        # Generate and cache
        embedding = self.base_service.generate_embedding(text)
        self.cache[cache_key] = embedding.copy()
        
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings with caching"""
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = text.strip().lower()
            if cache_key in self.cache:
                results.append(self.cache[cache_key].copy())
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.base_service.generate_embeddings_batch(uncached_texts)
            
            # Update results and cache
            for idx, embedding, text in zip(uncached_indices, new_embeddings, uncached_texts):
                results[idx] = embedding
                cache_key = text.strip().lower()
                self.cache[cache_key] = embedding.copy()
        
        return results
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache.clear()
        print("Embedding cache cleared")