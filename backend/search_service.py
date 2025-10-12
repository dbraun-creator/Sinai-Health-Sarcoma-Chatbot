"""
Semantic search service module for performing cosine similarity searches
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.spatial.distance import cdist
from data_loader import DataLoader
from embedding_service import EmbeddingService
from typing import Optional
from fallback_handler import FallbackResponseHandler


class SemanticSearchService:
    """Service for performing semantic search using cosine similarity"""
    
    def __init__(self, data_loader: DataLoader, 
                 embedding_service: EmbeddingService,
                 fallback_handler: Optional[FallbackResponseHandler] = None):
        """
        Initialize semantic search service
        
        Args:
            data_loader: Data loader instance
            embedding_service: Embedding service instance
            fallback_handler: Optional fallback response handler for intelligent responses
        """
        self.data_loader = data_loader
        self.embedding_service = embedding_service
        self.fallback_handler = fallback_handler
        
        # Default responses for when no good match is found
        self.default_response = {
            "answer": "I don't have enough information to answer that question accurately. "
                     "Please try rephrasing your question with more specific details about sarcoma, "
                     "or consult with a medical professional for personalized advice.",
            "source": "System Default Response"
        }
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Handle edge cases
        if vec1.shape != vec2.shape:
            raise ValueError(f"Vector dimensions don't match: {vec1.shape} vs {vec2.shape}")
        
        # Calculate dot product and norms
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure result is in valid range due to floating point errors
        return float(np.clip(similarity, -1.0, 1.0))
    
    def batch_cosine_similarity(self, query_embedding: np.ndarray, 
                               embeddings_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and all embeddings efficiently
        
        Args:
            query_embedding: Query vector (1D array)
            embeddings_matrix: Matrix of embeddings (2D array)
            
        Returns:
            Array of similarity scores
        """
        # Reshape query for broadcasting
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity using scipy's cdist
        # Using 'cosine' distance and converting to similarity
        cosine_distances = cdist(query_embedding, embeddings_matrix, metric='cosine')
        
        # Convert distance to similarity (similarity = 1 - distance)
        similarities = 1 - cosine_distances[0]
        
        return similarities
    
    def search(self, query: str, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Perform semantic search for a query
        
        Args:
            query: User's search query
            threshold: Minimum similarity threshold (0 to 1)
            
        Returns:
            Dictionary with answer, source, and similarity score
        """
        try:
            # Generate embedding for the query
            print(f"Generating embedding for query: {query[:100]}...")
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Get all embeddings from dataset
            dataset_embeddings = self.data_loader.get_embeddings()
            
            # Calculate similarities
            similarities = self.batch_cosine_similarity(query_embedding, dataset_embeddings)
            
            # Find best match
            best_index = np.argmax(similarities)
            best_score = similarities[best_index]
            
            print(f"Best match score: {best_score:.4f} (threshold: {threshold})")
            
            # Get the best matching entry for reference
            best_match = self.data_loader.get_entry_by_index(int(best_index))
            
            # Check if best match meets threshold
            if best_score < threshold:
                # Check for emergency keywords first
                if self.fallback_handler and hasattr(self.fallback_handler, 'check_emergency_keywords'):
                    emergency_response = self.fallback_handler.check_emergency_keywords(query)
                    if emergency_response:
                        return {
                            "answer": emergency_response,
                            "source": "Emergency Response System",
                            "similarity_score": float(best_score),
                            "matched_question": None,
                            "threshold_used": threshold,
                            "emergency_response": True
                        }
                
                # Use intelligent fallback if available
                if self.fallback_handler:
                    print("Using intelligent fallback response...")
                    fallback_response = self.fallback_handler.generate_fallback_response(
                        user_query=query,
                        similarity_score=float(best_score),
                        closest_match=best_match['question']
                    )
                    fallback_response['threshold_used'] = threshold
                    return fallback_response
                else:
                    # Use default fallback if no handler available
                    return {
                        **self.default_response,
                        "similarity_score": float(best_score),
                        "matched_question": None,
                        "threshold_used": threshold
                    }
            
            return {
                "answer": best_match['answer'],
                "source": best_match['source'],
                "similarity_score": float(best_score),
                "matched_question": best_match['question'],
                "threshold_used": threshold
            }
            
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "source": "Error Response",
                "similarity_score": 0.0,
                "matched_question": None,
                "error": str(e)
            }
    
    def search_top_k(self, query: str, k: int = 5, 
                     threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find top-k most similar results
        
        Args:
            query: User's search query
            k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of top matches with scores
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Get all embeddings from dataset
            dataset_embeddings = self.data_loader.get_embeddings()
            
            # Calculate similarities
            similarities = self.batch_cosine_similarity(query_embedding, dataset_embeddings)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                score = similarities[idx]
                
                # Skip if below threshold
                if score < threshold:
                    break
                
                match = self.data_loader.get_entry_by_index(int(idx))
                results.append({
                    "answer": match['answer'],
                    "source": match['source'],
                    "question": match['question'],
                    "similarity_score": float(score)
                })
            
            return results
            
        except Exception as e:
            print(f"Error in top-k search: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search service"""
        try:
            embeddings = self.data_loader.get_embeddings()
            return {
                "total_embeddings": len(embeddings),
                "embedding_dimension": embeddings.shape[1] if len(embeddings) > 0 else 0,
                "memory_usage_mb": embeddings.nbytes / (1024 * 1024) if len(embeddings) > 0 else 0
            }
        except:
            return {
                "total_embeddings": 0,
                "embedding_dimension": 0,
                "memory_usage_mb": 0
            }
    
    def find_similar_questions(self, question_index: int, 
                               k: int = 5) -> List[Tuple[int, float]]:
        """
        Find questions similar to a given question in the dataset
        
        Args:
            question_index: Index of the question in dataset
            k: Number of similar questions to find
            
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            embeddings = self.data_loader.get_embeddings()
            
            if question_index >= len(embeddings):
                raise ValueError(f"Invalid question index: {question_index}")
            
            query_embedding = embeddings[question_index]
            
            # Calculate similarities
            similarities = self.batch_cosine_similarity(query_embedding, embeddings)
            
            # Get top-k indices (excluding the query itself)
            top_indices = np.argsort(similarities)[::-1][1:k+1]
            
            results = [(int(idx), float(similarities[idx])) for idx in top_indices]
            
            return results
            
        except Exception as e:
            print(f"Error finding similar questions: {str(e)}")
            return []


class SearchOptimizer:
    """Helper class for optimizing search performance"""
    
    @staticmethod
    def preprocess_query(query: str) -> str:
        """
        Preprocess query for better matching
        
        Args:
            query: Raw user query
            
        Returns:
            Preprocessed query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Add context if query is too short
        if len(query.split()) < 3:
            query = f"sarcoma {query}"
        
        return query
    
    @staticmethod
    def analyze_threshold_performance(search_service: SemanticSearchService,
                                     test_queries: List[str],
                                     thresholds: List[float]) -> Dict[str, Any]:
        """
        Analyze performance across different thresholds
        
        Args:
            search_service: Search service instance
            test_queries: List of test queries
            thresholds: List of thresholds to test
            
        Returns:
            Analysis results
        """
        results = {}
        
        for threshold in thresholds:
            matches = 0
            total_score = 0
            
            for query in test_queries:
                result = search_service.search(query, threshold)
                if result.get('matched_question'):
                    matches += 1
                    total_score += result['similarity_score']
            
            results[threshold] = {
                'match_rate': matches / len(test_queries),
                'avg_score': total_score / matches if matches > 0 else 0
            }
        
        return results