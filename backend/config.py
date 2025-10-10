"""
Configuration module for the Sarcoma Q&A Backend
"""
import os
from typing import Optional


class Config:
    """Configuration class with all settings"""
    
    # Data source configuration
    CSV_URL = "https://raw.githubusercontent.com/dbraun-creator/Sinai-Health-Sarcoma-Chatbot/refs/heads/main/backend/data/q_and_a_with_embd.csv"
    
    # OpenAI configuration
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSION = 1536  # Default for text-embedding-3-small
    
    # Search configuration
    DEFAULT_SIMILARITY_THRESHOLD = 0.6  # Default threshold for matching
    MIN_SIMILARITY_THRESHOLD = 0.5  # Minimum allowed threshold
    MAX_SIMILARITY_THRESHOLD = 0.95  # Maximum allowed threshold
    
    # API configuration
    MAX_QUERY_LENGTH = 1000  # Maximum characters in a query
    MAX_BATCH_SIZE = 20  # Maximum queries in batch request
    
    # Performance configuration
    CACHE_EMBEDDINGS = True  # Whether to cache embeddings in memory
    MAX_CACHE_SIZE = 1000  # Maximum number of cached embeddings
    
    # Response configuration
    DEFAULT_ERROR_MESSAGE = "An error occurred while processing your request. Please try again."
    DEFAULT_NO_MATCH_MESSAGE = (
        "I don't have enough information to answer that question accurately. "
        "Please try rephrasing your question with more specific details about sarcoma, "
        "or consult with a medical professional for personalized advice."
    )
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Development/Production settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    
    @classmethod
    def get_openai_key(cls) -> Optional[str]:
        """Get OpenAI API key from environment"""
        return os.getenv("OPENAI_API_KEY")
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        errors = []
        
        if not cls.get_openai_key():
            errors.append("OPENAI_API_KEY environment variable not set")
        
        if cls.DEFAULT_SIMILARITY_THRESHOLD < 0 or cls.DEFAULT_SIMILARITY_THRESHOLD > 1:
            errors.append("DEFAULT_SIMILARITY_THRESHOLD must be between 0 and 1")
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        
        return True
    
    @classmethod
    def get_config_dict(cls) -> dict:
        """Get configuration as dictionary"""
        return {
            "csv_url": cls.CSV_URL,
            "embedding_model": cls.EMBEDDING_MODEL,
            "embedding_dimension": cls.EMBEDDING_DIMENSION,
            "default_threshold": cls.DEFAULT_SIMILARITY_THRESHOLD,
            "environment": cls.ENVIRONMENT,
            "debug": cls.DEBUG,
            "cache_embeddings": cls.CACHE_EMBEDDINGS
        }


class DevelopmentConfig(Config):
    """Development-specific configuration"""
    DEBUG = True
    ENVIRONMENT = "development"
    LOG_LEVEL = "DEBUG"
    
    # Use mock embedding service in development to save API calls
    USE_MOCK_EMBEDDINGS = os.getenv("USE_MOCK_EMBEDDINGS", "False").lower() == "true"


class TestingConfig(Config):
    """Testing-specific configuration"""
    ENVIRONMENT = "testing"
    USE_MOCK_EMBEDDINGS = True  # Always use mock embeddings in tests
    CSV_URL = "test_data.csv"  # Use test data


class ProductionConfig(Config):
    """Production-specific configuration"""
    DEBUG = False
    ENVIRONMENT = "production"
    LOG_LEVEL = "WARNING"
    
    # Stricter thresholds in production
    DEFAULT_SIMILARITY_THRESHOLD = 0.75
    MIN_SIMILARITY_THRESHOLD = 0.6


def get_config(environment: Optional[str] = None) -> Config:
    """
    Get configuration based on environment
    
    Args:
        environment: Environment name (development, testing, production)
                    If None, uses ENVIRONMENT env variable
    
    Returns:
        Configuration instance
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "production").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "testing": TestingConfig,
        "production": ProductionConfig
    }
    
    config_class = config_map.get(environment, ProductionConfig)
    return config_class()


# Example usage for different deployment scenarios
"""
# For Render deployment (production):
# Set environment variables:
# OPENAI_API_KEY=your_key_here
# ENVIRONMENT=production

# For local development:
# Create .env file with:
# OPENAI_API_KEY=your_key_here
# ENVIRONMENT=development
# USE_MOCK_EMBEDDINGS=False  # Set to True to avoid API calls

# For testing:
# ENVIRONMENT=testing
# USE_MOCK_EMBEDDINGS=True
"""