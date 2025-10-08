"""
Data loading module with abstract base class for easy future migration to databases
"""
import pandas as pd
import numpy as np
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class DataLoader(ABC):
    """Abstract base class for data loading - easily extensible to databases"""
    
    @abstractmethod
    def load_data(self) -> None:
        """Load data from source"""
        pass
    
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """Get loaded data"""
        pass
    
    @abstractmethod
    def get_embeddings(self) -> np.ndarray:
        """Get embeddings as numpy array"""
        pass
    
    @abstractmethod
    def get_entry_by_index(self, index: int) -> Dict[str, Any]:
        """Get a single entry by index"""
        pass


class CSVDataLoader(DataLoader):
    """CSV-specific implementation of DataLoader"""
    
    def __init__(self, csv_url: str):
        """
        Initialize CSV data loader
        
        Args:
            csv_url: URL to the CSV file on GitHub
        """
        self.csv_url = csv_url
        self.data = None
        self.embeddings_array = None
    
    def load_data(self) -> None:
        """Load data from CSV URL and process embeddings"""
        try:
            # Load CSV from URL
            self.data = pd.read_csv(self.csv_url)
            
            # Validate required columns
            required_columns = ['questions', 'answers', 'sources', 'embeddings']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Parse embeddings from string to numpy arrays
            self._parse_embeddings()
            
            print(f"Successfully loaded {len(self.data)} entries from CSV")
            
        except Exception as e:
            raise Exception(f"Failed to load data from CSV: {str(e)}")
    
    def _parse_embeddings(self) -> None:
        """Parse embedding strings to numpy arrays"""
        embeddings_list = []
        
        for idx, embedding_str in enumerate(self.data['embeddings']):
            try:
                # Parse string representation of list to actual list
                # Handle both JSON format and Python list format
                if isinstance(embedding_str, str):
                    # Remove brackets and split by comma
                    embedding_str = embedding_str.strip('[]')
                    embedding = [float(x.strip()) for x in embedding_str.split(',')]
                else:
                    # If it's already a list (shouldn't happen with CSV, but safety check)
                    embedding = embedding_str
                
                embeddings_list.append(embedding)
                
            except Exception as e:
                print(f"Warning: Failed to parse embedding at index {idx}: {str(e)}")
                # Use zero vector as fallback
                if embeddings_list:
                    # Use same dimension as previous embeddings
                    embeddings_list.append([0] * len(embeddings_list[0]))
                else:
                    # Default to 1536 dimensions (text-embedding-3-small default)
                    embeddings_list.append([0] * 1536)
        
        # Convert to numpy array for efficient computation
        self.embeddings_array = np.array(embeddings_list, dtype=np.float32)
        print(f"Parsed embeddings shape: {self.embeddings_array.shape}")
    
    def get_data(self) -> pd.DataFrame:
        """Get the loaded dataframe"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data
    
    def get_embeddings(self) -> np.ndarray:
        """Get embeddings as numpy array for efficient similarity computation"""
        if self.embeddings_array is None:
            raise ValueError("Embeddings not loaded. Call load_data() first.")
        return self.embeddings_array
    
    def get_entry_by_index(self, index: int) -> Dict[str, Any]:
        """
        Get a single entry by index
        
        Args:
            index: Row index
            
        Returns:
            Dictionary with question, answer, and source
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if index < 0 or index >= len(self.data):
            raise IndexError(f"Index {index} out of range (0-{len(self.data)-1})")
        
        row = self.data.iloc[index]
        return {
            'question': row['questions'],
            'answer': row['answers'],
            'source': row['sources']
        }
    
    def reload_data(self) -> None:
        """Reload data from source - useful for updates"""
        self.data = None
        self.embeddings_array = None
        self.load_data()


class DatabaseDataLoader(DataLoader):
    """
    Placeholder for future database implementation
    This class would connect to PostgreSQL, MongoDB, or other databases
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize database connection
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        # In real implementation, initialize database connection here
        raise NotImplementedError("Database loader not yet implemented")
    
    def load_data(self) -> None:
        """Load data from database"""
        # Implementation would query database and load data
        pass
    
    def get_data(self) -> pd.DataFrame:
        """Get data as DataFrame"""
        pass
    
    def get_embeddings(self) -> np.ndarray:
        """Get embeddings from database"""
        pass
    
    def get_entry_by_index(self, index: int) -> Dict[str, Any]:
        """Get entry from database by index"""
        pass