"""
Main Flask application for Sarcoma Q&A Semantic Search Backend
"""
import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from data_loader import CSVDataLoader
from embedding_service import OpenAIEmbeddingService
from search_service import SemanticSearchService
from config import Config

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize services
config = Config()
data_loader = CSVDataLoader(config.CSV_URL)
embedding_service = OpenAIEmbeddingService(os.getenv("OPENAI_API_KEY"))
search_service = SemanticSearchService(data_loader, embedding_service)

# Load data at startup
print("Loading dataset...")
data_loader.load_data()
print(f"Dataset loaded: {len(data_loader.get_data())} entries")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy",
        "dataset_size": len(data_loader.get_data())
    })


@app.route('/search', methods=['POST'])
def search():
    """
    Main search endpoint
    Expects JSON: {"query": "user question", "threshold": 0.7 (optional)}
    Returns: {"answer": "...", "source": "...", "processing_time": 0.123, "similarity_score": 0.89}
    """
    start_time = time.time()
    
    try:
        # Get request data
        data = request.json
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body"
            }), 400
        
        user_query = data.get('query', '').strip()
        if not user_query:
            return jsonify({
                "error": "Query cannot be empty"
            }), 400
        
        # Get threshold (optional, defaults to config value)
        threshold = data.get('threshold', config.DEFAULT_SIMILARITY_THRESHOLD)
        
        # Perform search
        result = search_service.search(user_query, threshold)
        
        # Calculate processing time
        processing_time = round(time.time() - start_time, 3)
        
        # Add processing time to result
        result['processing_time'] = processing_time
        
        return jsonify(result)
    
    except Exception as e:
        # Log the error (in production, use proper logging)
        print(f"Error in search endpoint: {str(e)}")
        
        processing_time = round(time.time() - start_time, 3)
        return jsonify({
            "error": "An error occurred processing your request",
            "message": str(e),
            "processing_time": processing_time
        }), 500


@app.route('/search/batch', methods=['POST'])
def batch_search():
    """
    Batch search endpoint for testing multiple queries
    Expects JSON: {"queries": ["question1", "question2"], "threshold": 0.7 (optional)}
    """
    start_time = time.time()
    
    try:
        data = request.json
        if not data or 'queries' not in data:
            return jsonify({
                "error": "Missing 'queries' in request body"
            }), 400
        
        queries = data.get('queries', [])
        threshold = data.get('threshold', config.DEFAULT_SIMILARITY_THRESHOLD)
        
        results = []
        for query in queries:
            if query.strip():
                result = search_service.search(query.strip(), threshold)
                result['query'] = query
                results.append(result)
        
        processing_time = round(time.time() - start_time, 3)
        
        return jsonify({
            "results": results,
            "total_processing_time": processing_time,
            "threshold_used": threshold
        })
    
    except Exception as e:
        print(f"Error in batch search endpoint: {str(e)}")
        return jsonify({
            "error": "An error occurred processing your request",
            "message": str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about the dataset"""
    try:
        data = data_loader.get_data()
        return jsonify({
            "total_questions": len(data),
            "csv_url": config.CSV_URL,
            "embedding_model": config.EMBEDDING_MODEL,
            "default_threshold": config.DEFAULT_SIMILARITY_THRESHOLD
        })
    except Exception as e:
        return jsonify({
            "error": "Failed to get statistics",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    # For local development
    app.run(debug=True, port=5000)