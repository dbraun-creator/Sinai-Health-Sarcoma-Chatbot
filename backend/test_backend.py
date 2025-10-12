"""
Test script for Sarcoma Q&A Backend
Run this after starting the backend server to test functionality
"""
import requests
import json
import time
from typing import Dict, List, Any


class BackendTester:
    """Test suite for the backend API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize tester
        
        Args:
            base_url: Base URL of the backend server
        """
        self.base_url = base_url
        self.results = []
    
    def test_health(self) -> bool:
        """Test health endpoint"""
        print("\n" + "="*50)
        print("Testing Health Endpoint")
        print("="*50)
        
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed")
                print(f"   Dataset size: {data.get('dataset_size', 'unknown')}")
                return True
            else:
                print(f"❌ Health check failed: Status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    def test_search(self, query: str, threshold: float = 0.7) -> Dict[str, Any]:
        """Test search endpoint"""
        print("\n" + "="*50)
        print(f"Testing Search: '{query}'")
        print(f"Threshold: {threshold}")
        print("="*50)
        
        try:
            response = requests.post(
                f"{self.base_url}/search",
                json={"query": query, "threshold": threshold}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Search successful")
                print(f"   Similarity Score: {data.get('similarity_score', 0):.3f}")
                print(f"   Processing Time: {data.get('processing_time', 0):.3f}s")
                
                if data.get('matched_question'):
                    print(f"   Matched Question: {data['matched_question'][:100]}...")
                    print(f"   Answer Preview: {data['answer'][:200]}...")
                    print(f"   Source: {data['source']}")
                else:
                    print(f"   ⚠️  No match found (below threshold)")
                
                return data
            else:
                print(f"❌ Search failed: Status {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            return {}
    
    def test_batch_search(self, queries: List[str], threshold: float = 0.7):
        """Test batch search endpoint"""
        print("\n" + "="*50)
        print(f"Testing Batch Search ({len(queries)} queries)")
        print("="*50)
        
        try:
            response = requests.post(
                f"{self.base_url}/search/batch",
                json={"queries": queries, "threshold": threshold}
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                print(f"✅ Batch search successful")
                print(f"   Total Processing Time: {data.get('total_processing_time', 0):.3f}s")
                print(f"   Results returned: {len(results)}")
                
                for i, result in enumerate(results, 1):
                    score = result.get('similarity_score', 0)
                    matched = "✓" if result.get('matched_question') else "✗"
                    print(f"   Query {i}: Score={score:.3f} Match={matched}")
                
                return data
            else:
                print(f"❌ Batch search failed: Status {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ Batch search error: {str(e)}")
            return {}
    
    def test_threshold_sensitivity(self, query: str):
        """Test how different thresholds affect results"""
        print("\n" + "="*50)
        print(f"Testing Threshold Sensitivity")
        print(f"Query: '{query}'")
        print("="*50)
        
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        results = []
        
        for threshold in thresholds:
            response = requests.post(
                f"{self.base_url}/search",
                json={"query": query, "threshold": threshold}
            )
            
            if response.status_code == 200:
                data = response.json()
                score = data.get('similarity_score', 0)
                matched = "✓" if data.get('matched_question') else "✗"
                results.append((threshold, score, matched))
                print(f"   Threshold {threshold}: Score={score:.3f} Match={matched}")
        
        return results
    
    def test_stats(self):
        """Test statistics endpoint"""
        print("\n" + "="*50)
        print("Testing Statistics Endpoint")
        print("="*50)
        
        try:
            response = requests.get(f"{self.base_url}/stats")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Statistics retrieved")
                for key, value in data.items():
                    print(f"   {key}: {value}")
                return data
            else:
                print(f"❌ Stats failed: Status {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ Stats error: {str(e)}")
            return {}
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n" + "="*50)
        print("Testing Edge Cases")
        print("="*50)
        
        # Test empty query
        print("\n1. Testing empty query...")
        response = requests.post(f"{self.base_url}/search", json={"query": ""})
        if response.status_code == 400:
            print("   ✅ Empty query handled correctly")
        else:
            print(f"   ❌ Unexpected status: {response.status_code}")
        
        # Test missing query
        print("\n2. Testing missing query...")
        response = requests.post(f"{self.base_url}/search", json={})
        if response.status_code == 400:
            print("   ✅ Missing query handled correctly")
        else:
            print(f"   ❌ Unexpected status: {response.status_code}")
        
        # Test very long query
        print("\n3. Testing very long query...")
        long_query = "sarcoma " * 200
        response = requests.post(f"{self.base_url}/search", json={"query": long_query})
        if response.status_code in [200, 400]:
            print(f"   ✅ Long query handled (status: {response.status_code})")
        else:
            print(f"   ❌ Unexpected status: {response.status_code}")
        
        # Test invalid threshold
        print("\n4. Testing invalid threshold...")
        response = requests.post(f"{self.base_url}/search", 
                                json={"query": "test", "threshold": 1.5})
        print(f"   Status: {response.status_code}")
    
    def test_fallback_responses(self):
        """Test intelligent fallback responses"""
        print("\n" + "="*50)
        print("Testing Intelligent Fallback Responses")
        print("="*50)
        
        # Test queries that should trigger fallback
        fallback_queries = [
            "What's the weather like?",  # Off-topic
            "I have severe chest pain",  # Emergency
            "Hello!",  # Greeting
            "diagnose my symptoms",  # Personal medical advice
            "sarcomaa tretment"  # Typo that might still work
        ]
        
        for query in fallback_queries:
            print(f"\nTesting: '{query}'")
            response = requests.post(
                f"{self.base_url}/search",
                json={"query": query, "threshold": 0.8}  # High threshold to trigger fallback
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('fallback_used'):
                    print(f"   ✅ Fallback triggered")
                    print(f"   Model: {data.get('fallback_model', 'N/A')}")
                    print(f"   Answer preview: {data['answer'][:150]}...")
                else:
                    print(f"   ℹ️  Matched with score: {data.get('similarity_score', 0):.3f}")
            
            time.sleep(0.5)  # Rate limiting
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "🚀"*25)
        print("SARCOMA Q&A BACKEND TEST SUITE")
        print("🚀"*25)
        
        # Check if server is running
        if not self.test_health():
            print("\n❌ Server not responding. Please start the backend first.")
            return
        
        # Test statistics
        self.test_stats()
        
        # Test single searches with various queries
        test_queries = [
            "What is sarcoma?",
            "symptoms of soft tissue sarcoma",
            "treatment options for bone sarcoma",
            "Can sarcoma be cured?",
            "random unrelated query about weather",  # Should trigger fallback
            "I have severe chest pain and fever",  # Should trigger emergency response
            "hello, how are you?",  # Greeting - should trigger fallback
            "what is the meaning of life?"  # Off-topic - should trigger fallback
        ]
        
        for query in test_queries:
            self.test_search(query)
            time.sleep(0.5)  # Be nice to the API
        
        # Test batch search
        batch_queries = [
            "What causes sarcoma?",
            "How is sarcoma diagnosed?",
            "What are the risk factors?"
        ]
        self.test_batch_search(batch_queries)
        
        # Test threshold sensitivity
        self.test_threshold_sensitivity("sarcoma treatment")
        
        # Test intelligent fallback responses
        self.test_fallback_responses()
        
        # Test edge cases
        self.test_edge_cases()
        
        print("\n" + "="*50)
        print("✅ TEST SUITE COMPLETED")
        print("="*50)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Sarcoma Q&A Backend")
    parser.add_argument(
        "--url",
        default="http://localhost:5000",
        help="Backend URL (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--query",
        help="Single query to test"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    tester = BackendTester(args.url)
    
    if args.query:
        # Test single query
        tester.test_health()
        tester.test_search(args.query, args.threshold)
    else:
        # Run full test suite
        tester.run_all_tests()


if __name__ == "__main__":
    main()