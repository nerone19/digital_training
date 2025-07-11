"""
Test script for the YouTube RAG FastAPI application
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint"""
    response = requests.get(f"{BASE_URL}/")
    print("Root endpoint response:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_status():
    """Test status endpoint"""
    response = requests.get(f"{BASE_URL}/status")
    print("Status endpoint response:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_data_info():
    """Test data info endpoint"""
    response = requests.get(f"{BASE_URL}/data")
    print("Data info endpoint response:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_query():
    """Test query endpoint"""
    query_data = {
        "question": "What is the main topic discussed?",
        "use_rag": True
    }
    
    response = requests.post(f"{BASE_URL}/query", json=query_data)
    print("Query endpoint response:")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")
    print()

def test_process_videos():
    """Test video processing endpoint"""
    process_data = {
        "urls": ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],  # Rick Roll for testing
        "batch_size": 1
    }
    
    response = requests.post(f"{BASE_URL}/process", json=process_data)
    print("Process videos endpoint response:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_sub_queries():
    """Test sub-query generation"""
    query_data = {
        "question": "How to improve calisthenics performance?"
    }
    
    response = requests.post(f"{BASE_URL}/generate-sub-queries", json=query_data)
    print("Sub-queries endpoint response:")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")
    print()

def test_step_back_query():
    """Test step-back query generation"""
    query_data = {
        "question": "What specific exercises improve pull-up strength?"
    }
    
    response = requests.post(f"{BASE_URL}/generate-step-back-query", json=query_data)
    print("Step-back query endpoint response:")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")
    print()

if __name__ == "__main__":
    print("Testing YouTube RAG FastAPI Application")
    print("=" * 50)
    
    try:
        # Test basic endpoints
        test_root()
        test_status()
        test_data_info()
        
        # Test query-related endpoints
        test_query()
        test_sub_queries()
        test_step_back_query()
        
        # Note: We're not testing video processing as it requires actual setup
        # test_process_videos()
        
    except requests.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the FastAPI server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error during testing: {e}")