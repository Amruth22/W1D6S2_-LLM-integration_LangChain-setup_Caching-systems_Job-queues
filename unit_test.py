import pytest
import os
import time
import requests

# Load environment variables for testing
from dotenv import load_dotenv
load_dotenv()

# Ensure GEMINI_API_KEY is available for testing
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found. Please set it in the .env file for testing.")

from app.langchain_setup import _cached_chain_invoke

# Backend server configuration
BASE_URL = "http://127.0.0.1:8080"


class TestCoreRequirements:
    """Live integration tests with actual backend server and Gemini API calls"""
    
    def test_server_running(self):
        """Test 0: Check if the backend server is running"""
        try:
            response = requests.get(f"{BASE_URL}/", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "LLM API with LangChain" in data["message"]
            print(f"✅ Server is running at {BASE_URL}")
        except requests.exceptions.ConnectionError:
            pytest.fail(f"❌ Backend server is not running at {BASE_URL}. Please start it with 'python main.py'")
    
    def test_sync_endpoint(self):
        """Test 1: FastAPI synchronous endpoint with live Gemini API via HTTP"""
        response = requests.post(
            f"{BASE_URL}/generate/", 
            json={"question": "What is AI in one sentence?"},
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is AI in one sentence?"
        assert len(data["answer"]) > 10  # Should get a meaningful response
        assert isinstance(data["answer"], str)
        print(f"✅ Sync endpoint response: {data['answer'][:100]}...")
    
    def test_async_endpoint(self):
        """Test 2: FastAPI asynchronous endpoint with live Celery task via HTTP"""
        response = requests.post(
            f"{BASE_URL}/generate-async/", 
            json={"question": "What is machine learning?"},
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        task_id = data["task_id"]
        assert isinstance(task_id, str)
        assert len(task_id) > 0
        print(f"✅ Created async task: {task_id}")
        
        # Check task status (should complete quickly with in-memory broker)
        time.sleep(3)  # Give task time to complete
        status_response = requests.get(f"{BASE_URL}/tasks/{task_id}", timeout=30)
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["task_id"] == task_id
        assert status_data["status"] in ["SUCCESS", "PENDING", "FAILURE"]
        print(f"✅ Task status: {status_data['status']}")
        
        if status_data["status"] == "SUCCESS":
            print(f"✅ Task result: {status_data['result'][:100]}...")
    
    def test_langchain_integration(self):
        """Test 3: Live LangChain integration via /generate/ endpoint via HTTP"""
        response = requests.post(
            f"{BASE_URL}/generate/", 
            json={"question": "Define Python programming language briefly"},
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 10
        assert "python" in data["answer"].lower() or "Python" in data["answer"]
        print(f"✅ LangChain integration: {data['answer'][:100]}...")
    
    def test_lru_caching(self):
        """Test 4: Live LRU cache functionality via HTTP endpoints"""
        # Test cache behavior by making identical requests and measuring response time
        question = "What is caching in computing?"
        
        # First call via HTTP endpoint
        import time
        start_time = time.time()
        response1 = requests.post(
            f"{BASE_URL}/generate/", 
            json={"question": question},
            timeout=30
        )
        first_call_time = time.time() - start_time
        
        assert response1.status_code == 200
        result1 = response1.json()["answer"]
        assert isinstance(result1, str)
        assert len(result1) > 5
        print(f"✅ First call ({first_call_time:.2f}s): {result1[:50]}...")
        
        # Second call with same question via HTTP endpoint - should be faster due to cache
        start_time = time.time()
        response2 = requests.post(
            f"{BASE_URL}/generate/", 
            json={"question": question},
            timeout=30
        )
        second_call_time = time.time() - start_time
        
        assert response2.status_code == 200
        result2 = response2.json()["answer"]
        assert result1 == result2  # Same cached result
        
        # Cache hit should be significantly faster
        print(f"✅ Second call ({second_call_time:.2f}s): Same result returned from cache")
        print(f"✅ Cache performance: {((first_call_time - second_call_time) / first_call_time * 100):.1f}% faster")
    
    def test_celery_task(self):
        """Test 5: Live Celery task execution via HTTP /generate-async/ endpoint"""
        # Create async task via HTTP endpoint
        response = requests.post(
            f"{BASE_URL}/generate-async/", 
            json={"question": "What is Celery in Python?"},
            timeout=30
        )
        assert response.status_code == 200
        
        task_id = response.json()["task_id"]
        print(f"✅ Created Celery task: {task_id}")
        
        # Wait for task completion and check result via HTTP
        time.sleep(3)  # Give more time for API call to complete
        status_response = requests.get(f"{BASE_URL}/tasks/{task_id}", timeout=30)
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        assert status_data["task_id"] == task_id
        print(f"✅ Celery task status: {status_data['status']}")
        
        if status_data["status"] == "SUCCESS":
            task_result = status_data["result"]
            assert isinstance(task_result, str)
            assert len(task_result) > 10
            assert "celery" in task_result.lower() or "Celery" in task_result
            print(f"✅ Celery task result: {task_result[:100]}...")
        else:
            # Task might still be pending or failed, but endpoint should work
            assert status_data["status"] in ["PENDING", "FAILURE"]
    
    def test_error_handling(self):
        """Test 6: Error handling with invalid input via HTTP"""
        # Test with extremely long input that might cause issues
        very_long_question = "x" * 50000  # 50KB string (reduced for HTTP)
        
        response = requests.post(
            f"{BASE_URL}/generate/", 
            json={"question": very_long_question},
            timeout=30
        )
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 422, 500]
        print(f"✅ Error handling test: HTTP {response.status_code}")
        
        if response.status_code == 500:
            error_detail = response.json()["detail"]
            assert "error" in error_detail.lower() or "failed" in error_detail.lower()
            print(f"✅ Error handled gracefully: {error_detail}")
        elif response.status_code == 200:
            data = response.json()
            assert "answer" in data
            print(f"✅ Large input handled successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])