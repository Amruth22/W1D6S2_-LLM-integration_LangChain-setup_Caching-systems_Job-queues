import unittest
import os
import sys
import tempfile
import shutil
import asyncio
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables
load_dotenv()

# Add the current directory to Python path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class CoreLLMIntegrationTests(unittest.TestCase):
    """Core 5 unit tests for LLM Integration with LangChain, Caching, and Job Queues"""
    
    @classmethod
    def setUpClass(cls):
        """Load configuration and validate setup"""
        # Note: This system requires GEMINI_API_KEY but we'll mock it for testing
        print("Setting up LLM Integration System tests...")
        
        # Initialize LLM Integration components (classes only, no heavy initialization)
        try:
            # Import main application components
            import main
            from config import GEMINI_API_KEY
            
            # Import FastAPI testing client
            from fastapi.testclient import TestClient
            
            cls.app = main.app
            cls.client = TestClient(main.app)
            cls.gemini_api_key = GEMINI_API_KEY
            
            # Import app components
            from app import langchain_setup, celery_app, tasks
            
            cls.langchain_setup = langchain_setup
            cls.celery_app = celery_app
            cls.tasks = tasks
            
            # Store models
            cls.QuestionRequest = main.QuestionRequest
            cls.TaskResponse = main.TaskResponse
            cls.ResultResponse = main.ResultResponse
            cls.GenerateResponse = main.GenerateResponse
            
            print("LLM integration components loaded successfully")
        except ImportError as e:
            raise unittest.SkipTest(f"Required LLM integration components not found: {e}")

    def setUp(self):
        """Set up test fixtures"""
        # Test data
        self.test_question = "What is artificial intelligence?"
        self.test_long_question = "Explain the concept of machine learning in detail with examples"
        self.test_simple_question = "Define AI"
        
        # Mock responses for testing
        self.mock_ai_response = "Artificial intelligence (AI) is the simulation of human intelligence in machines."
        self.mock_ml_response = "Machine learning is a subset of AI that enables computers to learn without explicit programming."

    def tearDown(self):
        """Clean up test fixtures"""
        # Clear any cache if needed
        try:
            if hasattr(self.langchain_setup, '_cached_chain_invoke'):
                self.langchain_setup._cached_chain_invoke.cache_clear()
        except:
            pass

    def test_01_langchain_setup_and_configuration(self):
        """Test 1: LangChain Setup and Configuration"""
        print("Running Test 1: LangChain Setup and Configuration")
        
        # Test that LangChain components are properly imported and configured
        self.assertIsNotNone(self.langchain_setup)
        
        # Test that required components exist
        self.assertTrue(hasattr(self.langchain_setup, 'llm'))
        self.assertTrue(hasattr(self.langchain_setup, 'prompt_template'))
        self.assertTrue(hasattr(self.langchain_setup, 'basic_chain'))
        self.assertTrue(hasattr(self.langchain_setup, '_cached_chain_invoke'))
        self.assertTrue(hasattr(self.langchain_setup, 'get_cached_llm_response'))
        
        # Test that functions are callable
        self.assertTrue(callable(self.langchain_setup._cached_chain_invoke))
        self.assertTrue(callable(self.langchain_setup.get_cached_llm_response))
        
        # Test prompt template structure
        prompt_template = self.langchain_setup.prompt_template
        self.assertIsNotNone(prompt_template)
        
        # Test that the prompt template has the expected input variable
        self.assertIn('question', prompt_template.input_variables)
        
        # Test LRU cache decorator
        cached_function = self.langchain_setup._cached_chain_invoke
        self.assertTrue(hasattr(cached_function, 'cache_info'))
        self.assertTrue(hasattr(cached_function, 'cache_clear'))
        
        # Test cache info structure
        cache_info = cached_function.cache_info()
        self.assertIsNotNone(cache_info)
        self.assertTrue(hasattr(cache_info, 'hits'))
        self.assertTrue(hasattr(cache_info, 'misses'))
        self.assertTrue(hasattr(cache_info, 'maxsize'))
        self.assertTrue(hasattr(cache_info, 'currsize'))
        
        # Verify cache configuration
        self.assertEqual(cache_info.maxsize, 128)  # Should match the configured maxsize
        
        print("PASS: LangChain components properly imported")
        print("PASS: Prompt template configured with question variable")
        print("PASS: LRU cache decorator applied with maxsize=128")
        print("PASS: Cache info and management functions available")
        print("PASS: LangChain setup and configuration validated")

    def test_02_lru_caching_system(self):
        """Test 2: LRU Caching System Functionality"""
        print("Running Test 2: LRU Caching System")
        
        # Clear cache before testing
        self.langchain_setup._cached_chain_invoke.cache_clear()
        
        # Test initial cache state
        cache_info = self.langchain_setup._cached_chain_invoke.cache_info()
        self.assertEqual(cache_info.hits, 0)
        self.assertEqual(cache_info.misses, 0)
        self.assertEqual(cache_info.currsize, 0)
        
        # Mock the LangChain chain to avoid actual API calls
        with patch.object(self.langchain_setup, 'basic_chain') as mock_chain:
            mock_chain.invoke.return_value = self.mock_ai_response
            
            # Test first call (should be a cache miss)
            result1 = self.langchain_setup._cached_chain_invoke(self.test_question)
            self.assertEqual(result1, self.mock_ai_response)
            
            # Check cache stats after first call
            cache_info = self.langchain_setup._cached_chain_invoke.cache_info()
            self.assertEqual(cache_info.hits, 0)
            self.assertEqual(cache_info.misses, 1)
            self.assertEqual(cache_info.currsize, 1)
            
            # Test second call with same question (should be a cache hit)
            result2 = self.langchain_setup._cached_chain_invoke(self.test_question)
            self.assertEqual(result2, self.mock_ai_response)
            self.assertEqual(result1, result2)  # Same result
            
            # Check cache stats after second call
            cache_info = self.langchain_setup._cached_chain_invoke.cache_info()
            self.assertEqual(cache_info.hits, 1)
            self.assertEqual(cache_info.misses, 1)
            self.assertEqual(cache_info.currsize, 1)
            
            # Verify that the chain was only called once (second call used cache)
            self.assertEqual(mock_chain.invoke.call_count, 1)
            
            # Test different question (should be another cache miss)
            mock_chain.invoke.return_value = self.mock_ml_response
            result3 = self.langchain_setup._cached_chain_invoke(self.test_long_question)
            self.assertEqual(result3, self.mock_ml_response)
            
            # Check cache stats after third call
            cache_info = self.langchain_setup._cached_chain_invoke.cache_info()
            self.assertEqual(cache_info.hits, 1)
            self.assertEqual(cache_info.misses, 2)
            self.assertEqual(cache_info.currsize, 2)
            
            # Test cache capacity management
            # Fill cache beyond capacity to test LRU eviction
            for i in range(130):  # More than maxsize=128
                question = f"Test question {i}"
                mock_chain.invoke.return_value = f"Test response {i}"
                self.langchain_setup._cached_chain_invoke(question)
            
            # Cache should not exceed maxsize
            cache_info = self.langchain_setup._cached_chain_invoke.cache_info()
            self.assertLessEqual(cache_info.currsize, 128)
        
        # Test cache clear functionality
        self.langchain_setup._cached_chain_invoke.cache_clear()
        cache_info = self.langchain_setup._cached_chain_invoke.cache_info()
        self.assertEqual(cache_info.currsize, 0)
        self.assertEqual(cache_info.hits, 0)
        self.assertEqual(cache_info.misses, 0)
        
        print("PASS: Cache initialization and state tracking")
        print("PASS: Cache hit and miss functionality")
        print("PASS: Cache performance optimization")
        print("PASS: Cache capacity management and LRU eviction")
        print("PASS: Cache clear and reset functionality")
        print("PASS: LRU caching system validated")

    def test_03_celery_job_queue_system(self):
        """Test 3: Celery Job Queue System and Task Management"""
        print("Running Test 3: Celery Job Queue System")
        
        # Test Celery app configuration
        celery_app = self.celery_app.celery_app
        self.assertIsNotNone(celery_app)
        self.assertEqual(celery_app.main, 'llm_tasks')
        
        # Test Celery configuration
        self.assertEqual(celery_app.conf.task_serializer, 'json')
        self.assertEqual(celery_app.conf.result_serializer, 'json')
        self.assertEqual(celery_app.conf.timezone, 'UTC')
        self.assertTrue(celery_app.conf.enable_utc)
        
        # Test broker and backend configuration
        self.assertEqual(celery_app.conf.broker_url, 'memory://')
        self.assertEqual(celery_app.conf.result_backend, 'cache+memory://')
        
        # Test task registration
        self.assertTrue(hasattr(self.tasks, 'generate_content_task'))
        task_func = self.tasks.generate_content_task
        self.assertTrue(callable(task_func))
        
        # Test task properties
        self.assertTrue(hasattr(task_func, 'delay'))
        self.assertTrue(hasattr(task_func, 'apply_async'))
        self.assertTrue(callable(task_func.delay))
        self.assertTrue(callable(task_func.apply_async))
        
        # Mock the LangChain response to avoid actual API calls
        with patch.object(self.langchain_setup, 'get_cached_llm_response') as mock_llm:
            mock_llm.return_value = self.mock_ai_response
            
            # Test task execution
            result = task_func(self.test_question)
            self.assertEqual(result, self.mock_ai_response)
            mock_llm.assert_called_once_with(self.test_question)
            
            # Test async task creation
            mock_llm.reset_mock()
            async_result = task_func.delay(self.test_simple_question)
            self.assertIsNotNone(async_result)
            self.assertIsNotNone(async_result.id)
            
            # Test task ID format (should be UUID-like)
            task_id = async_result.id
            self.assertIsInstance(task_id, str)
            self.assertGreater(len(task_id), 10)
            
            # Test task state
            self.assertIn(async_result.state, ['PENDING', 'SUCCESS', 'FAILURE'])
            
            # Test AsyncResult functionality
            async_result_obj = celery_app.AsyncResult(task_id)
            self.assertEqual(async_result_obj.id, task_id)
            self.assertIn(async_result_obj.state, ['PENDING', 'SUCCESS', 'FAILURE'])
        
        print("PASS: Celery app configuration and setup")
        print("PASS: Task registration and properties")
        print("PASS: Task execution and result handling")
        print("PASS: Async task creation and management")
        print("PASS: Task ID generation and AsyncResult functionality")
        print("PASS: Celery job queue system validated")

    def test_04_api_integration_and_models(self):
        """Test 4: API Integration and Pydantic Models"""
        print("Running Test 4: API Integration and Models")
        
        # Test Pydantic models
        # Test QuestionRequest model
        question_request = self.QuestionRequest(question=self.test_question)
        self.assertEqual(question_request.question, self.test_question)
        
        # Test TaskResponse model
        test_task_id = str(uuid.uuid4())
        task_response = self.TaskResponse(task_id=test_task_id)
        self.assertEqual(task_response.task_id, test_task_id)
        
        # Test ResultResponse model
        result_response = self.ResultResponse(
            task_id=test_task_id,
            status="SUCCESS",
            result=self.mock_ai_response,
            error=None
        )
        self.assertEqual(result_response.task_id, test_task_id)
        self.assertEqual(result_response.status, "SUCCESS")
        self.assertEqual(result_response.result, self.mock_ai_response)
        self.assertIsNone(result_response.error)
        
        # Test GenerateResponse model
        generate_response = self.GenerateResponse(
            question=self.test_question,
            answer=self.mock_ai_response
        )
        self.assertEqual(generate_response.question, self.test_question)
        self.assertEqual(generate_response.answer, self.mock_ai_response)
        
        # Test API endpoints structure
        # Test root endpoint
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        root_data = response.json()
        self.assertIn("message", root_data)
        self.assertIn("LLM API with LangChain", root_data["message"])
        
        # Test sync endpoint with mocked LLM response
        with patch.object(self.langchain_setup, 'get_cached_llm_response') as mock_llm:
            mock_llm.return_value = self.mock_ai_response
            
            response = self.client.post("/generate/", json={"question": self.test_question})
            self.assertEqual(response.status_code, 200)
            
            response_data = response.json()
            self.assertIn("question", response_data)
            self.assertIn("answer", response_data)
            self.assertEqual(response_data["question"], self.test_question)
            self.assertEqual(response_data["answer"], self.mock_ai_response)
            
            mock_llm.assert_called_once_with(self.test_question)
        
        # Test async endpoint with mocked task
        with patch.object(self.tasks, 'generate_content_task') as mock_task:
            mock_task_result = Mock()
            mock_task_result.id = str(uuid.uuid4())
            mock_task.delay.return_value = mock_task_result
            
            response = self.client.post("/generate-async/", json={"question": self.test_question})
            self.assertEqual(response.status_code, 200)
            
            response_data = response.json()
            self.assertIn("task_id", response_data)
            self.assertEqual(response_data["task_id"], mock_task_result.id)
            
            mock_task.delay.assert_called_once_with(self.test_question)
        
        # Test task status endpoint with mocked Celery result
        test_task_id = str(uuid.uuid4())
        with patch.object(self.celery_app.celery_app, 'AsyncResult') as mock_async_result:
            mock_result = Mock()
            mock_result.state = 'SUCCESS'
            mock_result.result = self.mock_ai_response
            mock_result.info = None
            mock_async_result.return_value = mock_result
            
            response = self.client.get(f"/tasks/{test_task_id}")
            self.assertEqual(response.status_code, 200)
            
            response_data = response.json()
            self.assertEqual(response_data["task_id"], test_task_id)
            self.assertEqual(response_data["status"], "SUCCESS")
            self.assertEqual(response_data["result"], self.mock_ai_response)
            self.assertIsNone(response_data["error"])
        
        # Test error handling in sync endpoint
        with patch.object(self.langchain_setup, 'get_cached_llm_response') as mock_llm:
            mock_llm.side_effect = Exception("LLM API Error")
            
            response = self.client.post("/generate/", json={"question": self.test_question})
            self.assertEqual(response.status_code, 500)
            
            error_data = response.json()
            self.assertIn("detail", error_data)
            self.assertIn("LLM call via LangChain failed", error_data["detail"])
        
        print("PASS: Pydantic model validation and structure")
        print("PASS: API endpoint structure and responses")
        print("PASS: Sync endpoint with LangChain integration")
        print("PASS: Async endpoint with Celery task creation")
        print("PASS: Task status endpoint with result handling")
        print("PASS: Error handling and exception management")
        print("PASS: API integration and models validated")

    def test_05_end_to_end_llm_workflow(self):
        """Test 5: End-to-End LLM Workflow Integration"""
        print("Running Test 5: End-to-End LLM Workflow")
        
        # Test complete workflow: Question -> LangChain -> Cache -> Response
        
        # Mock the entire LangChain workflow
        with patch.object(self.langchain_setup, 'basic_chain') as mock_chain:
            mock_chain.invoke.return_value = self.mock_ai_response
            
            # Clear cache for clean test
            self.langchain_setup._cached_chain_invoke.cache_clear()
            
            # Test first call (cache miss)
            result1 = self.langchain_setup.get_cached_llm_response(self.test_question)
            self.assertEqual(result1, self.mock_ai_response)
            
            # Verify chain was called
            mock_chain.invoke.assert_called_once_with(self.test_question)
            
            # Test cache state after first call
            cache_info = self.langchain_setup._cached_chain_invoke.cache_info()
            self.assertEqual(cache_info.misses, 1)
            self.assertEqual(cache_info.hits, 0)
            self.assertEqual(cache_info.currsize, 1)
            
            # Test second call with same question (cache hit)
            mock_chain.reset_mock()
            result2 = self.langchain_setup.get_cached_llm_response(self.test_question)
            self.assertEqual(result2, self.mock_ai_response)
            self.assertEqual(result1, result2)
            
            # Verify chain was NOT called again (cache hit)
            mock_chain.invoke.assert_not_called()
            
            # Test cache state after second call
            cache_info = self.langchain_setup._cached_chain_invoke.cache_info()
            self.assertEqual(cache_info.hits, 1)
            self.assertEqual(cache_info.misses, 1)
            self.assertEqual(cache_info.currsize, 1)
        
        # Test async workflow with mocked components
        with patch.object(self.langchain_setup, 'get_cached_llm_response') as mock_cached_llm:
            mock_cached_llm.return_value = self.mock_ml_response
            
            # Test task creation and execution
            task_result = self.tasks.generate_content_task(self.test_long_question)
            self.assertEqual(task_result, self.mock_ml_response)
            mock_cached_llm.assert_called_once_with(self.test_long_question)
        
        # Test error handling in workflow
        with patch.object(self.langchain_setup, 'basic_chain') as mock_chain:
            mock_chain.invoke.side_effect = Exception("API Rate Limit Exceeded")
            
            # Clear cache to ensure we hit the error
            self.langchain_setup._cached_chain_invoke.cache_clear()
            
            with self.assertRaises(Exception) as context:
                self.langchain_setup._cached_chain_invoke(self.test_simple_question)
            
            self.assertIn("API Rate Limit Exceeded", str(context.exception))
        
        # Test configuration validation
        from config import GEMINI_API_KEY
        # API key might be empty in test environment, which is fine
        self.assertIsInstance(GEMINI_API_KEY, (str, type(None)))
        
        # Test environment configuration
        import config
        self.assertTrue(hasattr(config, 'GEMINI_API_KEY'))
        
        # Test that dotenv is loaded
        from dotenv import load_dotenv
        # This should not raise an error
        load_dotenv()
        
        print("PASS: Complete LLM workflow (Question -> LangChain -> Cache -> Response)")
        print("PASS: Cache hit and miss behavior validation")
        print("PASS: Async task workflow integration")
        print("PASS: Error handling in LLM workflow")
        print("PASS: Configuration and environment validation")
        print("PASS: End-to-end LLM workflow validated")

def run_core_tests():
    """Run core tests and provide summary"""
    print("=" * 70)
    print("[*] Core LLM Integration Unit Tests (5 Tests)")
    print("Testing with LOCAL LLM Integration Components")
    print("=" * 70)
    
    print("[INFO] This system integrates LangChain, LRU Cache, and Celery Job Queues")
    print("[INFO] Tests validate LangChain Setup, LRU Cache, Celery Queue, API Integration, E2E Workflow")
    print()
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(CoreLLMIntegrationTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("[*] Test Results:")
    print(f"[*] Tests Run: {result.testsRun}")
    print(f"[*] Failures: {len(result.failures)}")
    print(f"[*] Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n[FAILURES]:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            print(f"    {traceback}")
    
    if result.errors:
        print("\n[ERRORS]:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            print(f"    {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n[SUCCESS] All 5 core LLM integration tests passed!")
        print("[OK] LLM integration components working correctly with local implementation")
        print("[OK] LangChain Setup, LRU Cache, Celery Queue, API Integration, E2E Workflow validated")
    else:
        print(f"\n[WARNING] {len(result.failures) + len(result.errors)} test(s) failed")
    
    return success

if __name__ == "__main__":
    print("[*] Starting Core LLM Integration Tests")
    print("[*] 5 essential tests with local LLM integration implementation")
    print("[*] Components: LangChain Setup, LRU Cache, Celery Queue, API Integration, E2E Workflow")
    print()
    
    success = run_core_tests()
    exit(0 if success else 1)