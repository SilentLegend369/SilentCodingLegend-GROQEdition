"""
Tests for the API integration module.
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock

from src.api_integration import (
    initialize_groq_client,
    handle_api_errors,
    prepare_api_request,
    process_api_response
)

class TestApiIntegration:
    
    def test_initialize_groq_client(self):
        """Test client initialization with valid API key"""
        with patch('src.config.GROQ_API_KEY', 'fake_api_key'), \
             patch('groq.Client') as mock_client:
            client = initialize_groq_client()
            mock_client.assert_called_once_with(api_key='fake_api_key')
            assert client == mock_client.return_value
    
    def test_handle_api_errors(self):
        """Test error handling wrapper function"""
        # Test successful function execution
        test_func = lambda: "success"
        result = handle_api_errors(test_func)()
        assert result == "success"
        
        # Test handling of API errors
        def failing_func():
            import groq
            import httpx
            # Create a proper httpx.Request object for the APIError
            request = httpx.Request("POST", "https://api.groq.com/v1/chat/completions")
            # Use the main groq module exceptions with proper parameters
            raise groq.APIError("API error occurred", request=request, body={"error": "test"})
        
        with pytest.raises(Exception) as exc_info:
            handle_api_errors(failing_func)()
        assert "API error occurred" in str(exc_info.value)
    
    def test_prepare_api_request(self):
        """Test API request preparation"""
        # Test with basic parameters
        basic_params = prepare_api_request(
            model="llama-3.1-70b",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7
        )
        assert basic_params["model"] == "llama-3.1-70b"
        assert len(basic_params["messages"]) == 1
        assert basic_params["temperature"] == 0.7
        
        # Test with JSON mode
        json_params = prepare_api_request(
            model="llama-3.1-70b",
            messages=[{"role": "user", "content": "Hello"}],
            json_mode=True
        )
        assert json_params["response_format"] == {"type": "json_object"}
        assert json_params["stream"] == False
    
    def test_process_api_response(self):
        """Test processing of API responses"""
        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.total_tokens = 50
        
        processed = process_api_response(mock_response)
        assert processed["content"] == "Test response"
        assert processed["tokens"] == 50
        
        # Test streaming response
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Partial content"
        
        processed_chunk = process_api_response(mock_chunk, is_streaming=True)
        assert processed_chunk["content"] == "Partial content"
