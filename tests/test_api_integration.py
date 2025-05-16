"""
Tests for the API integration functionality.
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock

from src.utils import query_groq_model, query_groq_vision_model
from src.performance_opt import cached_query_groq_model, cached_query_groq_vision_model, ResponseCache

# Test Groq API integration
class TestGroqApi:
    
    def test_query_groq_model(self, mock_groq_client):
        """Test query_groq_model sends proper parameters and returns response"""
        model_id = "llama-3.1-70b"
        messages = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "Hello"}
        ]
        temperature = 0.7
        max_tokens = 1000
        
        # Call the function
        response = query_groq_model(model_id, messages, temperature, max_tokens)
        
        # Verify client was called with correct parameters
        mock_groq_client.chat.completions.create.assert_called_once()
        call_args = mock_groq_client.chat.completions.create.call_args[1]
        assert call_args["model"] == model_id
        assert call_args["messages"] == messages
        assert call_args["temperature"] == temperature
        assert call_args["max_completion_tokens"] == max_tokens
        assert call_args["stream"] == True
        
        # Verify response was returned correctly
        assert response == mock_groq_client.chat.completions.create.return_value
    
    def test_query_groq_model_json_mode(self, mock_groq_client):
        """Test query_groq_model with JSON mode enabled"""
        model_id = "llama-3.1-70b"
        messages = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "Provide data in JSON format"}
        ]
        
        # Call with json_mode=True
        response = query_groq_model(model_id, messages, json_mode=True)
        
        # Verify JSON mode parameters
        call_args = mock_groq_client.chat.completions.create.call_args[1]
        assert call_args["response_format"] == {"type": "json_object"}
        assert call_args["stream"] == False
    
    def test_query_groq_vision_model(self, mock_groq_client):
        """Test query_groq_vision_model sends proper parameters for vision API"""
        model_id = "llama-3.1-70b-vision"
        messages = [
            {"role": "system", "content": "You are a vision AI."},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "What's in this image?"}, 
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
                ]
            }
        ]
        
        # Call the function
        response = query_groq_vision_model(model_id, messages)
        
        # Verify client was called with correct parameters
        call_args = mock_groq_client.chat.completions.create.call_args[1]
        assert call_args["model"] == model_id
        assert call_args["messages"] == messages
        assert "max_completion_tokens" in call_args
        assert call_args["stream"] == True

# Test cached API calls
class TestCachedApiCalls:
    
    def test_cached_query_groq_model(self, mock_groq_client, temp_data_dir):
        """Test cached_query_groq_model caches responses correctly"""
        # Configure cache to use the temp directory
        mock_cache = MagicMock(spec=ResponseCache)
        # Mock cache behavior - first call returns None (cache miss), second call returns cached result
        mock_response = MagicMock() 
        mock_cache.get.side_effect = [None, mock_response]  # First call cache miss, second call cache hit
        mock_cache.set = MagicMock()  # Just mock the set method, don't try to serialize
        
        # Also mock streamlit to avoid st.toast errors
        with patch('src.performance_opt.response_cache', mock_cache), \
             patch('streamlit.toast'):
            model_id = "llama-3.1-70b"
            messages = [{"role": "user", "content": "Hello"}]
            
            # First call should hit the API
            response1 = cached_query_groq_model(model_id, messages, use_cache=True, stream=False)
            assert mock_groq_client.chat.completions.create.call_count == 1
            
            # Second call with same parameters should use cache
            response2 = cached_query_groq_model(model_id, messages, use_cache=True, stream=False)
            # API call count should still be 1
            assert mock_groq_client.chat.completions.create.call_count == 1
            
            # Call with use_cache=False should hit API again
            response3 = cached_query_groq_model(model_id, messages, use_cache=False)
            assert mock_groq_client.chat.completions.create.call_count == 2
    
    def test_cached_query_groq_vision_model(self, mock_groq_client, temp_data_dir):
        """Test cached_query_groq_vision_model caches responses correctly"""
        # Configure cache to use the temp directory
        mock_cache = MagicMock(spec=ResponseCache)
        # Mock cache behavior - first call returns None (cache miss), second call returns cached result
        mock_response = MagicMock() 
        mock_cache.get.side_effect = [None, mock_response]  # First call cache miss, second call cache hit
        mock_cache.set = MagicMock()  # Just mock the set method, don't try to serialize
        
        # Also mock streamlit to avoid st.toast errors
        with patch('src.performance_opt.response_cache', mock_cache), \
             patch('streamlit.toast'):
            model_id = "llama-3.1-70b-vision"
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": "What's in this image?"}, 
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
                ]}
            ]
            
            # First call should hit the API
            response1 = cached_query_groq_vision_model(model_id, messages, use_cache=True, stream=False)
            assert mock_groq_client.chat.completions.create.call_count == 1
            
            # Second call with same parameters should use cache
            response2 = cached_query_groq_vision_model(model_id, messages, use_cache=True, stream=False)
            # API call count should still be 1
            assert mock_groq_client.chat.completions.create.call_count == 1
