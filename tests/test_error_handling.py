"""
Tests for error handling and edge cases.
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock

import streamlit as st
import groq
# For backward compatibility
AuthenticationError = groq.AuthenticationError if hasattr(groq, 'AuthenticationError') else Exception
InvalidRequestError = groq.InvalidRequestError if hasattr(groq, 'InvalidRequestError') else Exception
APIError = groq.APIError if hasattr(groq, 'APIError') else Exception

from src.utils import (
    get_groq_client,
    query_groq_model,
    query_groq_vision_model,
    is_valid_file_extension,
    is_valid_image_extension,
    save_uploaded_file,
    save_uploaded_image
)

# Test error handling in API client
class TestApiErrorHandling:
    
    def test_authentication_error_handling(self):
        """Test handling of authentication errors from Groq API"""
        with patch('groq.Client') as mock_client:
            # Configure mock to raise AuthenticationError
            mock_instance = mock_client.return_value
            mock_instance.chat.completions.create.side_effect = AuthenticationError(
                message="Invalid API key",
                http_status=401,
                request_id="req_123"
            )
            
            # Patch config to have a "valid" API key
            with patch('src.config.GROQ_API_KEY', 'fake_api_key'), \
                 patch('streamlit.error') as mock_error:
                
                # Test that error is handled properly
                try:
                    client = get_groq_client()
                    response = query_groq_model(
                        model_id="llama-3.1-70b",
                        messages=[{"role": "user", "content": "Hello"}]
                    )
                    assert False, "Should have raised an exception"
                except AuthenticationError as e:
                    assert "Invalid API key" in str(e)
    
    def test_invalid_request_error_handling(self):
        """Test handling of invalid request errors from Groq API"""
        with patch('src.utils.get_groq_client') as mock_get_client:
            # Configure mock to raise InvalidRequestError
            mock_client = mock_get_client.return_value
            mock_client.chat.completions.create.side_effect = InvalidRequestError(
                message="Invalid model ID specified",
                http_status=400,
                request_id="req_456"
            )
            
            # Test that error is handled properly
            try:
                response = query_groq_model(
                    model_id="invalid-model",
                    messages=[{"role": "user", "content": "Hello"}]
                )
                assert False, "Should have raised an exception"
            except InvalidRequestError as e:
                assert "Invalid model ID" in str(e)
    
    def test_api_timeout_error_handling(self):
        """Test handling of API timeout errors"""
        with patch('src.utils.get_groq_client') as mock_get_client:
            # Configure mock to raise Timeout error
            mock_client = mock_get_client.return_value
            mock_client.chat.completions.create.side_effect = APIError(
                message="Request timed out",
                http_status=408,
                request_id="req_789"
            )
            
            # Test that error is handled properly
            try:
                response = query_groq_model(
                    model_id="llama-3.1-70b",
                    messages=[{"role": "user", "content": "Hello"}]
                )
                assert False, "Should have raised an exception"
            except APIError as e:
                assert "Request timed out" in str(e)

# Test edge cases in file handling
class TestFileHandlingEdgeCases:
    
    def test_empty_file_handling(self, tmp_path):
        """Test handling of empty files"""
        with patch('src.config.UPLOAD_PATH', str(tmp_path)):
            mock_file = MagicMock()
            mock_file.name = "empty.txt"
            mock_file.getvalue.return_value = b""
            
            file_path = save_uploaded_file(mock_file)
            
            assert os.path.exists(file_path)
            assert os.path.getsize(file_path) == 0
    
    def test_invalid_file_extensions(self):
        """Test invalid file extension handling"""
        with patch('src.config.ALLOWED_EXTENSIONS', ['txt', 'pdf']), \
             patch('src.config.ALLOWED_IMAGE_EXTENSIONS', ['jpg', 'png']):
            
            # Test text file extensions
            assert is_valid_file_extension('document.txt') == True
            assert is_valid_file_extension('document.docx') == False
            assert is_valid_file_extension('.htaccess') == False
            assert is_valid_file_extension('file.txt.exe') == False
            
            # Test image file extensions
            assert is_valid_image_extension('image.jpg') == True
            assert is_valid_image_extension('image.png') == True
            assert is_valid_image_extension('image.gif') == False
            assert is_valid_image_extension('image.jpg.php') == False
    
    def test_large_file_handling(self, tmp_path):
        """Test handling of extremely large files"""
        with patch('src.config.UPLOAD_PATH', str(tmp_path)):
            # Create a mock file with 10MB of data
            mock_file = MagicMock()
            mock_file.name = "large.txt"
            mock_file.getvalue.return_value = b"0" * (10 * 1024 * 1024)  # 10MB of zeros
            
            file_path = save_uploaded_file(mock_file)
            
            assert os.path.exists(file_path)
            assert os.path.getsize(file_path) == 10 * 1024 * 1024
