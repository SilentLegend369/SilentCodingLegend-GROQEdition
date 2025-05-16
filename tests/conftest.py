"""
Configuration for pytest.
"""

import os
import sys
import pytest
import json
import tempfile
import streamlit as st
from unittest.mock import MagicMock, patch

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock streamlit session state
@pytest.fixture
def mock_streamlit_session():
    """Fixture to mock streamlit session state"""
    with patch.object(st, 'session_state', {}) as mocked_state:
        yield mocked_state

# Mock Groq API client
@pytest.fixture
def mock_groq_client():
    """Fixture to mock Groq API client responses"""
    mock_client = MagicMock()
    
    # Setup mock response for chat completions
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response from mock Groq API"
    mock_response.choices[0].delta.content = "Test response from mock Groq API"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    
    # Configure the mock client
    mock_client.chat.completions.create.return_value = mock_response
    
    with patch('src.utils.get_groq_client', return_value=mock_client):
        yield mock_client

# Create a temporary directory for testing
@pytest.fixture
def temp_data_dir():
    """Fixture to create a temporary directory for test data files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

# Sample test messages
@pytest.fixture
def sample_chat_messages():
    """Fixture to provide sample chat messages for testing"""
    return [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking! How can I assist you today?"},
        {"role": "user", "content": "Tell me about Python."}
    ]

# Sample image data
@pytest.fixture
def sample_image_data():
    """Fixture to provide sample image data for testing"""
    # Create a 1x1 pixel transparent PNG in base64
    sample_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    return {
        "name": "test_image.png",
        "type": "image/png",
        "content": sample_base64
    }
