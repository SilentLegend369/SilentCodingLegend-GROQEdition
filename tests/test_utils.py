"""
Tests for the utils.py module.
"""

import pytest
import os
import json
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock

import streamlit as st
import groq

from src.utils import (
    get_groq_client,
    is_valid_file_extension,
    create_directories,
    save_uploaded_file,
    backup_chat_history,
    query_groq_model,
    get_system_prompt
)

# Test get_groq_client
def test_get_groq_client_with_api_key():
    """Test get_groq_client when API key is available"""
    with patch('src.config.GROQ_API_KEY', 'fake_api_key'), \
         patch('groq.Client') as mock_client:
        client = get_groq_client()
        mock_client.assert_called_once_with(api_key='fake_api_key')
        assert client == mock_client.return_value

def test_get_groq_client_without_api_key():
    """Test get_groq_client when no API key is available"""
    with patch('src.config.GROQ_API_KEY', ''), \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.stop') as mock_stop:
        with pytest.raises(SystemExit):
            get_groq_client()
            mock_error.assert_called_once()
            mock_stop.assert_called_once()

# Test is_valid_file_extension
def test_is_valid_file_extension():
    """Test is_valid_file_extension with various file extensions"""
    with patch('src.utils.ALLOWED_EXTENSIONS', ['txt', 'pdf', 'docx']):
        assert is_valid_file_extension('document.txt') == True
        assert is_valid_file_extension('document.pdf') == True
        assert is_valid_file_extension('document.docx') == True
        assert is_valid_file_extension('document.jpg') == False
        assert is_valid_file_extension('document') == False
        assert is_valid_file_extension('') == False
        assert is_valid_file_extension(None) == False

# Test create_directories
def test_create_directories():
    """Test create_directories creates all necessary directories"""
    with patch('os.makedirs') as mock_makedirs, \
         patch('src.config.ASSETS_PATH', '/fake/assets'), \
         patch('src.config.UPLOAD_PATH', '/fake/uploads'), \
         patch('src.config.KNOWLEDGE_PATH', '/fake/knowledge'), \
         patch('src.config.VECTOR_DB_PATH', '/fake/vector_db'), \
         patch('src.config.CHAT_HISTORY_PATH', '/fake/chat_history'), \
         patch('src.config.IMAGE_UPLOAD_PATH', '/fake/images'):
        
        create_directories()
        assert mock_makedirs.call_count == 6
        mock_makedirs.assert_any_call('/fake/assets', exist_ok=True)
        mock_makedirs.assert_any_call('/fake/uploads', exist_ok=True)
        mock_makedirs.assert_any_call('/fake/knowledge', exist_ok=True)
        mock_makedirs.assert_any_call('/fake/vector_db', exist_ok=True)
        mock_makedirs.assert_any_call('/fake/chat_history', exist_ok=True)
        mock_makedirs.assert_any_call('/fake/images', exist_ok=True)

# Test save_uploaded_file
def test_save_uploaded_file(tmp_path):
    """Test save_uploaded_file saves the content to a temporary file"""
    with patch('src.config.UPLOAD_PATH', str(tmp_path)):
        mock_file = MagicMock()
        mock_file.name = "test.txt"
        mock_file.getvalue.return_value = b"Test content"
        
        file_path = save_uploaded_file(mock_file)
        
        assert os.path.exists(file_path)
        assert os.path.basename(file_path).endswith("_test.txt")
        with open(file_path, 'rb') as f:
            content = f.read()
        assert content == b"Test content"
        
        # Clean up
        os.remove(file_path)

# Test backup_chat_history
def test_backup_chat_history(tmp_path):
    """Test backup_chat_history saves the chat history to a file"""
    with patch('src.utils.CHAT_HISTORY_PATH', str(tmp_path)), \
         patch.object(st, 'session_state', {'chat_session_id': 'test_session'}):
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        backup_path = backup_chat_history(messages, "test_doc", "test_model")
        
        assert os.path.exists(backup_path)
        with open(backup_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert data["metadata"]["document"] == "test_doc"
        assert data["metadata"]["model"] == "test_model"
        assert data["metadata"]["session_id"] == "test_session"
        assert "messages" in data
        assert data["messages"] == messages
        
        # Clean up
        os.remove(backup_path)

# Test query_groq_model
def test_query_groq_model(mock_groq_client):
    """Test query_groq_model calls the API correctly"""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Test with default parameters
    response = query_groq_model("model-id", messages)
    
    mock_groq_client.chat.completions.create.assert_called_with(
        model="model-id",
        messages=messages,
        temperature=0.7,
        max_completion_tokens=2048,
        stream=True,
        top_p=1.0
    )
    
    # Test with custom parameters
    query_groq_model(
        "model-id",
        messages,
        temperature=0.5,
        max_tokens=1000,
        stream=False,
        json_mode=True,
        top_p=0.8
    )
    
    mock_groq_client.chat.completions.create.assert_called_with(
        model="model-id",
        messages=messages,
        temperature=0.5,
        max_completion_tokens=1000,
        stream=False,
        top_p=0.8,
        response_format={"type": "json_object"}
    )

# Test get_system_prompt
def test_get_system_prompt():
    """Test get_system_prompt returns the correct prompt"""
    with patch('src.utils.DEFAULT_SYSTEM_PROMPT', 'Base prompt'), \
         patch('src.utils.JSON_SYSTEM_PROMPT_ADDITION', ' JSON addition'):
        
        # Test without JSON mode
        assert get_system_prompt() == 'Base prompt'
        
        # Test with JSON mode
        assert get_system_prompt(json_mode=True) == 'Base prompt JSON addition'
