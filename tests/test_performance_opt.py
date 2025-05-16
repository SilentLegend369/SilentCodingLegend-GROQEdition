"""
Tests for the performance_opt.py module.
"""

import pytest
import os
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import streamlit as st

from src.performance_opt import (
    ResponseCache, 
    lazy_load_manager,
    bg_task_manager,
    update_progress,
    lazy_load
)

# Test ResponseCache
class TestResponseCache:
    
    def test_cache_key_generation(self):
        """Test cache key generation produces consistent keys"""
        cache = ResponseCache()
        model_id = "test-model"
        messages = [{"role": "user", "content": "Hello"}]
        
        # Generate keys and ensure they're consistent for the same input
        key1 = cache.get_cache_key(model_id, messages, 0.7, 1000)
        key2 = cache.get_cache_key(model_id, messages, 0.7, 1000)
        assert key1 == key2
        
        # Different parameters should produce different keys
        key3 = cache.get_cache_key(model_id, messages, 0.8, 1000)
        assert key1 != key3
        
        key4 = cache.get_cache_key("other-model", messages, 0.7, 1000)
        assert key1 != key4
    
    def test_memory_cache(self):
        """Test in-memory caching works correctly"""
        cache = ResponseCache()
        key = "test_key"
        value = {"data": "test_value"}
        
        # Set value in cache and retrieve it
        cache.set(key, value)
        result = cache.get(key)
        assert result == value
        assert key in cache.memory_cache
        
        # Test cache expiration
        with patch('datetime.datetime') as mock_datetime:
            mock_now = datetime.now() + timedelta(hours=25)
            mock_datetime.now.return_value = mock_now
            
            result = cache.get(key)
            assert result is None
            assert key not in cache.memory_cache
    
    def test_disk_cache(self, temp_data_dir):
        """Test disk-based caching works correctly"""
        cache = ResponseCache(cache_dir=temp_data_dir)
        key = "test_key"
        value = {"data": "test_value"}
        
        # Set value in cache
        cache.set(key, value)
        
        # Clear memory cache to force disk reading
        cache.memory_cache = {}
        
        # Verify file exists
        assert os.path.exists(os.path.join(temp_data_dir, f"{key}.json"))
        
        # Get value from disk cache
        result = cache.get(key)
        assert result == value
        assert key in cache.memory_cache  # Should be loaded into memory
        
        # Test cache expiration
        with patch('datetime.datetime') as mock_datetime:
            mock_now = datetime.now() + timedelta(hours=25)
            mock_datetime.now.return_value = mock_now
            
            # Clear memory cache again
            cache.memory_cache = {}
            
            result = cache.get(key)
            assert result is None
            assert not os.path.exists(os.path.join(temp_data_dir, f"{key}.json"))
    
    def test_clear_cache(self, temp_data_dir):
        """Test cache clearing functionality"""
        cache = ResponseCache(cache_dir=temp_data_dir)
        
        # Set multiple cache entries
        for i in range(5):
            key = f"test_key_{i}"
            value = {"data": f"test_value_{i}"}
            cache.set(key, value)
        
        assert len(cache.memory_cache) == 5
        assert len(os.listdir(temp_data_dir)) == 5
        
        # Clear cache with default expiry (should not remove anything fresh)
        removed = cache.clear()
        assert removed == 0
        assert len(cache.memory_cache) == 5
        
        # Clear cache with 0 hour expiry (should remove everything)
        removed = cache.clear(max_age_hours=0)
        assert removed == 5
        assert len(cache.memory_cache) == 0
        assert len(os.listdir(temp_data_dir)) == 0

# Test LazyLoadManager
class TestLazyLoadManager:
    
    def test_section_tracking(self):
        """Test section loaded state tracking"""
        # Reset session state
        if hasattr(st, 'session_state'):
            st.session_state.loaded_sections = set()
            
        # Test initial state
        assert not lazy_load_manager.is_section_loaded("test_section")
        
        # Mark section as loaded
        lazy_load_manager.mark_section_loaded("test_section")
        assert lazy_load_manager.is_section_loaded("test_section")
        
        # Reset section
        lazy_load_manager.reset_section("test_section")
        assert not lazy_load_manager.is_section_loaded("test_section")
        
        # Mark multiple sections loaded
        lazy_load_manager.mark_section_loaded("section1")
        lazy_load_manager.mark_section_loaded("section2")
        assert lazy_load_manager.is_section_loaded("section1")
        assert lazy_load_manager.is_section_loaded("section2")
        
        # Reset all sections
        lazy_load_manager.reset_all()
        assert not lazy_load_manager.is_section_loaded("section1")
        assert not lazy_load_manager.is_section_loaded("section2")
    
    def test_lazy_load_decorator(self):
        """Test the lazy_load decorator functionality"""
        # Reset session state
        if hasattr(st, 'session_state'):
            st.session_state.loaded_sections = set()
            
        # Track function calls
        call_count = 0
        
        # Define a function with the lazy_load decorator
        @lazy_load("test_function")
        def test_function():
            nonlocal call_count
            call_count += 1
            return "result"
        
        # Call function for the first time - should mark as loaded
        with patch('streamlit.spinner'):
            result = test_function()
            
        assert result == "result"
        assert call_count == 1
        assert lazy_load_manager.is_section_loaded("test_function")
        
        # Call function again - should not increment count or show spinner
        with patch('streamlit.spinner') as mock_spinner:
            result = test_function()
            
        assert result == "result"
        assert call_count == 2  # Function is still called
        assert lazy_load_manager.is_section_loaded("test_function")
        mock_spinner.assert_not_called()

# Test BackgroundTaskManager
class TestBackgroundTaskManager:
    
    def test_submit_task(self):
        """Test task submission and tracking"""
        # Reset session state
        if hasattr(st, 'session_state'):
            st.session_state.bg_tasks = {}
            st.session_state.bg_results = {}
        
        # Mock function to run as a task
        def test_task(arg1, arg2=None):
            return f"{arg1}-{arg2}"
        
        # Submit a task
        task_id = bg_task_manager.submit_task(
            test_task, 
            args=("value1",), 
            kwargs={"arg2": "value2"}, 
            task_name="Test Task"
        )
        
        # Verify task was added to session state
        assert task_id in st.session_state.bg_tasks
        assert st.session_state.bg_tasks[task_id]["status"] == "queued"
        assert st.session_state.bg_tasks[task_id]["name"] == "Test Task"
    
    def test_update_progress(self):
        """Test task progress updates"""
        # Reset session state
        if hasattr(st, 'session_state'):
            st.session_state.bg_tasks = {
                "test_task": {
                    "status": "running",
                    "progress": 0,
                    "message": "Initial"
                }
            }
            
        # Update progress
        update_progress("test_task", 50, "Halfway done")
        
        # Verify progress was updated
        assert st.session_state.bg_tasks["test_task"]["progress"] == 50
        assert st.session_state.bg_tasks["test_task"]["message"] == "Halfway done"
        
        # Test progress limits
        update_progress("test_task", 120, "Too high")
        assert st.session_state.bg_tasks["test_task"]["progress"] == 100
        
        update_progress("test_task", -10, "Too low")
        assert st.session_state.bg_tasks["test_task"]["progress"] == 0
