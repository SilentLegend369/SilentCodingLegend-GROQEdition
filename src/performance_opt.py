"""
Performance optimizations for SilentCodingLegend AI application.
Includes response caching, lazy loading, and background processing.
"""

import streamlit as st
import time
import os
import json
import hashlib
import threading
import queue
import functools
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
import pickle
from src.model_metrics import metrics_tracker


# ===== Response Caching =====

class ResponseCache:
    """Cache for API responses to avoid redundant calls."""
    
    def __init__(self, cache_dir: str = None, max_age_hours: int = 24):
        """
        Initialize the response cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_hours: Maximum age of cache entries in hours
        """
        self.max_age = timedelta(hours=max_age_hours)
        
        # Create cache directory if it doesn't exist
        if cache_dir is None:
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
        else:
            self.cache_dir = cache_dir
            
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize in-memory cache
        self.memory_cache = {}
        
    def get_cache_key(self, model_id: str, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """
        Generate a unique cache key for an API request.
        
        Args:
            model_id: The model identifier
            messages: The messages to send to the API
            temperature: The temperature parameter
            max_tokens: The max_tokens parameter
            
        Returns:
            A unique hash string representing this request
            
        Note:
            For template-specific caching, template_id is appended to this hash in the calling functions
        """
        # Create a string representation of the request parameters
        key_data = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Convert to a stable string and hash it
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached response or None if not found or expired
        """
        # First check memory cache
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if datetime.now() - entry["timestamp"] <= self.max_age:
                return entry["data"]
            else:
                # Expired, remove from memory cache
                del self.memory_cache[key]
        
        # Then check disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    entry = json.load(f)
                
                # Check if entry is still valid
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if datetime.now() - timestamp <= self.max_age:
                    # Update memory cache and return
                    self.memory_cache[key] = {
                        "data": entry["data"],
                        "timestamp": timestamp
                    }
                    return entry["data"]
                else:
                    # Expired, delete file
                    os.remove(cache_file)
            except (json.JSONDecodeError, KeyError, ValueError, IOError):
                # Invalid cache entry, delete it
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                
        return None
    
    def set(self, key: str, value: Dict) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: The cache key
            value: The data to store
        """
        timestamp = datetime.now()
        
        # Update memory cache
        self.memory_cache[key] = {
            "data": value,
            "timestamp": timestamp
        }
        
        # Update disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    "timestamp": timestamp.isoformat(),
                    "data": value
                }, f)
        except IOError as e:
            st.warning(f"Failed to write to cache: {e}")
    
    def clear(self, max_age_hours: Optional[int] = None) -> int:
        """
        Clear expired entries from the cache.
        
        Args:
            max_age_hours: Override the default max age
            
        Returns:
            Number of entries removed
        """
        if max_age_hours is not None:
            max_age = timedelta(hours=max_age_hours)
        else:
            max_age = self.max_age
            
        count = 0
        
        # Clear memory cache
        now = datetime.now()
        keys_to_remove = []
        for key, entry in self.memory_cache.items():
            if now - entry["timestamp"] > max_age:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.memory_cache[key]
            count += 1
            
        # Clear disk cache
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        entry = json.load(f)
                    
                    timestamp = datetime.fromisoformat(entry["timestamp"])
                    if now - timestamp > max_age:
                        os.remove(filepath)
                        count += 1
                except (json.JSONDecodeError, KeyError, ValueError, IOError):
                    # Invalid cache entry, delete it
                    os.remove(filepath)
                    count += 1
                    
        return count


# Create a singleton cache instance
response_cache = ResponseCache()


def cached_query_groq_model(
    model_id: str, 
    messages: List[Dict], 
    temperature: float = 0.7, 
    max_tokens: int = 2048, 
    stream: bool = True, 
    json_mode: bool = False, 
    top_p: float = 1.0,
    use_cache: bool = True,
    template_id: Optional[str] = None
) -> Any:
    """
    Query the Groq model with caching support.
    
    Args:
        model_id: The model to use
        messages: The messages to send to the API
        temperature: Temperature parameter
        max_tokens: Maximum tokens to generate
        stream: Whether to stream the response
        json_mode: Whether to use JSON mode
        top_p: Top-p parameter
        use_cache: Whether to use cache
        
    Returns:
        The model's response
        
    Raises:
        ValueError: For invalid parameter values
        ConnectionError: For network issues
        Various Groq API exceptions for API-related errors
    """
    import logging
    from src.utils import query_groq_model
    
    # Validate inputs
    if not model_id or not isinstance(model_id, str):
        raise ValueError("Invalid model_id provided")
    
    if not messages or not isinstance(messages, list):
        raise ValueError("Invalid messages format")
        
    # Skip cache for streaming or if caching is disabled
    if stream or not use_cache or json_mode:
        logging.info(f"Bypassing cache for model {model_id}: stream={stream}, use_cache={use_cache}, json_mode={json_mode}")
        return query_groq_model(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            json_mode=json_mode,
            top_p=top_p
        )
    
    try:
        # Generate cache key
        cache_key = response_cache.get_cache_key(model_id, messages, temperature, max_tokens)
        if template_id:
            # Append template_id to the cache key to differentiate responses for different templates
            cache_key = f"{cache_key}_{template_id}"
        
        # Start tracking metrics
        start_time = metrics_tracker.start_tracking()
        
        # Try to get from cache
        cached_response = response_cache.get(cache_key)
        if cached_response:
            logging.info(f"Cache hit for model {model_id}")
            st.toast("Using cached response", icon="✓")
            
            # Record cache hit metrics with improved token estimation
            # For cached responses, use a better estimation based on characters and tokens
            # Messages length estimation
            messages_text = ""
            for msg in messages:
                if isinstance(msg.get("content"), str):
                    messages_text += msg.get("content", "")
                elif isinstance(msg.get("content"), list):
                    for item in msg.get("content", []):
                        if isinstance(item, dict) and item.get("type") == "text":
                            messages_text += item.get("text", "")
            
            # Response length estimation
            response_text = ""
            if hasattr(cached_response, "choices") and cached_response.choices:
                for choice in cached_response.choices:
                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        response_text += choice.message.content or ""
            
            # Estimate tokens (approx. 4 chars per token for English text)
            char_per_token = 4
            input_tokens = max(1, len(messages_text) // char_per_token)
            output_tokens = max(1, len(response_text) // char_per_token)
            
            metrics_tracker.record_metrics(
                model_id=model_id,
                start_time=start_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                is_cached=True,
                query_type="text",
                success=True,
                template_id=template_id
            )
            
            return cached_response
        
        logging.info(f"Cache miss for model {model_id}")
        
        # Not in cache, query the API
        response = query_groq_model(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            json_mode=json_mode,
            top_p=top_p,
            template_id=template_id
        )
        
        # Cache the response
        response_cache.set(cache_key, response)
        
        return response
    
    except Exception as e:
        logging.error(f"Error in cached_query_groq_model: {str(e)}")
        raise


def cached_query_groq_vision_model(
    model_id: str, 
    messages: List[Dict], 
    temperature: float = 0.7, 
    max_tokens: int = 1024, 
    stream: bool = False,
    use_cache: bool = True,
    template_id: Optional[str] = None
) -> Any:
    """
    Query the Groq vision model with caching support.
    
    Args:
        model_id: The model to use
        messages: The messages to send to the API
        temperature: Temperature parameter
        max_tokens: Maximum tokens to generate
        stream: Whether to stream the response
        use_cache: Whether to use cache
        template_id: ID of the chat template being used, if any
        
    Returns:
        The model's response
        
    Raises:
        ValueError: For invalid parameter values
        ConnectionError: For network issues
        Various Groq API exceptions for API-related errors
    """
    import logging
    from src.utils import query_groq_vision_model
    
    # Validate inputs
    if not model_id or not isinstance(model_id, str):
        raise ValueError("Invalid model_id provided")
    
    if not messages or not isinstance(messages, list):
        raise ValueError("Invalid messages format")
        
    # Skip cache for streaming or if caching is disabled
    if stream or not use_cache:
        logging.info(f"Bypassing cache for vision model {model_id}: stream={stream}, use_cache={use_cache}")
        return query_groq_vision_model(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            template_id=template_id
        )
    
    try:
        # For vision models, we need to handle base64 encoded images
        # We'll modify the messages to include only image hashes instead of full base64
        messages_for_cache = []
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                content_for_cache = []
                for item in msg["content"]:
                    if item.get("type") == "image_url" and "base64" in item.get("image_url", {}).get("url", ""):
                        # Extract just the hash of the image data to avoid storing large base64 strings
                        try:
                            image_data = item["image_url"]["url"].split(",")[1]
                            image_hash = hashlib.md5(image_data.encode()).hexdigest()
                            content_for_cache.append({
                                "type": "image_url",
                                "image_url": {"hash": image_hash}
                            })
                        except (IndexError, KeyError, AttributeError) as e:
                            logging.warning(f"Error processing image for cache: {str(e)}")
                            content_for_cache.append({"type": "text", "text": "[Image processing error]"})
                    else:
                        content_for_cache.append(item)
                msg_copy = msg.copy()
                msg_copy["content"] = content_for_cache
                messages_for_cache.append(msg_copy)
            else:
                messages_for_cache.append(msg)
        
        # Generate cache key using the modified messages and template_id
        # Add template_id to the cache key to ensure different templates get different cache entries
        cache_key = response_cache.get_cache_key(model_id, messages_for_cache, temperature, max_tokens) 
        if template_id:
            # Append template_id to the cache key to differentiate responses for different templates
            cache_key = f"{cache_key}_{template_id}"
        
        # Start tracking metrics
        start_time = metrics_tracker.start_tracking()
        
        # Try to get from cache
        cached_response = response_cache.get(cache_key)
        if cached_response:
            logging.info(f"Cache hit for vision model {model_id}")
            st.toast("Using cached vision response", icon="✓")
            
            # Record cache hit metrics with improved token estimation
            # For vision models, images add substantial token count
            
            # Messages length estimation (text)
            messages_text = ""
            image_count = 0
            for msg in messages:
                if isinstance(msg.get("content"), str):
                    messages_text += msg.get("content", "")
                elif isinstance(msg.get("content"), list):
                    for item in msg.get("content", []):
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                messages_text += item.get("text", "")
                            elif item.get("type") == "image_url":
                                image_count += 1
            
            # Response length estimation
            response_text = ""
            if hasattr(cached_response, "choices") and cached_response.choices:
                for choice in cached_response.choices:
                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        response_text += choice.message.content or ""
            
            # Estimate tokens (approx. 4 chars per token for English text)
            # And approx. 850 tokens per image based on averages
            char_per_token = 4
            image_tokens_approx = 850 * image_count
            input_tokens = max(1, (len(messages_text) // char_per_token) + image_tokens_approx)
            output_tokens = max(1, len(response_text) // char_per_token)
            
            metrics_tracker.record_metrics(
                model_id=model_id,
                start_time=start_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                is_cached=True,
                query_type="vision",
                success=True,
                template_id=template_id
            )
            
            return cached_response
        
        logging.info(f"Cache miss for vision model {model_id}")
        
        # Not in cache, query the API
        response = query_groq_vision_model(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            template_id=template_id
        )
        
        # Cache the response
        response_cache.set(cache_key, response)
        
        return response
        
    except Exception as e:
        logging.error(f"Error in cached_query_groq_vision_model: {str(e)}")
        raise


def setup_cache_management():
    """Set up the cache management UI."""
    st.header("⚡ Response Caching")
    
    # Show cache statistics
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')] if os.path.exists(cache_dir) else []
    memory_cache_size = len(response_cache.memory_cache)
    
    # Cache statistics
    st.markdown(f"""
        <div style="padding: 15px; border-radius: 8px; background-color: #2d2d2d; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #7e57c2;">Cache Statistics</h3>
            <p><b>Disk cache entries:</b> {len(cache_files)}</p>
            <p><b>Memory cache entries:</b> {memory_cache_size}</p>
            <p><b>Cache directory:</b> <code>{cache_dir}</code></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Cache settings
    col1, col2 = st.columns(2)
    
    with col1:
        current_max_age = st.session_state.get("cache_max_age", 24)
        new_max_age = st.number_input(
            "Cache expiration (hours)",
            min_value=1,
            max_value=168,  # 7 days
            value=current_max_age,
            help="Maximum age of cached responses in hours"
        )
        
        if new_max_age != current_max_age:
            st.session_state.cache_max_age = new_max_age
            response_cache.max_age = timedelta(hours=new_max_age)
            st.success(f"Cache expiration set to {new_max_age} hours")
    
    with col2:
        cache_enabled = st.toggle(
            "Enable response caching",
            value=st.session_state.get("cache_enabled", True),
            help="Toggle response caching on/off"
        )
        
        if "cache_enabled" not in st.session_state or cache_enabled != st.session_state.cache_enabled:
            st.session_state.cache_enabled = cache_enabled
            status = "enabled" if cache_enabled else "disabled"
            st.success(f"Response caching {status}")
    
    # Cache actions
    st.subheader("Cache Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Expired Cache Entries", type="secondary"):
            removed = response_cache.clear()
            st.success(f"Removed {removed} expired cache entries")
    
    with col2:
        if st.button("Clear All Cache", type="secondary"):
            # Clear memory cache
            response_cache.memory_cache = {}
            
            # Clear disk cache
            if os.path.exists(cache_dir):
                for filename in os.listdir(cache_dir):
                    if filename.endswith(".json"):
                        os.remove(os.path.join(cache_dir, filename))
            
            st.success("Cache cleared successfully")
    
    # Cache monitoring
    if cache_files:
        with st.expander("View Cache Contents", expanded=False):
            # Get information about cache files
            cache_info = []
            for filename in cache_files[:20]:  # Limit to 20 entries
                filepath = os.path.join(cache_dir, filename)
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    size = os.path.getsize(filepath) / 1024  # size in KB
                    
                    # Try to get the model and first few tokens of the prompt
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    model = data["data"].get("model", "unknown")
                    
                    # Try to get a snippet of the prompt
                    prompt_snippet = ""
                    if "messages" in data["data"].get("choices", [{}])[0].get("message", {}):
                        messages = data["data"]["choices"][0]["message"]["messages"]
                        for msg in messages:
                            if msg.get("role") == "user":
                                content = msg.get("content", "")
                                if isinstance(content, str):
                                    prompt_snippet = content[:50] + "..." if len(content) > 50 else content
                                    break
                    
                    cache_info.append({
                        "key": filename.replace(".json", ""),
                        "model": model,
                        "prompt": prompt_snippet,
                        "age": (datetime.now() - mtime).total_seconds() / 3600,  # age in hours
                        "size": f"{size:.2f} KB",
                        "time": mtime.strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception as e:
                    cache_info.append({
                        "key": filename.replace(".json", ""),
                        "model": "error",
                        "prompt": str(e),
                        "age": 0,
                        "size": "0 KB",
                        "time": "unknown"
                    })
            
            # Create a table
            if cache_info:
                st.markdown("### Recent Cache Entries")
                st.dataframe(
                    cache_info,
                    column_config={
                        "key": "Cache Key",
                        "model": "Model",
                        "prompt": "Prompt Snippet",
                        "age": st.column_config.NumberColumn("Age (hours)", format="%.1f"),
                        "size": "Size",
                        "time": "Timestamp"
                    },
                    hide_index=True
                )
                
                if len(cache_files) > 20:
                    st.info(f"Showing 20 of {len(cache_files)} cache entries")
            else:
                st.info("No cache entries found")


# ===== Lazy Loading =====

class LazyLoadManager:
    """Manager for lazy loading UI elements and data."""
    
    def __init__(self):
        """Initialize the lazy load manager."""
        # Track loaded sections
        if "loaded_sections" not in st.session_state:
            st.session_state.loaded_sections = set()
    
    def is_section_loaded(self, section_id: str) -> bool:
        """
        Check if a section has been loaded.
        
        Args:
            section_id: Unique identifier for the section
            
        Returns:
            True if the section has been loaded, False otherwise
        """
        return section_id in st.session_state.loaded_sections
    
    def mark_section_loaded(self, section_id: str) -> None:
        """
        Mark a section as loaded.
        
        Args:
            section_id: Unique identifier for the section
        """
        st.session_state.loaded_sections.add(section_id)
    
    def reset_section(self, section_id: str) -> None:
        """
        Reset a section's loaded state.
        
        Args:
            section_id: Unique identifier for the section
        """
        if section_id in st.session_state.loaded_sections:
            st.session_state.loaded_sections.remove(section_id)
    
    def reset_all(self) -> None:
        """Reset all loaded sections."""
        st.session_state.loaded_sections = set()


# Create a singleton lazy load manager
lazy_load_manager = LazyLoadManager()


def lazy_load(section_id: str, loading_message: str = "Loading..."):
    """
    Decorator for lazy loading functions.
    
    Args:
        section_id: Unique identifier for the section
        loading_message: Message to display while loading
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not lazy_load_manager.is_section_loaded(section_id):
                with st.spinner(loading_message):
                    result = func(*args, **kwargs)
                lazy_load_manager.mark_section_loaded(section_id)
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def setup_lazy_loading_demo():
    """Set up a demo of lazy loading functionality."""
    st.header("🚀 Lazy Loading")
    
    st.markdown("""
        Lazy loading improves performance by only loading resources when needed. 
        This demo shows how sections are loaded only when requested.
    """)
    
    # Demo control buttons
    col1, col2, col3 = st.columns(3)
    
    section_ids = ["demo_section_1", "demo_section_2", "demo_section_3"]
    section_names = ["Heavy Resource 1", "Heavy Resource 2", "Heavy Resource 3"]
    
    # Reset button
    if col1.button("Reset All Sections"):
        lazy_load_manager.reset_all()
        st.rerun()
    
    # Individual load buttons
    load_buttons = []
    for i, (section_id, section_name) in enumerate(zip(section_ids, section_names)):
        col = [col1, col2, col3][i]
        loaded = lazy_load_manager.is_section_loaded(section_id)
        button_text = f"{section_name} ✓" if loaded else f"Load {section_name}"
        button_type = "secondary" if loaded else "primary"
        button = col.button(button_text, key=f"load_{section_id}", type=button_type)
        load_buttons.append((button, section_id, loaded))
    
    # Handle button clicks
    for button, section_id, loaded in load_buttons:
        if button and not loaded:
            with st.spinner(f"Loading {section_id}..."):
                # Simulate loading time
                time.sleep(1)
                lazy_load_manager.mark_section_loaded(section_id)
            st.rerun()
    
    # Display loaded sections
    st.subheader("Loaded Sections")
    
    if not st.session_state.loaded_sections:
        st.info("No sections are currently loaded. Click the buttons above to load sections.")
    else:
        for i, (section_id, section_name) in enumerate(zip(section_ids, section_names)):
            if lazy_load_manager.is_section_loaded(section_id):
                with st.container(border=True):
                    st.markdown(f"### {section_name}")
                    st.markdown(f"This is the content for {section_name}.")
                    st.markdown("This section was lazy-loaded only when requested.")
                    
                    # Add a reset button for this section
                    if st.button(f"Reset this section", key=f"reset_{section_id}"):
                        lazy_load_manager.reset_section(section_id)
                        st.rerun()


# ===== Background Processing =====

class BackgroundTaskManager:
    """Manager for background tasks."""
    
    def __init__(self):
        """Initialize the background task manager."""
        # Tasks dictionary: task_id -> task_info
        if "bg_tasks" not in st.session_state:
            st.session_state.bg_tasks = {}
        
        # Results dictionary: task_id -> result
        if "bg_results" not in st.session_state:
            st.session_state.bg_results = {}
            
        # Task queue for pending tasks
        self.task_queue = queue.Queue()
        
        # Start worker thread if not already running
        if not hasattr(self, 'worker_thread') or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
    
    def submit_task(self, func: Callable, args: tuple = None, kwargs: dict = None, 
                   task_name: str = "Background Task") -> str:
        """
        Submit a task for background execution.
        
        Args:
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            task_name: Human-readable name for the task
            
        Returns:
            task_id: Unique identifier for the task
        """
        args = args or ()
        kwargs = kwargs or {}
        
        # Generate a unique task ID
        task_id = f"task_{int(time.time())}_{hash(func)}"
        
        # Store task info
        task_info = {
            "id": task_id,
            "name": task_name,
            "status": "queued",
            "start_time": time.time(),
            "end_time": None,
            "progress": 0,
            "message": "Queued"
        }
        st.session_state.bg_tasks[task_id] = task_info
        
        # Add to queue
        self.task_queue.put((task_id, func, args, kwargs))
        
        return task_id
    
    def _worker_loop(self):
        """Worker thread that processes tasks from the queue."""
        while True:
            try:
                # Get a task from the queue
                task_id, func, args, kwargs = self.task_queue.get()
                
                # Update task status
                if task_id in st.session_state.bg_tasks:
                    st.session_state.bg_tasks[task_id]["status"] = "running"
                    st.session_state.bg_tasks[task_id]["message"] = "Processing"
                
                # Execute the task
                try:
                    result = func(*args, **kwargs)
                    
                    # Store the result
                    st.session_state.bg_results[task_id] = result
                    
                    # Update task status
                    if task_id in st.session_state.bg_tasks:
                        st.session_state.bg_tasks[task_id]["status"] = "completed"
                        st.session_state.bg_tasks[task_id]["end_time"] = time.time()
                        st.session_state.bg_tasks[task_id]["progress"] = 100
                        st.session_state.bg_tasks[task_id]["message"] = "Completed"
                        
                except Exception as e:
                    # Update task status with error
                    if task_id in st.session_state.bg_tasks:
                        st.session_state.bg_tasks[task_id]["status"] = "failed"
                        st.session_state.bg_tasks[task_id]["end_time"] = time.time()
                        st.session_state.bg_tasks[task_id]["message"] = f"Error: {str(e)}"
                
                # Mark task as done in the queue
                self.task_queue.task_done()
                
            except Exception:
                # Log but don't crash the worker thread
                pass
    
    def get_task_info(self, task_id: str) -> Optional[Dict]:
        """
        Get information about a task.
        
        Args:
            task_id: The task identifier
            
        Returns:
            Dictionary with task information or None if not found
        """
        return st.session_state.bg_tasks.get(task_id)
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get the result of a completed task.
        
        Args:
            task_id: The task identifier
            
        Returns:
            Task result or None if not completed or found
        """
        return st.session_state.bg_results.get(task_id)
    
    def clear_completed_tasks(self, max_age_seconds: int = 3600) -> int:
        """
        Clear completed tasks older than the specified age.
        
        Args:
            max_age_seconds: Maximum age of completed tasks in seconds
            
        Returns:
            Number of tasks cleared
        """
        now = time.time()
        task_ids_to_remove = []
        
        for task_id, task_info in st.session_state.bg_tasks.items():
            if task_info["status"] in ["completed", "failed"]:
                end_time = task_info.get("end_time", 0)
                if now - end_time > max_age_seconds:
                    task_ids_to_remove.append(task_id)
        
        # Remove the tasks
        for task_id in task_ids_to_remove:
            if task_id in st.session_state.bg_tasks:
                del st.session_state.bg_tasks[task_id]
            if task_id in st.session_state.bg_results:
                del st.session_state.bg_results[task_id]
        
        return len(task_ids_to_remove)


# Create a singleton task manager
bg_task_manager = BackgroundTaskManager()


def update_progress(task_id: str, progress: float, message: str = None) -> None:
    """
    Update the progress of a background task.
    
    Args:
        task_id: The task identifier
        progress: Progress value (0-100)
        message: Optional status message
    """
    if task_id in st.session_state.bg_tasks:
        st.session_state.bg_tasks[task_id]["progress"] = min(100, max(0, progress))
        if message is not None:
            st.session_state.bg_tasks[task_id]["message"] = message


def setup_background_processing():
    """Set up the background processing UI."""
    st.header("⏱️ Background Processing")
    
    st.markdown("""
        Background processing allows long-running tasks to execute without blocking the UI.
        This demo shows how tasks can be submitted and monitored.
    """)
    
    # Show active tasks
    active_tasks = {tid: task for tid, task in st.session_state.bg_tasks.items() 
                   if task["status"] in ["queued", "running"]}
    
    completed_tasks = {tid: task for tid, task in st.session_state.bg_tasks.items() 
                      if task["status"] in ["completed", "failed"]}
    
    st.subheader("Task Management")
    
    col1, col2, col3 = st.columns(3)
    
    # Demo task creation
    with col1:
        task_type = st.selectbox(
            "Task Type",
            ["Quick Task (2s)", "Medium Task (5s)", "Long Task (10s)"]
        )
        
        if st.button("Start New Task", type="primary"):
            if task_type == "Quick Task (2s)":
                duration = 2
            elif task_type == "Medium Task (5s)":
                duration = 5
            else:
                duration = 10
                
            task_id = bg_task_manager.submit_task(
                func=simulate_long_task,
                args=(duration,),
                task_name=task_type
            )
            
            st.success(f"Task submitted (ID: {task_id})")
            st.rerun()
    
    # Clear completed tasks
    with col2:
        if st.button("Clear Completed Tasks", disabled=len(completed_tasks) == 0):
            cleared = bg_task_manager.clear_completed_tasks()
            st.success(f"Cleared {cleared} completed tasks")
            st.rerun()
    
    # Task statistics
    with col3:
        st.metric("Active Tasks", len(active_tasks))
        st.metric("Completed Tasks", len(completed_tasks))
    
    # Display active tasks
    if active_tasks:
        st.subheader("Active Tasks")
        
        for task_id, task_info in active_tasks.items():
            with st.container(border=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**{task_info['name']}** ({task_id})")
                    
                    # Progress bar
                    st.progress(task_info["progress"] / 100)
                    
                    # Status message
                    st.caption(f"Status: {task_info['message']}")
                
                with col2:
                    # Show elapsed time
                    elapsed = time.time() - task_info["start_time"]
                    st.write(f"Elapsed: {elapsed:.1f}s")
                    
                    # Status badge
                    status_color = {
                        "queued": "blue",
                        "running": "orange",
                        "completed": "green",
                        "failed": "red"
                    }.get(task_info["status"], "gray")
                    
                    st.markdown(f"<span style='background-color: {status_color}; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8em;'>{task_info['status'].upper()}</span>", unsafe_allow_html=True)
    else:
        st.info("No active tasks. Start a task from the menu above.")
    
    # Display completed tasks
    if completed_tasks:
        with st.expander("Completed Tasks", expanded=False):
            for task_id, task_info in list(completed_tasks.items())[-5:]:  # Show last 5
                status_color = "green" if task_info["status"] == "completed" else "red"
                duration = task_info.get("end_time", time.time()) - task_info["start_time"]
                
                st.markdown(f"""
                    <div style="padding: 8px; margin-bottom: 8px; border-left: 3px solid {status_color}; background-color: #2d2d2d;">
                        <div style="display: flex; justify-content: space-between;">
                            <div><strong>{task_info['name']}</strong></div>
                            <div><span style='background-color: {status_color}; color: white; padding: 2px 6px; border-radius: 10px; font-size: 0.8em;'>{task_info['status'].upper()}</span></div>
                        </div>
                        <div style="color: #bbbbbb; font-size: 0.9em; margin-top: 5px;">
                            Duration: {duration:.2f}s | Message: {task_info['message']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # If the task has a result and was completed successfully
                if task_info["status"] == "completed":
                    result = bg_task_manager.get_task_result(task_id)
                    if result is not None:
                        st.json(result)


def simulate_long_task(duration_seconds: int) -> Dict:
    """
    Simulate a long-running task with progress updates.
    
    Args:
        duration_seconds: Duration of the task in seconds
        
    Returns:
        Dictionary with result information
    """
    # Get the task_id from the current thread context
    thread = threading.current_thread()
    thread_name = thread.name
    
    # Find this task in the tasks dictionary
    task_id = None
    for tid, task in st.session_state.bg_tasks.items():
        if task["status"] == "running" and "worker" in thread_name.lower():
            task_id = tid
            break
    
    # Simulate work with progress updates
    steps = 10
    step_duration = duration_seconds / steps
    
    for i in range(steps):
        time.sleep(step_duration)
        progress = (i + 1) * 100 / steps
        
        if task_id:
            update_progress(task_id, progress, f"Processing step {i+1}/{steps}")
    
    # Return some result data
    return {
        "success": True,
        "duration": duration_seconds,
        "timestamp": time.time(),
        "result_value": duration_seconds * 10
    }


# ===== Optimization Management =====

def setup_performance_optimization():
    """Set up the performance optimization UI."""
    st.title("🚀 Performance Optimizations")
    
    st.markdown("""
        This page provides tools to optimize the performance of SilentCodingLegend AI.
        You can configure caching, lazy loading, and background processing.
    """)
    
    tab1, tab2, tab3 = st.tabs([
        "⚡ Response Caching", 
        "🚀 Lazy Loading", 
        "⏱️ Background Processing"
    ])
    
    with tab1:
        setup_cache_management()
        
    with tab2:
        setup_lazy_loading_demo()
        
    with tab3:
        setup_background_processing()
