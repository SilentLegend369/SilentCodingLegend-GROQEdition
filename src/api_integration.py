"""
API Integration functionality for SilentCodingLegend AI.
This module provides API key management, testing tools, and documentation.
"""

import streamlit as st
import time
import os
import uuid
import json
import base64
import tempfile
import logging
from typing import List, Dict, Any, Optional, Callable, TypeVar, Union
from functools import wraps

import groq

# Setup logging
logger = logging.getLogger(__name__)

# Type definitions for improved type checking
T = TypeVar('T')
ResponseType = Union[Dict[str, Any], List[Dict[str, Any]]]


def save_api_keys(keys_list: list) -> bool:
    """
    Save API keys to a persistent JSON file.
    
    Args:
        keys_list: List of API key dictionaries
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "api")
        os.makedirs(api_dir, exist_ok=True)
        
        # Write to file
        keys_file = os.path.join(api_dir, "api_keys.json")
        with open(keys_file, 'w') as f:
            json.dump(keys_list, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Failed to save API keys: {str(e)}")
        return False


def load_api_keys() -> list:
    """
    Load API keys from persistent storage.
    
    Returns:
        list: List of API key dictionaries
    """
    try:
        # Check if file exists
        api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "api")
        keys_file = os.path.join(api_dir, "api_keys.json")
        
        if not os.path.exists(keys_file):
            return []
        
        # Read from file
        with open(keys_file, 'r') as f:
            keys_list = json.load(f)
        
        return keys_list
    except Exception as e:
        st.error(f"Failed to load API keys: {str(e)}")
        return []


def generate_api_key() -> str:
    """Generate a new API key."""
    import secrets
    import string
    
    # Generate a random string for the API key
    alphabet = string.ascii_letters + string.digits
    key = ''.join(secrets.choice(alphabet) for _ in range(24))
    
    # Add prefix for better identification
    return f"scl_{key}"


def setup_enhanced_api_integration():
    """Set up the enhanced API integration interface."""
    st.header("🔌 API Integration")
    
    st.markdown("""
    This section provides tools to interact with the SilentCodingLegend AI through an API interface.
    You can generate API keys, test API calls, and view documentation for integrating with external applications.
    """)
    
    # API key management
    st.subheader("API Key Management")
    
    # Initialize session state for API keys from persistent storage
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = load_api_keys()
    
    # Generate new API key
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Generate New API Key", type="primary"):
            new_key = generate_api_key()
            st.session_state.api_keys.append({
                "key": new_key,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_used": None,
                "requests": 0,
                "active": True
            })
            # Save to persistent storage
            save_api_keys(st.session_state.api_keys)
            st.success(f"New API key generated: {new_key}")
            # This will cause the UI to update
            st.rerun()
    
    with col2:
        # Display number of active keys
        active_keys = sum(1 for key in st.session_state.api_keys if key.get("active", True))
        st.info(f"Active API keys: {active_keys}")
    
    # Display existing API keys
    if st.session_state.api_keys:
        st.subheader("Your API Keys")
        
        # Sort keys with active ones first
        sorted_keys = sorted(
            st.session_state.api_keys, 
            key=lambda x: (not x.get("active", True), x.get("created", "")), 
            reverse=False
        )
        
        for i, key_info in enumerate(sorted_keys):
            # Determine if key is active
            active = key_info.get("active", True)
            status_icon = "🟢" if active else "🔴"
            status_text = "Active" if active else "Revoked"
            
            # Create appropriate UI based on status
            with st.expander(f"{status_icon} API Key {i+1} - {status_text}"):
                st.text(f"Key: {key_info['key']}")
                st.text(f"Created: {key_info['created']}")
                st.text(f"Last Used: {key_info['last_used'] or 'Never'}")
                st.text(f"Requests: {key_info['requests']}")
                
                # Button actions depend on active status
                if active:
                    if st.button("Revoke Key", key=f"revoke_{i}", type="secondary"):
                        # Mark as inactive rather than delete
                        st.session_state.api_keys[i]["active"] = False
                        save_api_keys(st.session_state.api_keys)
                        st.warning("API key revoked")
                        st.rerun()
                else:
                    if st.button("Reactivate Key", key=f"reactivate_{i}"):
                        st.session_state.api_keys[i]["active"] = True
                        save_api_keys(st.session_state.api_keys)
                        st.success("API key reactivated")
                        st.rerun()
                    
                    if st.button("Delete Permanently", key=f"delete_{i}", type="secondary"):
                        st.session_state.api_keys.pop(i)
                        save_api_keys(st.session_state.api_keys)
                        st.error("API key deleted permanently")
                        st.rerun()
    
    # API Documentation
    with st.expander("API Documentation", expanded=False):
        st.markdown("""
        ### SilentCodingLegend AI API
        
        This API allows you to interact with our AI models programmatically.
        
        #### Authentication
        All API requests require an API key to be included in the header:
        
        ```
        X-API-Key: your_api_key_here
        ```
        
        #### Endpoints
        
        ##### 1. Text Completion
        
        **URL:** `/api/v1/completions`
        
        **Method:** `POST`
        
        **Body:**
        ```json
        {
            "model": "llama3-70b-8192",
            "prompt": "Write a function to calculate fibonacci numbers",
            "max_tokens": 500,
            "temperature": 0.7
        }
        ```
        
        **Response:**
        ```json
        {
            "id": "cmpl-123abc",
            "object": "text_completion",
            "created": 1715857230,
            "model": "llama3-70b-8192",
            "choices": [
                {
                    "text": "def fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    \n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[i-1] + fib[i-2])\n    \n    return fib\n\n# Example usage\nprint(fibonacci(10))",
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 181,
                "total_tokens": 189
            }
        }
        ```
        
        ##### 2. Chat Completion
        
        **URL:** `/api/v1/chat/completions`
        
        **Method:** `POST`
        
        **Body:**
        ```json
        {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "How do I create a React component?"}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        ```
        
        **Response:**
        ```json
        {
            "id": "chatcmpl-456def",
            "object": "chat.completion",
            "created": 1715857230,
            "model": "llama3-70b-8192",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "To create a React component, you can use either a function or a class approach. Here's how to create a simple functional component..."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 150,
                "total_tokens": 175
            }
        }
        ```
        
        ##### 3. Vision Analysis
        
        **URL:** `/api/v1/vision/analyze`
        
        **Method:** `POST`
        
        **Body:** Form data with an 'image' file and 'prompt' text field
        
        **Response:** Similar to Chat Completion but includes analysis of the image
        
        #### Rate Limits
        
        - Free tier: 100 requests per day
        - Standard tier: 1,000 requests per day
        - Premium tier: 10,000 requests per day
        
        #### Error Codes
        
        - 401: Unauthorized - Invalid API key
        - 403: Forbidden - Rate limit exceeded
        - 404: Not found - Endpoint does not exist
        - 422: Validation error - Invalid request body
        - 500: Server error
        
        #### Language Support
        
        The API supports the following programming languages:
        
        - Python
        - JavaScript/TypeScript
        - Java
        - C#
        - PHP
        - Go
        - Ruby
        - And many others
        
        #### Code Examples
        
        **Python Example:**
        ```python
        import requests
        import json
        
        api_key = "your_api_key_here"
        
        url = "https://api.silentcodinglegend.ai/v1/chat/completions"
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a Python coding expert."},
                {"role": "user", "content": "Write a function to sort a dictionary by values."}
            ],
            "temperature": 0.7
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
        
        response = requests.post(url, json=payload, headers=headers)
        result = response.json()
        
        print(result["choices"][0]["message"]["content"])
        ```
        
        **JavaScript Example:**
        ```javascript
        async function callAPI() {
            const apiKey = 'your_api_key_here';
            const url = 'https://api.silentcodinglegend.ai/v1/chat/completions';
            
            const payload = {
                model: 'llama3-70b-8192',
                messages: [
                    {role: 'system', content: 'You are a JavaScript coding expert.'},
                    {role: 'user', content: 'Write a function to deep clone an object.'}
                ],
                temperature: 0.7
            };
            
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': apiKey
                },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            console.log(result.choices[0].message.content);
        }
        ```
        """)
    
    # API Test Interface
    st.subheader("API Testing")
    
    # Select endpoint to test
    endpoint = st.selectbox(
        "Select endpoint",
        ["Text Completion", "Chat Completion", "Vision Analysis"]
    )
    
    # Based on endpoint, show different forms
    if endpoint == "Text Completion":
        test_text_completion_api()
    elif endpoint == "Chat Completion":
        test_chat_completion_api()
    else:
        test_vision_api()


def test_text_completion_api():
    """Test the text completion API endpoint."""
    from src.config import MODEL_INFO
    
    # Get all models
    models = list(MODEL_INFO.keys())
    
    # Form for testing
    with st.form("text_completion_form"):
        # Select model
        model = st.selectbox("Model", models)
        
        # Prompt
        prompt = st.text_area(
            "Prompt",
            value="Write a function to calculate the factorial of a number.",
            height=100
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        with col2:
            max_tokens = st.slider("Max Tokens", 50, 2000, 500, 50)
            
        # Submit button
        submitted = st.form_submit_button("Test API Call")
    
    # Handle submission
    if submitted:
        # Construct the API request body
        request_body = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Display the API call that would be made
        st.subheader("API Request")
        st.code(json.dumps(request_body, indent=2), language="json")
        
        # Simulate API response
        from src.utils import query_groq_model
        
        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            with st.spinner("Calling API..."):
                response = query_groq_model(
                    model_id=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
            
            # Construct response JSON
            api_response = {
                "id": f"cmpl-{uuid.uuid4()}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "text": response.choices[0].message.content,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # Display the response
            st.subheader("API Response")
            st.code(json.dumps(api_response, indent=2), language="json")
            
            # Display formatted code response
            from src.enhanced_features import process_code_blocks
            st.subheader("Formatted Response")
            st.markdown(process_code_blocks(response.choices[0].message.content), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")


def test_chat_completion_api():
    """Test the chat completion API endpoint."""
    from src.config import MODEL_INFO
    
    # Get all models
    models = list(MODEL_INFO.keys())
    
    # Initialize messages in session state if needed
    if "api_test_messages" not in st.session_state:
        st.session_state.api_test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do I create a React component?"}
        ]
    
    # Form for testing
    with st.form("chat_completion_form"):
        # Select model
        model = st.selectbox("Model", models)
        
        # Messages
        st.subheader("Messages")
        
        messages = st.session_state.api_test_messages.copy()
        updated_messages = []
        
        for i, msg in enumerate(messages):
            col1, col2 = st.columns([1, 3])
            with col1:
                role = st.selectbox(f"Role {i+1}", ["system", "user", "assistant"], index=["system", "user", "assistant"].index(msg["role"]), key=f"role_{i}")
            with col2:
                content = st.text_area(f"Content {i+1}", value=msg["content"], height=100, key=f"content_{i}")
                
            updated_messages.append({"role": role, "content": content})
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        with col2:
            max_tokens = st.slider("Max Tokens", 50, 2000, 500, 50)
            
        # Add message option
        add_message = st.checkbox("Add a new message")
        
        # Submit button
        submitted = st.form_submit_button("Test API Call")
        
        # Update session state with messages
        st.session_state.api_test_messages = updated_messages
        
        # Add new message if requested
        if add_message:
            st.session_state.api_test_messages.append({"role": "user", "content": ""})
    
    # Handle submission
    if submitted:
        # Construct the API request body
        request_body = {
            "model": model,
            "messages": st.session_state.api_test_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Display the API call that would be made
        st.subheader("API Request")
        st.code(json.dumps(request_body, indent=2), language="json")
        
        # Simulate API response
        from src.utils import query_groq_model
        
        try:
            with st.spinner("Calling API..."):
                response = query_groq_model(
                    model_id=model,
                    messages=st.session_state.api_test_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
            
            # Construct response JSON
            api_response = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.choices[0].message.content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # Display the response
            st.subheader("API Response")
            st.code(json.dumps(api_response, indent=2), language="json")
            
            # Display formatted response
            from src.enhanced_features import process_code_blocks
            st.subheader("Formatted Response")
            st.markdown(process_code_blocks(response.choices[0].message.content), unsafe_allow_html=True)
            
            # Add the assistant's response to the conversation
            st.session_state.api_test_messages.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")


def test_vision_api():
    """Test the vision API endpoint."""
    from src.config import VISION_MODELS, ALLOWED_IMAGE_EXTENSIONS
    
    # Get vision models
    vision_models = list(VISION_MODELS.keys())
    
    # Form for testing
    with st.form("vision_api_form"):
        # Select model
        model = st.selectbox("Vision Model", vision_models)
        
        # Upload image
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=list(ALLOWED_IMAGE_EXTENSIONS)
        )
        
        # Prompt
        prompt = st.text_area(
            "Prompt",
            value="Describe what you see in this image in detail.",
            height=100
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        with col2:
            max_tokens = st.slider("Max Tokens", 50, 2000, 500, 50)
            
        # Submit button
        submitted = st.form_submit_button("Test API Call")
    
    # Handle submission
    if submitted and uploaded_file:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            image_path = tmp_file.name
            
        try:
            # Display image
            st.image(image_path, caption="Uploaded Image", width=300)
            
            # Encode image to base64
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Get file extension
            file_ext = os.path.splitext(uploaded_file.name)[1].lower().replace('.', '')
            if not file_ext or file_ext not in ['jpg', 'jpeg', 'png', 'webp']:
                file_ext = 'jpeg'  # Default to jpeg
            
            # Construct the API request body
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{file_ext};base64,{base64_image}"
                        }
                    }
                ]
            }
            
            request_body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an image analysis assistant."},
                    # User message would include the image data
                    {"role": "user", "content": "The image data would be included here with the prompt: " + prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Display the API call that would be made
            st.subheader("API Request")
            st.code(json.dumps(request_body, indent=2), language="json")
            
            # Simulate API response
            from src.utils import query_groq_vision_model
            
            # Prepare messages for API call
            system_prompt = "You are an expert image analyst."
            
            messages = [
                {"role": "system", "content": system_prompt},
                user_message
            ]
            
            # Call the vision model API
            with st.spinner("Analyzing image..."):
                response = query_groq_vision_model(
                    model_id=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
            
            # Construct response JSON
            api_response = {
                "id": f"visn-{uuid.uuid4()}",
                "object": "vision.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.choices[0].message.content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # Display the response
            st.subheader("API Response")
            st.code(json.dumps(api_response, indent=2), language="json")
            
            # Display the formatted response
            from src.enhanced_features import process_code_blocks
            st.subheader("Analysis Result")
            st.markdown(process_code_blocks(response.choices[0].message.content), unsafe_allow_html=True)
            
            # Clean up temporary file
            try:
                os.remove(image_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            
            # Clean up temporary file
            try:
                os.remove(image_path)
            except:
                pass
    elif submitted:
        st.warning("Please upload an image to test the vision API.")


def initialize_groq_client() -> groq.Client:
    """
    Initialize the Groq API client with proper error handling.
    
    Returns:
        groq.Client: The initialized Groq client
        
    Raises:
        AuthenticationError: If the API key is invalid
        RuntimeError: For other initialization errors
    """
    # Get API key from environment or config
    from src.config import GROQ_API_KEY
    
    if not GROQ_API_KEY:
        error_msg = "Groq API key not found. Please add it to your .env file."
        logger.error(error_msg)
        st.error(error_msg)
        st.stop()
    
    try:
        return groq.Client(api_key=GROQ_API_KEY)
    except Exception as e:
        error_msg = f"Error initializing Groq client: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        raise RuntimeError(error_msg)


def handle_api_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle API errors consistently across the application.
    
    Args:
        func: The function to wrap with error handling
        
    Returns:
        The wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Handle API-specific errors
            if hasattr(e, 'message') and hasattr(e, 'http_status'):
                error_msg = f"API error: {getattr(e, 'message')} (Status: {getattr(e, 'http_status')})"
                logger.error(error_msg)
                st.error(error_msg)
            else:
                error_msg = f"Unexpected error during API call: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
            raise
    return wrapper


def prepare_api_request(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = True,
    json_mode: bool = False,
    top_p: float = 1.0,
) -> Dict[str, Any]:
    """
    Prepare a standardized API request dictionary for the Groq API.
    
    Args:
        model: The model ID to use
        messages: The conversation messages
        temperature: Controls randomness (0-1)
        max_tokens: Maximum number of tokens to generate
        stream: Whether to return a streaming response
        json_mode: Whether to return response in JSON format
        top_p: Nucleus sampling parameter
        
    Returns:
        Dict with properly formatted API request parameters
    """
    # Validate input parameters
    if not model or not isinstance(model, str):
        raise ValueError("Invalid model ID")
    
    if not messages or not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Messages must be a non-empty list")
    
    if temperature < 0.0 or temperature > 1.0:
        logger.warning(f"Temperature value {temperature} outside recommended range (0-1)")
    
    # Prepare the base request parameters
    api_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "top_p": top_p
    }
    
    # Add JSON mode if selected
    if json_mode:
        api_params["response_format"] = {"type": "json_object"}
        api_params["stream"] = False
    else:
        api_params["stream"] = stream
    
    return api_params


def process_api_response(response: Any, is_streaming: bool = False) -> Dict[str, Any]:
    """
    Process the API response into a standardized format.
    
    Args:
        response: The raw API response
        is_streaming: Whether this is a streaming response chunk
        
    Returns:
        Dict with standardized response data
    """
    try:
        result = {}
        
        if is_streaming:
            # Handle streaming response chunk
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'delta'):
                    result["content"] = response.choices[0].delta.content or ""
                    result["finish_reason"] = response.choices[0].finish_reason
                    result["is_chunk"] = True
        else:
            # Handle regular response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                result["content"] = response.choices[0].message.content
                result["finish_reason"] = response.choices[0].finish_reason
                result["role"] = response.choices[0].message.role
            
            # Add usage statistics if available
            if hasattr(response, 'usage'):
                result["tokens"] = response.usage.total_tokens
                result["prompt_tokens"] = response.usage.prompt_tokens
                result["completion_tokens"] = response.usage.completion_tokens
                
        return result
        
    except Exception as e:
        logger.error(f"Error processing API response: {str(e)}")
        return {
            "error": str(e),
            "content": "Error processing response"
        }
