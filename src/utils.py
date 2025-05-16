"""
Utility functions for SilentCodingLegend AI application.
"""

import os
import tempfile
import json
import uuid
import datetime
from typing import List, Dict, Any, Optional
import streamlit as st
import groq
import logging
from src.config import GROQ_API_KEY, ALLOWED_EXTENSIONS, DEFAULT_SYSTEM_PROMPT, JSON_SYSTEM_PROMPT_ADDITION, CHAT_HISTORY_PATH
from src.model_metrics import metrics_tracker

def get_groq_client():
    """Get initialized Groq client with API key."""
    if not GROQ_API_KEY:
        st.error("Groq API key not found. Please add it to your .env file.")
        st.stop()
    
    try:
        return groq.Client(api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        raise RuntimeError(f"Failed to create Groq client: {str(e)}")

def is_valid_file_extension(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    if not filename:
        return False
    extension = filename.split('.')[-1].lower() if '.' in filename else ''
    return extension in ALLOWED_EXTENSIONS

def create_directories() -> None:
    """Create all necessary directories for the application."""
    from src.config import ASSETS_PATH, UPLOAD_PATH, KNOWLEDGE_PATH, VECTOR_DB_PATH, CHAT_HISTORY_PATH, IMAGE_UPLOAD_PATH
    
    for directory in [ASSETS_PATH, UPLOAD_PATH, KNOWLEDGE_PATH, VECTOR_DB_PATH, CHAT_HISTORY_PATH, IMAGE_UPLOAD_PATH]:
        os.makedirs(directory, exist_ok=True)

def save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded file to disk and return the path."""
    from src.config import UPLOAD_PATH
    
    if uploaded_file is None:
        raise ValueError("No file was provided for upload")
    
    try:
        # Create upload directory if it doesn't exist
        os.makedirs(UPLOAD_PATH, exist_ok=True)
        
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_PATH, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except OSError as e:
        # Handle file system errors
        error_msg = f"File system error while saving uploaded file: {str(e)}"
        st.error(error_msg)
        raise OSError(error_msg)
    except Exception as e:
        # Handle other unexpected errors
        error_msg = f"Unexpected error while saving uploaded file: {str(e)}"
        st.error(error_msg)
        raise RuntimeError(error_msg)

def backup_chat_history(messages: List[Dict], document_name: str, model_id: str) -> str:
    """
    Backup chat history to a JSON file with date, time, session, and model information.
    Returns the path to the saved backup file.
    """
    # Create a unique session ID if not exists
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())[:8]
    
    session_id = st.session_state.chat_session_id
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename with document name, timestamp, session ID, and model
    safe_doc_name = "".join(c if c.isalnum() else "_" for c in document_name)
    safe_model_id = model_id.replace("/", "-").replace(".", "-")
    
    filename = f"{timestamp}_{safe_doc_name}_{session_id}_{safe_model_id}.json"
    backup_path = os.path.join(CHAT_HISTORY_PATH, filename)
    
    # Prepare data with metadata
    backup_data = {
        "metadata": {
            "document": document_name,
            "timestamp": timestamp,
            "session_id": session_id,
            "model": model_id,
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.datetime.now().strftime("%H:%M:%S")
        },
        "messages": messages
    }
    
    # Save to file
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(backup_data, f, indent=2, ensure_ascii=False)
    
    return backup_path

def get_chat_history_backups() -> List[Dict]:
    """
    Get list of all chat history backups with metadata.
    Returns a list of dictionaries with backup information.
    """
    backups = []
    
    # Ensure directory exists
    os.makedirs(CHAT_HISTORY_PATH, exist_ok=True)
    
    # List all JSON files in the chat history directory
    for filename in os.listdir(CHAT_HISTORY_PATH):
        if filename.endswith('.json'):
            file_path = os.path.join(CHAT_HISTORY_PATH, filename)
            try:
                # Read the file and extract metadata
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                metadata = data.get('metadata', {})
                # Add file path and filename to metadata
                metadata['file_path'] = file_path
                metadata['filename'] = filename
                
                backups.append(metadata)
            except Exception as e:
                print(f"Error reading backup file {filename}: {str(e)}")
    
    # Sort backups by timestamp (newest first)
    backups.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return backups

def load_chat_history_backup(file_path: str) -> List[Dict]:
    """
    Load chat messages from a backup file.
    Returns the list of chat messages.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('messages', [])
    except Exception as e:
        print(f"Error loading chat history backup: {str(e)}")
        return []

def reset_kb_state():
    """Reset Knowledge Base session state for troubleshooting."""
    if "kb_messages" in st.session_state:
        st.session_state.kb_messages = []
    if "chat_session_id" in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())[:8]

def apply_custom_style() -> None:
    """Apply custom styling to the Streamlit UI based on selected theme."""
    from src.config import (
        DARK_THEME, LIGHT_THEME, PRIMARY_COLOR
    )
    
    # Initialize theme in session state if not already set
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "dark"
    
    # Get the current theme colors
    current_theme = DARK_THEME if st.session_state.ui_theme == "dark" else LIGHT_THEME
    
    # Extract colors from the current theme
    background_color = current_theme["background_color"]
    secondary_background_color = current_theme["secondary_background_color"]
    text_color = current_theme["text_color"]
    user_message_color = current_theme["user_message_color"]
    assistant_message_color = current_theme["assistant_message_color"]
    border_color = current_theme["border_color"]
    input_background = current_theme["input_background"]
    sidebar_color = current_theme["sidebar_color"]
    code_background = current_theme["code_background"]
    code_text = current_theme["code_text"]
    
    # Add viewport meta tag for responsive design
    st.markdown("""
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <style>
        /* Theme base colors */
        :root {{
            --background-color: {background_color};
            --secondary-background-color: {secondary_background_color};
            --text-color: {text_color};
            --user-message-color: {user_message_color};
            --assistant-message-color: {assistant_message_color};
            --accent-color: {PRIMARY_COLOR};
            --border-color: {border_color};
            --input-background: {input_background};
            --sidebar-color: {sidebar_color};
        }}
        
        /* Main app background */
        .stApp {{
            background-color: var(--background-color);
            color: var(--text-color);
        }}
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: var(--sidebar-color);
            border-right: 1px solid var(--border-color);
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--text-color) !important;
        }}
        
        /* Text input styling */
        .stTextInput > div > div > input {{
            background-color: var(--input-background);
            color: var(--text-color);
            border: 1px solid var(--border-color);
        }}
        
        /* Chat message styling */
        .chat-message {{
            padding: 1.5rem; 
            border-radius: 0.8rem; 
            margin-bottom: 1rem; 
            display: flex;
            align-items: flex-start;
            border: 1px solid var(--border-color);
        }}
        
        /* User message styling */
        .chat-message.user {{
            background-color: var(--user-message-color);
            color: white;
        }}
        
        /* Assistant message styling */
        .chat-message.assistant {{
            background-color: var(--assistant-message-color);
            color: var(--text-color);
        }}
        
        /* Avatar styling */
        .chat-message .avatar {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            object-fit: cover;
            margin-right: 1rem;
            border: 1px solid var(--border-color);
        }}
        
        /* Message content */
        .chat-message .message {{
            flex: 1;
        }}
        
        /* Buttons */
        .stButton button {{
            background-color: var(--accent-color);
            color: white;
            border: none;
        }}
        
        /* Slider */
        .stSlider div[data-baseweb="slider"] {{
            color: var(--accent-color);
        }}
        
        /* Selectbox */
        div[data-baseweb="select"] {{
            background-color: var(--input-background);
            border-color: var(--border-color);
        }}
        
        div[data-baseweb="select"] > div {{
            background-color: var(--input-background);
            color: var(--text-color);
        }}
        
        /* Code blocks */
        code {{
            background-color: #2b2b2b;
            color: #e6e6e6;
        }}
        
        /* Pre blocks */
        pre {{
            background-color: #2b2b2b;
            border: 1px solid var(--border-color);
            border-radius: 5px;
        }}
        
        /* Custom styling for the chat input */
        .stChatInput > div {{
            background-color: {input_background} !important;
            border: 1px solid {border_color} !important;
        }}
        .stChatInput input {{
            color: {text_color} !important;
        }}
        .stChatInput button {{
            background-color: {PRIMARY_COLOR} !important;
        }}
        
        /* Responsive styles for mobile devices */
        @media (max-width: 768px) {{
            /* Main container adjustments */
            [data-testid="stAppViewContainer"] > div {{
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }}
            
            /* Header size adjustments */
            h1 {{
                font-size: 1.5rem !important;
            }}
            h2 {{
                font-size: 1.3rem !important;
            }}
            h3 {{
                font-size: 1.1rem !important;
            }}
            
            /* Sidebar optimizations */
            section[data-testid="stSidebar"] {{
                width: 85vw !important;
                min-width: 85vw !important;
            }}
            
            /* Chat message adjustments */
            .chat-message {{
                padding: 0.8rem !important;
                margin-bottom: 0.6rem !important;
            }}
            
            /* Column adjustments */
            [data-testid="column"] {{
                padding: 0.2rem !important;
            }}
            
            /* Better button touch targets */
            .stButton button {{
                min-height: 44px !important;
                margin: 0.3rem 0 !important;
            }}
            
            /* Form input adjustments */
            input, textarea, select {{
                font-size: 16px !important; /* Prevents iOS zoom on focus */
            }}
            
            /* Improve tab usability */
            button[role="tab"] {{
                padding: 0.6rem 0.8rem !important;
            }}
            
            /* Code block adjustments */
            pre {{
                max-width: 100% !important;
                overflow-x: auto !important;
                font-size: 0.8rem !important;
                padding: 0.5rem !important;
            }}
            
            /* File uploader adjustments */
            [data-testid="stFileUploader"] {{
                width: 100% !important;
            }}
            
            /* Reduce markdown spacing */
            [data-testid="stMarkdown"] {{
                padding-top: 0.2rem !important;
                padding-bottom: 0.2rem !important;
            }}
        }}
        
        /* Tablet optimizations */
        @media (min-width: 769px) and (max-width: 1024px) {{
            /* Slightly adjusted styles for tablets */
            [data-testid="stAppViewContainer"] > div {{
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}
            
            /* Sidebar width for tablet */
            section[data-testid="stSidebar"] {{
                width: 320px !important;
                min-width: 320px !important;
            }}
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # Apply additional mobile optimizations if mobile_optimization is imported
    try:
        from src.mobile_optimization import optimize_for_mobile
        optimize_for_mobile()
    except ImportError:
        pass

def query_groq_model(
    model_id: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = True,
    json_mode: bool = False,
    top_p: float = 1.0,
    template_id: Optional[str] = None,
) -> Any:
    """
    Send a query to the Groq API and return the response.
    
    Args:
        model_id: The ID of the Groq model to use
        messages: List of message dictionaries with role and content
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in response
        stream: Whether to stream the response
        json_mode: Whether to request JSON formatted response
        top_p: Nucleus sampling parameter
        template_id: ID of the chat template being used, if any
        
    Returns:
        The Groq API response object
        
    Raises:
        ValueError: For invalid parameter values
        ConnectionError: For network issues
        AuthenticationError: For API key issues
        InvalidRequestError: For invalid request parameters
        APIError: For other API errors
    """
    import logging
    
    if not model_id:
        raise ValueError("Model ID cannot be empty")
    
    if not messages or not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Messages must be a non-empty list")
    
    if temperature < 0.0 or temperature > 1.0:
        raise ValueError("Temperature must be between 0.0 and 1.0")
        
    # Start tracking metrics
    start_time = metrics_tracker.start_tracking()
    success = False
    error_type = None
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    
    try:
        client = get_groq_client()
        
        # Log the API request (without sensitive content)
        logger = logging.getLogger(__name__)
        logger.info(f"Querying Groq model: {model_id}, temperature: {temperature}, max_tokens: {max_tokens}")
        
        # Prepare API parameters
        api_params = {
            "model": model_id,
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
        
        # Call Groq API
        response = client.chat.completions.create(**api_params)
        
        # Set metrics for successful response
        success = True
        if hasattr(response, 'usage') and response.usage is not None:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
        
        # Record metrics (for non-streaming responses)
        if not stream:
            metrics_tracker.record_metrics(
                model_id=model_id,
                start_time=start_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                is_cached=False,
                query_type="text",
                success=success,
                template_id=template_id
            )
        
        return response
        
    except groq.errors.AuthenticationError as e:
        error_msg = f"Authentication error: {str(e)}. Please check your API key."
        logging.error(error_msg)
        error_type = "AuthenticationError"
        raise
        
    except groq.errors.InvalidRequestError as e:
        error_msg = f"Invalid request: {str(e)}. Please check model ID and parameters."
        logging.error(error_msg)
        error_type = "InvalidRequestError"
        raise
        
    except groq.errors.APIError as e:
        error_msg = f"Groq API error: {str(e)}"
        logging.error(error_msg)
        error_type = "APIError"
        raise
        
    except (ConnectionError, TimeoutError) as e:
        error_msg = f"Network error while connecting to Groq API: {str(e)}"
        logging.error(error_msg)
        error_type = "NetworkError"
        raise ConnectionError(error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error during Groq API call: {str(e)}"
        logging.error(error_msg)
        error_type = "UnexpectedError"
        raise RuntimeError(error_msg)
        
    finally:
        # Record metrics for failed requests
        if not success:
            metrics_tracker.record_metrics(
                model_id=model_id,
                start_time=start_time,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                is_cached=False,
                query_type="text",
                success=False,
                error_type=error_type,
                template_id=template_id
            )

def get_system_prompt(json_mode: bool = False) -> str:
    """
    Get the system prompt with optional JSON mode instructions.
    
    If a template is active and has defined a system prompt,
    use that instead of the default.
    """
    # Check if there's a template-specific system prompt in session state
    if "system_prompt" in st.session_state:
        prompt = st.session_state.system_prompt
    else:
        prompt = DEFAULT_SYSTEM_PROMPT
        
    # Add JSON mode instructions if needed
    if json_mode:
        prompt += JSON_SYSTEM_PROMPT_ADDITION
        
    return prompt
    
def is_valid_image_extension(filename: str) -> bool:
    """Check if the file has an allowed image extension."""
    if not filename:
        return False
    extension = filename.split('.')[-1].lower() if '.' in filename else ''
    from src.config import ALLOWED_IMAGE_EXTENSIONS
    return extension in ALLOWED_IMAGE_EXTENSIONS

def save_uploaded_image(uploaded_file) -> str:
    """Save an uploaded image file to disk and return the path."""
    from src.config import IMAGE_UPLOAD_PATH, ALLOWED_IMAGE_EXTENSIONS, MAX_IMAGE_SIZE_MB
    import os
    
    if uploaded_file is None:
        raise ValueError("No image file was provided for upload")
    
    # Validate file extension
    if not is_valid_image_extension(uploaded_file.name):
        error_msg = f"Invalid image format. Allowed formats: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        st.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate file size
    file_size_mb = uploaded_file.size / (1024 * 1024)  # Convert to MB
    if file_size_mb > MAX_IMAGE_SIZE_MB:
        error_msg = f"Image size exceeds maximum allowed size of {MAX_IMAGE_SIZE_MB}MB"
        st.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(IMAGE_UPLOAD_PATH, exist_ok=True)
        
        # Use a timestamp in the filename to avoid duplicates
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add a UUID for additional uniqueness
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}_{uploaded_file.name}"
        file_path = os.path.join(IMAGE_UPLOAD_PATH, filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return file_path
    except OSError as e:
        # Handle file system errors
        error_msg = f"File system error while saving image: {str(e)}"
        st.error(error_msg)
        raise OSError(error_msg)
    except Exception as e:
        # Handle other unexpected errors
        error_msg = f"Unexpected error while saving image: {str(e)}"
        st.error(error_msg)
        raise RuntimeError(error_msg)

def query_groq_vision_model(
    model_id: str,
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = True,
    template_id: Optional[str] = None,
) -> Any:
    """
    Send a query to the Groq API for vision models and return the response.
    Messages should include proper format for images with content and image_url.
    
    Args:
        model_id: The ID of the Groq vision model to use
        messages: List of message dictionaries with role and content (including images)
        temperature: Controls randomness (0-1)
        max_tokens: Maximum tokens in response
        stream: Whether to stream the response
        template_id: ID of the chat template being used, if any
        
    Returns:
        The Groq API response object
        
    Raises:
        ValueError: For invalid parameter values
        ConnectionError: For network issues
        groq.errors.AuthenticationError: For API key issues
        groq.errors.InvalidRequestError: For invalid request parameters
        groq.errors.APIError: For other API errors
    """
    import logging
    
    if not model_id:
        raise ValueError("Model ID cannot be empty")
    
    if not messages or not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Messages must be a non-empty list")
    
    # Verify that at least one message contains an image
    has_image = False
    for message in messages:
        content = message.get('content', [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'image_url':
                    has_image = True
                    break
        if has_image:
            break
    
    if not has_image:
        logging.warning("No images found in messages. Ensure proper format for vision models.")
        
    # Start tracking metrics
    start_time = metrics_tracker.start_tracking()
    success = False
    error_type = None
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    
    try:
        client = get_groq_client()
        
        # Log the API request
        logging.info(f"Querying Groq vision model: {model_id}, temperature: {temperature}, max_tokens: {max_tokens}")
        
        # Prepare API parameters
        api_params = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": stream
        }
        
        # Call Groq API
        response = client.chat.completions.create(**api_params)
        
        # Set metrics for successful response
        success = True
        if hasattr(response, 'usage') and response.usage is not None:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
        
        # Record metrics (for non-streaming responses)
        if not stream:
            metrics_tracker.record_metrics(
                model_id=model_id,
                start_time=start_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                is_cached=False,
                query_type="vision",
                success=success,
                template_id=template_id
            )
        
        return response
        
    except groq.errors.AuthenticationError as e:
        error_msg = f"Authentication error: {str(e)}. Please check your API key."
        logging.error(error_msg)
        st.error(error_msg)
        error_type = "AuthenticationError"
        raise
        
    except groq.errors.InvalidRequestError as e:
        error_msg = f"Invalid request: {str(e)}. Please check model ID and message format."
        logging.error(error_msg)
        st.error(error_msg)
        error_type = "InvalidRequestError"
        raise
        
    except groq.errors.APIError as e:
        error_msg = f"Groq API error: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)
        error_type = "APIError"
        raise
        
    except (ConnectionError, TimeoutError) as e:
        error_msg = f"Network error while connecting to Groq API: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)
        error_type = "NetworkError"
        raise ConnectionError(error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error during Groq vision API call: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)
        error_type = "UnexpectedError"
        raise RuntimeError(error_msg)
        
    finally:
        # Record metrics for failed requests
        if not success:
            metrics_tracker.record_metrics(
                model_id=model_id,
                start_time=start_time,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                is_cached=False,
                query_type="vision",
                success=False,
                error_type=error_type,
                template_id=template_id
            )
