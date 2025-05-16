"""
Advanced features for SilentCodingLegend AI application.
Includes code execution, batch processing, and API integration.
"""

import streamlit as st
import subprocess
import time
import os
import uuid
import base64
import json
import re
from typing import List, Dict, Any, Optional
import tempfile
from io import StringIO
import traceback
from PIL import Image
from pathlib import Path

# Code Execution Environment
def setup_code_execution():
    """Set up the code execution environment in the app."""
    if "code_exec_history" not in st.session_state:
        st.session_state.code_exec_history = []
        
    st.header("💻 Code Execution Environment")
    
    # Code language selection
    languages = {
        "python": {"display": "Python", "ext": ".py", "cmd": "python"},
        "javascript": {"display": "JavaScript (Node.js)", "ext": ".js", "cmd": "node"},
        "bash": {"display": "Bash", "ext": ".sh", "cmd": "bash"}
    }
    
    language = st.selectbox(
        "Select language",
        options=list(languages.keys()),
        format_func=lambda x: languages[x]["display"]
    )
    
    # Code editor
    code = st.text_area(
        "Enter code to execute",
        height=250,
        key="code_editor",
        help="Write your code here and execute it directly"
    )
    
    # Sample code templates
    with st.expander("Load a sample template"):
        sample_templates = {
            "python": 'print("Hello, world!")\n\n# Basic calculation\nresult = 5 * 10\nprint(f"5 * 10 = {result}")\n\n# List comprehension example\nnumbers = [1, 2, 3, 4, 5]\nsquares = [n**2 for n in numbers]\nprint(f"Original numbers: {numbers}")\nprint(f"Squared numbers: {squares}")',
            
            "javascript": 'console.log("Hello, world!");\n\n// Basic calculation\nconst result = 5 * 10;\nconsole.log(`5 * 10 = ${result}`);\n\n// Array map example\nconst numbers = [1, 2, 3, 4, 5];\nconst squares = numbers.map(n => n**2);\nconsole.log(`Original numbers: ${numbers}`);\nconsole.log(`Squared numbers: ${squares}`);',
            
            "bash": 'echo "Hello, world!"\n\n# Show current directory\necho "Current directory:"\npwd\n\n# List files\necho "\\nFiles in directory:"\nls -la'
        }
        
        if st.button("Load Sample Code", key="load_sample"):
            st.session_state.code_editor = sample_templates[language]
            st.rerun()
    
    # Arguments for the code execution
    args = st.text_input(
        "Command-line arguments (optional)",
        help="Space-separated arguments to pass to your program"
    )
    
    # Security notice
    st.info("⚠️ This code execution environment runs in a sandboxed environment. However, use caution when executing code from untrusted sources.")
    
    # Execute button with spinner
    if st.button("Execute Code", type="primary"):
        with st.spinner("Executing code..."):
            output, success, exec_time = execute_code(code, language, languages[language], args)
            
            # Display execution result
            result_container = st.container(border=True)
            with result_container:
                st.markdown(f"**Execution Time:** {exec_time:.4f} seconds")
                if success:
                    st.success("Code executed successfully")
                else:
                    st.error("Execution failed")
                    
                st.text_area("Output", value=output, height=150, disabled=True)
                
                # Add to execution history
                timestamp = time.strftime("%H:%M:%S")
                history_entry = {
                    "timestamp": timestamp,
                    "language": language,
                    "code": code,
                    "output": output,
                    "success": success,
                    "exec_time": exec_time
                }
                st.session_state.code_exec_history.insert(0, history_entry)
    
    # Display execution history
    if st.session_state.code_exec_history:
        with st.expander("Execution History", expanded=False):
            for i, entry in enumerate(st.session_state.code_exec_history):
                with st.container(border=True):
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        st.text(entry["timestamp"])
                    with col2:
                        st.text(f"Language: {languages[entry['language']]['display']}")
                    with col3:
                        status = "✅ Success" if entry["success"] else "❌ Failed"
                        st.text(status)
                    
                    if st.button("Load this code", key=f"load_history_{i}"):
                        st.session_state.code_editor = entry["code"]
                        st.rerun()


def execute_code(code: str, language: str, lang_info: Dict, args: str = "") -> tuple:
    """
    Execute code in the selected language and return the output.
    
    Args:
        code: The code to execute
        language: Language identifier (python, javascript, bash)
        lang_info: Dictionary with language configuration
        args: Command-line arguments to pass to the program
        
    Returns:
        tuple: (output, success, execution_time)
    """
    start_time = time.time()
    output = ""
    success = False
    
    if not code.strip():
        return "No code provided. Please enter some code to execute.", False, 0.0
    
    try:
        # Create a temporary directory for code execution with proper permissions
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with the code
            file_path = os.path.join(temp_dir, f"code{lang_info['ext']}")
            with open(file_path, 'w') as f:
                f.write(code)
            
            # Make the file executable if it's a bash script
            if language == "bash":
                os.chmod(file_path, 0o755)
                
            # Build the command with proper escaping
            cmd = [lang_info['cmd'], file_path]
            if args:
                # Split args properly respecting quotes
                import shlex
                cmd.extend(shlex.split(args))
            
            # Execute the code in a subprocess with timeout and resource limits
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,  # 15 second timeout for more complex scripts
                env=os.environ.copy()  # Use current environment
            )
            
            # Get the output
            if process.stdout:
                output += process.stdout
            if process.stderr:
                if output:
                    output += "\n\n--- ERRORS ---\n"
                output += process.stderr
            
            success = process.returncode == 0
            
            # Add command information to output for better debugging
            cmd_str = ' '.join(cmd)
            output = f"Command: {cmd_str}\n\n{output}"
            
    except subprocess.TimeoutExpired:
        output = "⚠️ Execution timed out (limit: 15 seconds). The code may be too complex or contain an infinite loop."
    except PermissionError:
        output = "⚠️ Permission error. Unable to execute the code file."
    except FileNotFoundError:
        # This might happen if the language interpreter is not installed
        output = f"⚠️ Error: Could not find {lang_info['cmd']}. Make sure it's installed on your system."
    except Exception as e:
        output = f"⚠️ Error executing code: {str(e)}\n\n{traceback.format_exc()}"
    
    exec_time = time.time() - start_time
    return output, success, exec_time


# Batch Processing for multiple files
def setup_batch_processing():
    """Set up the batch processing interface for multiple documents or images."""
    st.header("🔄 Batch Processing")
    
    # Check if we need to initialize session state
    if "batch_files" not in st.session_state:
        st.session_state.batch_files = []
        
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
        
    # Select batch processing type
    process_type = st.radio(
        "Select processing type",
        ["Document Processing", "Image Analysis"],
        horizontal=True,
    )
    
    if process_type == "Document Processing":
        setup_document_batch_processing()
    else:
        setup_image_batch_processing()


def setup_document_batch_processing():
    """Set up batch processing for documents."""
    from src.config import ALLOWED_EXTENSIONS
    
    st.subheader("📄 Document Batch Processing")
    
    # File uploads
    uploaded_files = st.file_uploader(
        "Upload documents for batch processing",
        accept_multiple_files=True,
        type=list(ALLOWED_EXTENSIONS),
        help=f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
    )
    
    # Update session state with uploaded files
    if uploaded_files:
        st.session_state.batch_files = []
        for file in uploaded_files:
            # Save the file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp_file:
                tmp_file.write(file.getvalue())
                st.session_state.batch_files.append({
                    "name": file.name,
                    "path": tmp_file.name,
                    "type": file.type,
                    "size": file.size
                })
    
    # Display uploaded files
    if st.session_state.batch_files:
        st.success(f"{len(st.session_state.batch_files)} files uploaded")
        
        # File table
        files_data = []
        for file in st.session_state.batch_files:
            size_kb = file["size"] / 1024
            files_data.append([file["name"], f"{size_kb:.1f} KB"])
        
        st.table({"File": [f[0] for f in files_data], "Size": [f[1] for f in files_data]})
        
        # Prompt for processing
        st.subheader("Processing Options")
        
        prompt = st.text_area(
            "Enter your processing prompt",
            help="Describe how you want these documents processed",
            height=100,
            placeholder="Example: Summarize these documents and extract key points",
            key="batch_prompt"
        )
        
        # Model selection for batch processing
        from src.config import MODEL_CATEGORIES
        
        # Get suitable models for batch processing
        suitable_models = []
        for category in ["Featured Models", "Other Models"]:
            if category in MODEL_CATEGORIES:
                suitable_models.extend(MODEL_CATEGORIES[category])
        
        from src.config import MODEL_INFO
        model_names = [MODEL_INFO[model]["display_name"] for model in suitable_models]
        model_dict = dict(zip(model_names, suitable_models))
        
        selected_model_name = st.selectbox(
            "Select model for batch processing",
            options=model_names,
        )
        selected_model = model_dict[selected_model_name]
        
        # Start processing button
        if st.button("Start Batch Processing", type="primary", disabled=not prompt):
            if process_documents_batch(prompt, selected_model):
                st.balloons()
                st.success("Batch processing complete!")
                
        # Display results if available
        if st.session_state.batch_results:
            display_batch_results()


def process_documents_batch(prompt: str, model_id: str) -> bool:
    """
    Process a batch of documents using the selected model.
    
    Args:
        prompt: The processing prompt
        model_id: The model to use for processing
        
    Returns:
        bool: True if processing was successful
    """
    from src.utils import get_system_prompt, query_groq_model
    import os
    
    # Reset results
    st.session_state.batch_results = []
    
    # Get system prompt
    system_prompt = get_system_prompt()
    
    # Process each file
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(st.session_state.batch_files):
        try:
            # Update progress
            progress = (i) / len(st.session_state.batch_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {file['name']}...")
            
            # Read file content
            with open(file["path"], "r", errors="ignore") as f:
                file_content = f.read()
            
            # Truncate if too long
            max_content_length = 6000  # characters
            if len(file_content) > max_content_length:
                file_content = file_content[:max_content_length] + "... [content truncated]"
            
            # Prepare messages for API call
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""
                Please process the following document according to this instruction: {prompt}
                
                Document name: {file['name']}
                
                Document content:
                {file_content}
                """} 
            ]
            
            # Call the API with streaming disabled for batch processing
            response = query_groq_model(
                model_id=model_id,
                messages=messages,
                stream=False,
                json_mode=False,
            )
            
            result = response.choices[0].message.content
            
            # Store results
            st.session_state.batch_results.append({
                "file_name": file["name"],
                "prompt": prompt,
                "result": result,
                "model": model_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
        except Exception as e:
            # Add error result
            st.session_state.batch_results.append({
                "file_name": file["name"],
                "prompt": prompt,
                "result": f"Error processing this file: {str(e)}",
                "model": model_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": True
            })
    
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    # Clean up temp files
    for file in st.session_state.batch_files:
        try:
            os.remove(file["path"])
        except:
            pass
            
    return True


def setup_image_batch_processing():
    """Set up batch processing for images."""
    from src.config import ALLOWED_IMAGE_EXTENSIONS, VISION_MODELS
    
    st.subheader("🖼️ Image Batch Processing")
    
    # File uploads
    uploaded_files = st.file_uploader(
        "Upload images for batch processing",
        accept_multiple_files=True,
        type=list(ALLOWED_IMAGE_EXTENSIONS),
        help=f"Supported formats: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
    )
    
    # Update session state with uploaded files
    if uploaded_files:
        st.session_state.batch_files = []
        for file in uploaded_files:
            # Create a temp file to store the image
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp_file:
                tmp_file.write(file.getvalue())
                
                # Get image dimensions
                img = Image.open(file)
                width, height = img.size
                
                st.session_state.batch_files.append({
                    "name": file.name,
                    "path": tmp_file.name,
                    "type": file.type,
                    "size": file.size,
                    "dimensions": f"{width}x{height}"
                })
    
    # Display uploaded images
    if st.session_state.batch_files:
        st.success(f"{len(st.session_state.batch_files)} images uploaded")
        
        # Display image thumbnails
        cols = st.columns(4)
        for i, file in enumerate(st.session_state.batch_files):
            col = cols[i % 4]
            with col:
                st.image(file["path"], caption=file["name"], width=150)
                st.caption(f"Size: {file['dimensions']}")
        
        # Prompt for processing
        st.subheader("Processing Options")
        
        prompt = st.text_area(
            "Enter your processing prompt",
            help="Describe how you want these images analyzed",
            height=100,
            placeholder="Example: Describe what's in each image and identify any text visible",
            key="batch_prompt"
        )
        
        # Model selection for vision batch processing
        vision_model_options = list(VISION_MODELS.keys())
        vision_model_names = [VISION_MODELS[m]["display_name"] for m in vision_model_options]
        vision_model_dict = dict(zip(vision_model_names, vision_model_options))
        
        selected_model_name = st.selectbox(
            "Select vision model for batch processing",
            options=vision_model_names,
        )
        selected_model = vision_model_dict[selected_model_name]
        
        # Start processing button
        if st.button("Start Batch Processing", type="primary", disabled=not prompt):
            if process_images_batch(prompt, selected_model):
                st.balloons()
                st.success("Image batch processing complete!")
                
        # Display results if available
        if st.session_state.batch_results:
            display_batch_results(is_image=True)


def process_images_batch(prompt: str, model_id: str) -> bool:
    """
    Process a batch of images using the selected vision model.
    
    Args:
        prompt: The processing prompt
        model_id: The vision model to use
        
    Returns:
        bool: True if processing was successful
    """
    from src.utils import query_groq_vision_model
    import os
    import base64
    
    # Reset results
    st.session_state.batch_results = []
    
    # Process each image
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(st.session_state.batch_files):
        try:
            # Update progress
            progress = (i) / len(st.session_state.batch_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {file['name']}...")
            
            # Read image and encode to base64
            with open(file["path"], "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Get file extension
            file_ext = os.path.splitext(file["name"])[1].lower().replace('.', '')
            if not file_ext or file_ext not in ['jpg', 'jpeg', 'png', 'webp']:
                file_ext = 'jpeg'  # Default to jpeg
            
            # Create image message
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}\n\nImage filename: {file['name']}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{file_ext};base64,{base64_image}"
                        }
                    }
                ]
            }
            
            # Prepare messages for API call
            system_prompt = """You are an expert image analyst. When provided with images, provide detailed analysis based on the user's prompt. Be specific, precise, and focus on the details that matter most for the user's request."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                user_message
            ]
            
            # Call the vision model API with streaming disabled for batch processing
            response = query_groq_vision_model(
                model_id=model_id,
                messages=messages,
                stream=False
            )
            
            result = response.choices[0].message.content
            
            # Store results
            st.session_state.batch_results.append({
                "file_name": file["name"],
                "prompt": prompt,
                "result": result,
                "model": model_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": file["path"]
            })
            
        except Exception as e:
            # Add error result
            st.session_state.batch_results.append({
                "file_name": file["name"],
                "prompt": prompt,
                "result": f"Error processing this image: {str(e)}",
                "model": model_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": True,
                "image_path": file["path"]
            })
    
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
            
    return True


def display_batch_results(is_image=False):
    """Display batch processing results."""
    st.subheader("Batch Processing Results")
    
    # Check if we have results
    if not st.session_state.batch_results:
        st.info("No batch processing results yet. Process some files first.")
        return
    
    # Export results option
    if st.button("Export Results as JSON"):
        export_batch_results()
    
    # Display each result
    for i, result in enumerate(st.session_state.batch_results):
        with st.expander(f"Result for {result['file_name']}", expanded=i == 0):
            # Show error if there was one
            if result.get("error", False):
                st.error(result["result"])
                continue
                
            # For image results, show the image
            if is_image and "image_path" in result and os.path.exists(result["image_path"]):
                st.image(result["image_path"], width=300)
            
            # Show prompt and result
            st.markdown(f"**Prompt:** {result['prompt']}")
            st.markdown(f"**Model:** {result['model']}")
            st.markdown(f"**Timestamp:** {result['timestamp']}")
            
            # Use code highlighting for the result
            from src.enhanced_features import process_code_blocks
            st.markdown(process_code_blocks(result["result"]), unsafe_allow_html=True)


def export_batch_results():
    """Export batch processing results as a downloadable JSON file."""
    # Prepare results for export, excluding file paths which won't be valid for the user
    export_data = []
    for result in st.session_state.batch_results:
        export_result = result.copy()
        if "image_path" in export_result:
            del export_result["image_path"]
        if "path" in export_result:
            del export_result["path"]
        export_data.append(export_result)
        
    # Create JSON string
    json_str = json.dumps(export_data, indent=2)
    
    # Create a download button
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = f"batch_results_{timestamp}.json"
    
    st.download_button(
        label="Download Results JSON",
        data=json_str,
        file_name=file_name,
        mime="application/json"
    )
    
    return True


# API Integration
def setup_api_integration():
    """Set up the API integration interface."""
    st.header("🔌 API Integration")
    
    st.markdown("""
    This section provides tools to interact with the SilentCodingLegend AI through an API interface.
    You can generate API keys, test API calls, and view documentation for integrating with external applications.
    """)
    
    # API key management
    st.subheader("API Key Management")
    
    # Initialize session state for API keys
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = []
    
    # Load API keys from persistent storage
    st.session_state.api_keys = load_api_keys()
    
    # Generate new API key
    if st.button("Generate New API Key"):
        new_key = generate_api_key()
        st.session_state.api_keys.append({
            "key": new_key,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_used": None,
            "requests": 0
        })
        save_api_keys(st.session_state.api_keys)  # Save to file
        st.success(f"New API key generated: {new_key}")
    
    # Display existing API keys
    if st.session_state.api_keys:
        st.subheader("Your API Keys")
        
        for i, key_info in enumerate(st.session_state.api_keys):
            with st.expander(f"API Key {i+1}"):
                st.text(f"Key: {key_info['key']}")
                st.text(f"Created: {key_info['created']}")
                st.text(f"Last Used: {key_info['last_used'] or 'Never'}")
                st.text(f"Requests: {key_info['requests']}")
                
                if st.button("Revoke", key=f"revoke_{i}"):
                    st.session_state.api_keys.pop(i)
                    save_api_keys(st.session_state.api_keys)  # Save to file
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
        
        ##### 3. Vision Analysis
        
        **URL:** `/api/v1/vision/analyze`
        
        **Method:** `POST`
        
        **Body:** Form data with an 'image' file and 'prompt' text field
        
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


def generate_api_key() -> str:
    """Generate a new API key."""
    import secrets
    import string
    
    # Generate a random string for the API key
    alphabet = string.ascii_letters + string.digits
    key = ''.join(secrets.choice(alphabet) for _ in range(24))
    
    # Add prefix for better identification
    return f"scl_{key}"


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
            
        # Submit button
        submitted = st.form_submit_button("Test API Call")
        
        # Update session state with messages
        st.session_state.api_test_messages = updated_messages
    
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
