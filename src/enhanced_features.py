"""
Enhanced features for SilentCodingLegend AI.
This module contains functions for theme toggle, model comparison, and response highlighting.
"""

import streamlit as st
import re

def add_theme_toggle():
    """Add a theme toggle button to the app."""
    # Initialize theme in session state if not already set
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "dark"
    
    theme = st.session_state.ui_theme
    theme_icon = "🌙" if theme == "dark" else "☀️"
    theme_text = "Switch to Light Mode" if theme == "dark" else "Switch to Dark Mode"
    
    # Create a container for the button in the sidebar
    with st.sidebar:
        if st.button(f"{theme_icon} {theme_text}", key="theme_toggle", use_container_width=True):
            # Toggle the theme
            st.session_state.ui_theme = "light" if theme == "dark" else "dark"
            # This will trigger a rerun of the app
            st.rerun()

def process_code_blocks(response_text):
    """Process code blocks in the response text for better highlighting."""
    # Early return if content is None or empty
    if not response_text:
        return response_text
    
    import uuid
    
    try:
        # Import pygments modules now that we've installed them
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name, guess_lexer
        from pygments.formatters import HtmlFormatter
        from pygments.util import ClassNotFound
        
        # Pattern to match markdown code blocks with language specification
        # For example: ```python ... ``` or ```javascript ... ```
        pattern = r'```(\w+)?\n([\s\S]*?)\n```'
        
        def replace_code_block(match):
            language = match.group(1) or ''
            code = match.group(2)
            block_id = f"code-block-{uuid.uuid4()}"
            
            # Add default language if not specified
            if not language:
                try:
                    lexer = guess_lexer(code)
                    language = lexer.name.lower()
                except ClassNotFound:
                    language = 'text'  # Default fallback
                    
            # Format with pygments if language is valid
            try:
                lexer = get_lexer_by_name(language.lower(), stripall=True)
                formatter = HtmlFormatter(style='monokai', cssclass=f'codehilite language-{language}')
                highlighted_code = highlight(code, lexer, formatter)
                
                # Add language tag and copy button
                language_display = language.capitalize()
                result = f"""
                <div class="code-block" id="{block_id}">
                    <div class="code-header">
                        <span class="code-language">{language_display}</span>
                        <button class="copy-button" onclick="copyCode('{block_id}')">Copy</button>
                    </div>
                    {highlighted_code}
                </div>
                """
                return result
            except ClassNotFound:
                # If language isn't recognized, use a simple format
                return f"""
                <div class="code-block" id="{block_id}">
                    <div class="code-header">
                        <span class="code-language">Code</span>
                        <button class="copy-button" onclick="copyCode('{block_id}')">Copy</button>
                    </div>
                    <pre><code>{code}</code></pre>
                </div>
                """
        
        # Replace all code blocks with highlighted versions
        result = re.sub(pattern, replace_code_block, response_text)
        
        # Add CSS and JavaScript for styling and functionality
        result += """
        <style>
        .code-block {
            margin: 1rem 0;
            border-radius: 0.5rem;
            overflow: hidden;
            border: 1px solid var(--border-color, #333333);
        }
        .code-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 1rem;
            background-color: var(--secondary-background-color, #1e1e1e);
            border-bottom: 1px solid var(--border-color, #333333);
        }
        .code-language {
            color: #7e57c2;
            font-weight: 500;
        }
        .copy-button {
            background-color: transparent;
            border: 1px solid var(--border-color, #333333);
            border-radius: 0.25rem;
            padding: 0.25rem 0.5rem;
            color: var(--text-color, #e0e0e0);
            cursor: pointer;
            font-size: 0.8rem;
        }
        .copy-button:hover {
            background-color: rgba(126, 87, 194, 0.1);
        }
        .codehilite {
            margin: 0;
            padding: 1rem;
            overflow-x: auto;
            background-color: var(--code-background, #2b2b2b) !important;
        }
        .codehilite code {
            color: var(--code-text, #e6e6e6) !important;
        }
        </style>
        <script>
        function copyCode(blockId) {
            const codeBlock = document.getElementById(blockId).querySelector('pre');
            const code = codeBlock.textContent;
            const button = document.getElementById(blockId).querySelector('.copy-button');
            
            navigator.clipboard.writeText(code).then(
                function() {
                    // Success - change button text temporarily
                    const originalText = button.textContent;
                    button.textContent = "Copied!";
                    setTimeout(function() {
                        button.textContent = originalText;
                    }, 1500);
                },
                function() {
                    // Failure
                    button.textContent = "Failed to copy";
                    setTimeout(function() {
                        button.textContent = "Copy";
                    }, 1500);
                }
            );
        }
        </script>
        """
        
        return result
    except ImportError:
        # If pygments is not installed, return the original text
        return response_text
        
        def replace_code_block(match):
            language = match.group(1) or ''
            code = match.group(2)
            
            # Add default language if not specified
            if not language:
                try:
                    lexer = guess_lexer(code)
                    language = lexer.name.lower()
                except ClassNotFound:
                    language = 'text'  # Default fallback
                    
            # Format with pygments if language is valid
            try:
                lexer = get_lexer_by_name(language.lower(), stripall=True)
                formatter = HtmlFormatter(style='monokai', cssclass=f'codehilite language-{language}')
                highlighted_code = highlight(code, lexer, formatter)
                
                # Add language tag and copy button
                language_display = language.capitalize()
                result = f"""
                <div class="code-block">
                    <div class="code-header">
                        <span class="code-language">{language_display}</span>
                        <button class="copy-button" onclick="copyCode(this)">Copy</button>
                    </div>
                    {highlighted_code}
                </div>
                """
                return result
            except ClassNotFound:
                # If language isn't recognized, use a simple format
                return f"""
                <div class="code-block">
                    <div class="code-header">
                        <span class="code-language">Code</span>
                        <button class="copy-button" onclick="copyCode(this)">Copy</button>
                    </div>
                    <pre><code>{code}</code></pre>
                </div>
                """
        
        # Replace all code blocks with highlighted versions
        result = re.sub(pattern, replace_code_block, response_text)
        
        # Add CSS and JavaScript for styling and functionality
        result += """
        <style>
        .code-block {
            margin: 1rem 0;
            border-radius: 0.5rem;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }
        .code-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 1rem;
            background-color: var(--secondary-background-color);
            border-bottom: 1px solid var(--border-color);
        }
        .code-language {
            color: #7e57c2;
            font-weight: 500;
        }
        .copy-button {
            background-color: transparent;
            border: 1px solid var(--border-color);
            border-radius: 0.25rem;
            padding: 0.25rem 0.5rem;
            color: var(--text-color);
            cursor: pointer;
            font-size: 0.8rem;
        }
        .copy-button:hover {
            background-color: rgba(126, 87, 194, 0.1);
        }
        .codehilite {
            margin: 0;
            padding: 1rem;
            overflow-x: auto;
            background-color: var(--code-background) !important;
        }
        .codehilite code {
            color: var(--code-text) !important;
        }
        </style>
        <script>
        function copyCode(button) {
            const codeBlock = button.parentElement.nextElementSibling;
            const code = codeBlock.textContent;
            
            navigator.clipboard.writeText(code).then(
                function() {
                    // Success - change button text temporarily
                    const originalText = button.textContent;
                    button.textContent = "Copied!";
                    setTimeout(function() {
                        button.textContent = originalText;
                    }, 1500);
                },
                function() {
                    // Failure
                    button.textContent = "Failed to copy";
                    setTimeout(function() {
                        button.textContent = "Copy";
                    }, 1500);
                }
            );
        }
        </script>
        """
        
        return result
    except ImportError:
        # If pygments is not installed, return the original text
        return response_text
    
    # Function to create HTML for a code block with language and copy button
    def replace_code_block(match):
        lang = match.group(1) or ""
        code = match.group(2)
        
        # Create a unique ID for this code block
        import uuid
        block_id = f"code-block-{uuid.uuid4()}"
        
        # HTML for code block with language name and copy button
        html = f"""
        <div class="response-code" id="{block_id}">
            <div class="code-header">
                <span class="language-{lang}">{lang}</span>
                <button class="copy-button" onclick="copyCode('{block_id}')">Copy</button>
            </div>
            <pre><code class="language-{lang}">{code}</code></pre>
        </div>
        """
        return html
    
    # Replace code blocks with HTML versions
    highlighted_text = re.sub(pattern, replace_code_block, response_text)
    
    # Add JavaScript for copy functionality
    copy_script = """
    <script>
    function copyCode(blockId) {
        const codeBlock = document.getElementById(blockId).querySelector('code');
        const textArea = document.createElement('textarea');
        textArea.value = codeBlock.textContent;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        
        // Show "Copied" message
        const button = document.getElementById(blockId).querySelector('.copy-button');
        const originalText = button.textContent;
        button.textContent = "Copied!";
        setTimeout(() => {
            button.textContent = originalText;
        }, 2000);
    }
    </script>
    """
    
    return highlighted_text + copy_script

def model_comparison(prompt, model_list=None, temperature=0.7, max_tokens=1024):
    """
    Compare responses from different models for the same prompt.
    Returns a dictionary of model_id -> response.
    """
    if model_list is None:
        # Default to some good models for comparison
        model_list = ["llama3-70b-8192", "llama-3.1-8b-instant", "llama-4-scout-instruct"]
    
    st.write("### Model Comparison")
    st.write(f"Comparing model responses for: *{prompt}*")
    
    from src.utils import query_groq_model
    
    # Create a container for all responses
    comparison_container = st.container()
    
    with comparison_container:
        # Create columns for the models
        if len(model_list) == 2:
            col1, col2 = st.columns(2)
            cols = [col1, col2]
        elif len(model_list) == 3:
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
        else:
            # For more than 3, use a 2-column layout
            cols = st.columns(2)
        
        # Display a loading spinner while getting responses
        with st.spinner("Comparing model responses..."):
            responses = {}
            
            # Prepare the messages
            system_prompt = "You are a helpful AI assistant. Provide a clear and concise response."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Get responses from each model
            for i, model_id in enumerate(model_list):
                # Query the model
                try:
                    response = query_groq_model(
                        model_id=model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False  # No streaming for comparison
                    )
                    
                    # Extract the response content
                    response_text = response.choices[0].message.content
                    responses[model_id] = response_text
                    
                    # Show in the appropriate column
                    with cols[i % len(cols)]:
                        st.markdown(f"#### {model_id}")
                        st.markdown(process_code_blocks(response_text), unsafe_allow_html=True)
                
                except Exception as e:
                    with cols[i % len(cols)]:
                        st.error(f"Error with {model_id}: {str(e)}")
            
            return responses

def setup_model_comparison():
    """Set up model comparison feature in the sidebar."""
    with st.sidebar:
        with st.expander("🔄 Compare Models", expanded=False):
            st.write("Compare responses from different models for the same prompt.")
            
            # Initialize the comparison settings in session state
            if "compare_models" not in st.session_state:
                st.session_state.compare_models = False
            if "comparison_models" not in st.session_state:
                st.session_state.comparison_models = []
                
            # Toggle for enabling comparison
            st.session_state.compare_models = st.toggle(
                "Enable Comparison", 
                value=st.session_state.compare_models,
                help="Generate responses from multiple models simultaneously"
            )
            
            if st.session_state.compare_models:
                # Get all available models from config
                from src.config import MODEL_INFO
                
                # Allow selecting multiple models for comparison
                available_models = list(MODEL_INFO.keys())
                display_names = [MODEL_INFO[m].get("display_name", m) for m in available_models]
                
                # Multi-select for models
                selected_indices = [i for i, m in enumerate(available_models) if m in st.session_state.comparison_models]
                
                selected_display_names = st.multiselect(
                    "Models to Compare",
                    options=display_names,
                    default=[display_names[i] for i in selected_indices] if selected_indices else [],
                    help="Select 2-3 models to compare (more may slow down responses)"
                )
                
                # Update the selected models in session state
                st.session_state.comparison_models = [
                    available_models[display_names.index(name)]
                    for name in selected_display_names if name in display_names
                ]
                
                # Warning if too many models are selected
                if len(st.session_state.comparison_models) > 3:
                    st.warning("⚠️ Comparing many models may slow down responses.")

def compare_models(prompt, model_ids, temperature=0.7, max_tokens=2048):
    """Generate and compare responses from multiple models."""
    from src.config import MODEL_INFO
    from src.utils import query_groq_model, get_system_prompt
    
    if not model_ids:
        st.warning("Please select at least one model for comparison.")
        return {}
    
    # Create tabs for each model
    tabs = st.tabs([MODEL_INFO[model]["display_name"] for model in model_ids])
    responses = {}
    
    # Get system prompt
    system_prompt = get_system_prompt()
    
    # Prepare messages for the API call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Query each model and display results in tabs
    for i, model_id in enumerate(model_ids):
        with tabs[i]:
            with st.spinner(f"Generating response from {MODEL_INFO[model_id]['display_name']}..."):
                try:
                    # Query the model
                    response = query_groq_model(
                        model_id=model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False  # No streaming for comparison
                    )
                    
                    # Extract and format response
                    full_response = response.choices[0].message.content
                    responses[model_id] = full_response
                    
                    # Display formatted response with code highlighting
                    st.markdown(process_code_blocks(full_response), unsafe_allow_html=True)
                    
                    # Show response metrics
                    st.divider()
                    completion_tokens = response.usage.completion_tokens
                    prompt_tokens = response.usage.prompt_tokens
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Completion Tokens", completion_tokens)
                    col2.metric("Prompt Tokens", prompt_tokens)
                    col3.metric("Total Tokens", completion_tokens + prompt_tokens)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    responses[model_id] = f"Error: {str(e)}"
    
    return responses
