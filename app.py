"""
SilentCodingLegend AI Assistant
Main application file for the Streamlit interface
"""

import streamlit as st
import time
import os

# Add debugging tools
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from the modular structure
from src.config import (
    APP_NAME, APP_VERSION, APP_ICON, APP_DESCRIPTION, CURRENT_YEAR,
    MODEL_INFO, MODEL_CATEGORIES, DEFAULT_MODEL, DEFAULT_CATEGORY, DEFAULT_TEMPERATURE, 
    DEFAULT_MAX_TOKENS, DEFAULT_TOP_P, GROQ_API_KEY
)
from src.utils import apply_custom_style, get_groq_client, query_groq_model, get_system_prompt, create_directories
from src.performance_opt import cached_query_groq_model, cached_query_groq_vision_model, response_cache
from src.chat_templates import get_all_templates, get_all_categories, get_templates_by_category, get_template_by_id
from src.export_utils import create_export_ui
from src.mobile_enhanced import (
    apply_mobile_optimizations,
    create_mobile_friendly_container,
    display_mobile_footer
)
from src.mobile_optimization import create_adaptive_container, get_device_type, create_mobile_navigation

# Ensure necessary directories exist
create_directories()

# Set page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/silentcodinglegend',
        'Report a bug': 'https://github.com/silentcodinglegend/issues',
        'About': f'{APP_NAME} - {APP_DESCRIPTION}'
    }
)

# Apply the custom styling from utils module
from src.utils import apply_custom_style
apply_custom_style()

# Apply mobile optimizations
from src.mobile_enhanced import apply_mobile_optimizations, create_mobile_friendly_container
device_type = apply_mobile_optimizations()

# Add enhanced features
from src.enhanced_features import add_theme_toggle, setup_model_comparison, process_code_blocks, compare_models
add_theme_toggle()
setup_model_comparison()

# Initialize Groq client
client = get_groq_client()

# Initialize session state for conversation history and model selection
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize model settings with defaults from config
if "model" not in st.session_state:
    st.session_state.model = DEFAULT_MODEL
if "category" not in st.session_state:
    st.session_state.category = DEFAULT_CATEGORY
    
# Initialize template settings
if "current_template" not in st.session_state:
    st.session_state.current_template = None
if "template_category" not in st.session_state:
    st.session_state.template_category = "all"

# App header with dark theme styling
st.title(f"💻 {APP_NAME}")
st.markdown(f"""
    <div style="padding: 10px; border-radius: 10px; background-color: #2d2d2d; margin-bottom: 20px; border-left: 4px solid #7e57c2;">
        <p style="color: #e0e0e0; margin: 0;">{APP_DESCRIPTION}</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with model settings and options
with st.sidebar:
    st.markdown("""
        <h2 style="color: #7e57c2; margin-bottom: 20px; text-align: center;">Settings</h2>
    """, unsafe_allow_html=True)
    
    # Create adaptive containers based on device type
    templates_container = create_adaptive_container(content_type="settings", mobile_collapsed=device_type=="mobile")
    
    # Chat Templates Section
    with templates_container:
        st.markdown("""
            <h3 style="color: #7e57c2; margin-bottom: 10px;">Chat Templates</h3>
        """, unsafe_allow_html=True)
    
    # Get all template categories and add "all" option
    template_categories = ["all"] + get_all_categories()
    
    # Template category selector
    selected_category = st.selectbox(
        "Template Category",
        template_categories,
        index=template_categories.index(st.session_state.template_category) if st.session_state.template_category in template_categories else 0,
        format_func=lambda x: x.capitalize() if x != "all" else "All Categories"
    )
    st.session_state.template_category = selected_category
    
    # Get templates for the selected category
    if selected_category == "all":
        templates = get_all_templates()
    else:
        templates = get_templates_by_category(selected_category)
    
    # Template selection
    template_options = ["No template"] + [f"{t.icon} {t.title}" for t in templates.values()]
    template_keys = [None] + list(templates.keys())
    
    current_index = 0
    if st.session_state.current_template in template_keys:
        current_index = template_keys.index(st.session_state.current_template)
    
    selected_template_name = st.selectbox(
        "Select a template",
        options=template_options,
        index=current_index,
        help="Choose a predefined template for your conversation"
    )
    
    # Get the template key from the selected name
    selected_index = template_options.index(selected_template_name)
    template_key = template_keys[selected_index]
    
    # Update the current template in session state
    if template_key != st.session_state.current_template:
        st.session_state.current_template = template_key
        
        # If a template was selected, update session state
        if template_key:
            template = templates[template_key]
            
            # Show template info
            st.markdown(f"""
                <div style="background-color: #2d2d2d; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 2px solid #7e57c2;">
                    <p style="color: #e0e0e0; font-size: 0.9em; margin: 0;"><strong>{template.title}</strong></p>
                    <p style="color: #b39ddb; font-size: 0.85em; margin: 5px 0 0 0;">{template.description}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Template action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Apply Template", key="apply_template"):
                    # Reset conversation
                    st.session_state.messages = []
                    
                    # Start with system message (not shown in UI)
                    if template.system_prompt:
                        st.session_state.system_prompt = template.system_prompt
                    
                    # Add any initial messages
                    if template.initial_messages:
                        st.session_state.messages.extend(template.initial_messages)
                    
                    # Set recommended model if provided
                    if template.recommended_model:
                        for cat, models in MODEL_CATEGORIES.items():
                            if template.recommended_model in models:
                                st.session_state.category = cat
                                st.session_state.model = template.recommended_model
                                break
                    
                    # Set recommended temperature
                    st.session_state.temperature = template.recommended_temperature
                    
                    st.success(f"Template '{template.title}' applied! Start chatting.")
                    st.rerun()
                    
            with col2:
                if st.button("View Example", key="view_template_example"):
                    # Show the suggested first prompt
                    if template.suggested_first_prompt:
                        st.info(f"Suggested prompt: '{template.suggested_first_prompt}'")
    
    # Add model category selection
    st.markdown('<p style="color: #b39ddb; font-weight: bold; margin-top: 20px;">Model Category:</p>', unsafe_allow_html=True)
    
    category = st.selectbox(
        "Model Category",
        list(MODEL_CATEGORIES.keys()),
        index=list(MODEL_CATEGORIES.keys()).index(st.session_state.category) if st.session_state.category in MODEL_CATEGORIES else 0,
        key="model_category",
        label_visibility="collapsed"
    )
    st.session_state.category = category
    
    # Model selection with custom styling
    st.markdown('<p style="color: #b39ddb; font-weight: bold; margin-top: 15px;">Select Groq model:</p>', unsafe_allow_html=True)
    
    # Get models for the selected category
    category_models = MODEL_CATEGORIES[category]
    display_options = []
    model_keys = []
    
    for model_key in category_models:
        if model_key in MODEL_INFO:
            display_name = MODEL_INFO[model_key]["display_name"]
            display_options.append(display_name)
            model_keys.append(model_key)
    
    selected_display = st.selectbox(
        "Groq model",
        display_options,
        index=0 if "model" not in st.session_state else (model_keys.index(st.session_state.model) if st.session_state.model in model_keys else 0),
        label_visibility="collapsed"
    )
    
    # Get selected model key
    selected_index = display_options.index(selected_display)
    model = model_keys[selected_index]
    st.session_state.model = model
    
    # Display model limits
    if model in MODEL_INFO:
        st.markdown(f"""
            <div style="background-color: #2d2d2d; padding: 10px; border-radius: 5px; margin-top: 10px; border-left: 2px solid #7e57c2;">
                <p style="color: #e0e0e0; font-size: 0.9em; margin: 0;"><strong>Limits:</strong></p>
                <p style="color: #b39ddb; font-size: 0.85em; margin: 5px 0 0 0;">• {MODEL_INFO[model]["req_limit"]}</p>
                <p style="color: #b39ddb; font-size: 0.85em; margin: 2px 0 0 0;">• {MODEL_INFO[model]["token_limit"]}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Model Information Section
    with st.expander("ℹ️ Model Details", expanded=False):
        st.markdown(f"""
            <div style="color: #e0e0e0;">
                <p><strong>{MODEL_INFO[model]['display_name']}</strong> model information:</p>
                <ul style="margin-top: 5px; padding-left: 20px;">
                    <li>Model ID: <code style="background-color: #333; padding: 2px 5px; border-radius: 3px;">{model}</code></li>
                    <li>Request limit: {MODEL_INFO[model]['req_limit']}</li>
                    <li>Token throughput: {MODEL_INFO[model]['token_limit']}</li>
                </ul>
                <p style="margin-top: 10px; font-size: 0.85em;">For optimal performance, consider the token limits when configuring max tokens setting.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Temperature control with custom styling
    st.markdown('<p style="color: #b39ddb; font-weight: bold; margin-top: 20px;">Temperature:</p>', unsafe_allow_html=True)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.1, label_visibility="collapsed")
    st.caption("Controls creativity (higher = more creative)")
    
    # Max tokens control with custom styling
    st.markdown('<p style="color: #b39ddb; font-weight: bold; margin-top: 20px;">Max tokens:</p>', unsafe_allow_html=True)
    
    # Get recommended max tokens for the selected model
    from src.config import get_recommended_max_tokens
    recommended_max = get_recommended_max_tokens(model)
    max_value = min(8192, int(recommended_max * 1.5))  # Cap at 8192 or 1.5x recommended
    
    max_tokens = st.slider(
        "Max tokens", 
        min_value=100, 
        max_value=max_value, 
        value=recommended_max,
        step=100,
        label_visibility="collapsed"
    )
    
    st.caption(f"Controls response length (recommended: {recommended_max})")
    
    # Advanced options section
    advanced_container = create_adaptive_container(content_type="options", mobile_collapsed=True)
    with advanced_container:
        st.markdown('<p style="color: #b39ddb; font-weight: bold;">🔧 Advanced Options</p>', unsafe_allow_html=True)
        # Model comparison option
        st.markdown('<p style="color: #b39ddb; font-weight: bold;">Compare models:</p>', unsafe_allow_html=True)
        if "enable_comparison" not in st.session_state:
            st.session_state.enable_comparison = False
            
        st.session_state.enable_comparison = st.toggle(
            "Enable model comparison",
            value=st.session_state.enable_comparison,
            help="Compare responses from multiple models for the same prompt"
        )
        
        if st.session_state.enable_comparison:
            st.markdown('<p style="color: #b39ddb; font-size: 0.85em;">Models to compare:</p>', unsafe_allow_html=True)
            models_for_comparison = []
            
            # Let user select models to compare
            all_models = ["llama3-70b-8192", "llama-3.1-8b-instant", "llama-4-scout-instruct", "mistral-saba-24b", "groq-compound-beta"]
            
            for model_id in all_models:
                if model_id in MODEL_INFO:
                    if st.checkbox(
                        MODEL_INFO[model_id]["display_name"],
                        value=model_id == st.session_state.model,
                        key=f"compare_{model_id}"
                    ):
                        models_for_comparison.append(model_id)
            
            # Store the selected models
            st.session_state.models_for_comparison = models_for_comparison
            
            if len(models_for_comparison) > 3:
                st.warning("Comparing more than 3 models may take longer and consume more resources.")
        
        st.divider()
        
        st.markdown('<p style="color: #b39ddb; font-weight: bold;">Output format:</p>', unsafe_allow_html=True)
        
        # Initialize session state if needed
        if "output_format" not in st.session_state:
            st.session_state.output_format = "Text"
        if "top_p" not in st.session_state:
            st.session_state.top_p = DEFAULT_TOP_P
            
        # Performance optimization settings
        st.divider()
        st.markdown('<p style="color: #b39ddb; font-weight: bold;">Performance settings:</p>', unsafe_allow_html=True)
        
        # Initialize cache settings if needed
        if "cache_enabled" not in st.session_state:
            st.session_state.cache_enabled = True
        
        # Response caching toggle
        cache_enabled = st.toggle(
            "Enable response caching",
            value=st.session_state.cache_enabled,
            help="Cache responses to improve performance for repeated queries"
        )
        
        if cache_enabled != st.session_state.cache_enabled:
            st.session_state.cache_enabled = cache_enabled
            status = "enabled" if cache_enabled else "disabled"
            st.success(f"Response caching {status}")
        
        st.caption("Caching improves performance but may use older responses")
        
        st.divider()
        
        output_format = st.radio(
            "Output format",
            ["Text", "JSON"],
            index=0 if st.session_state.output_format == "Text" else 1,
            horizontal=True,
            key="output_format_radio",
            help="JSON mode forces the model to return responses in valid JSON format",
            label_visibility="collapsed"
        )
        st.session_state.output_format = output_format
        
        if output_format == "JSON":
            st.markdown("""
                <div style="background-color: #2d2d2d; padding: 10px; border-radius: 5px; margin-top: 10px; border-left: 2px solid #ff9800;">
                    <p style="color: #e0e0e0; font-size: 0.9em; margin: 0;"><strong>Note:</strong></p>
                    <p style="color: #e0e0e0; font-size: 0.85em; margin: 5px 0 0 0;">• JSON mode doesn't support streaming</p>
                    <p style="color: #e0e0e0; font-size: 0.85em; margin: 2px 0 0 0;">• All responses will be structured as JSON objects</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown('<p style="color: #b39ddb; font-weight: bold; margin-top: 15px;">Top-p sampling:</p>', unsafe_allow_html=True)
        top_p = st.slider(
            "Top-p sampling",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.top_p,
            step=0.1,
            key="top_p_slider",
            help="Controls diversity via nucleus sampling (lower values = more focused outputs)",
            label_visibility="collapsed"
        )
        st.session_state.top_p = top_p
    
    # Export conversation section
    if st.session_state.messages:
        st.divider()
        st.markdown('<p style="color: #b39ddb; font-weight: bold;">Export conversation:</p>', unsafe_allow_html=True)
        export_container = create_adaptive_container(content_type="options", mobile_collapsed=True)
        with export_container:
            st.markdown('<p style="color: #b39ddb; font-weight: bold;">📤 Export Options</p>', unsafe_allow_html=True)
            # Get current conversation title from template or use default
            default_title = "SilentCodingLegend AI Chat"
            if st.session_state.current_template:
                template = get_template_by_id(st.session_state.current_template)
                if template:
                    default_title = f"{template.title} - Chat"
            
            # Create export UI
            create_export_ui(st.session_state.messages, title=default_title)
    
    st.divider()
    
    # About section with dark theme styling
    st.markdown(f"""
        <div style="background-color: #2d2d2d; padding: 15px; border-radius: 10px; border-left: 3px solid #7e57c2;">
            <h3 style="color: #7e57c2; margin-top: 0;">About</h3>
            <p style="color: #e0e0e0;">{APP_NAME} is an AI assistant that helps with coding tasks, answering technical questions, and providing programming guidance.</p>
            <p style="color: #e0e0e0; text-align: center; margin-top: 15px;">Built with <span style="color: #ff5252;">❤️</span> using Groq API and Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

# Display chat history with enhanced code highlighting
from src.enhanced_features import process_code_blocks
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        if role == "assistant":
            # Process code blocks for better highlighting
            st.markdown(process_code_blocks(content), unsafe_allow_html=True)
        else:
            st.markdown(content)

# User input for chat
user_prompt = st.chat_input(f"Ask {APP_NAME} something...")

if user_prompt:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Display a spinner while waiting for the API response
        with st.spinner(f"⚡ {APP_NAME} is thinking..."):
            try:
                # Get the system prompt with optional JSON mode
                json_mode = st.session_state.get("output_format") == "JSON"
                system_prompt = get_system_prompt(json_mode=json_mode)
                
                # Prepare messages for the API call
                messages = [
                    {"role": "system", "content": system_prompt},
                ]
                
                # Add conversation history
                for msg in st.session_state.messages:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Query the Groq model with caching if enabled
                use_cache = st.session_state.get("cache_enabled", True)
                response = cached_query_groq_model(
                    model_id=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=not json_mode,
                    json_mode=json_mode,
                    top_p=st.session_state.get("top_p", DEFAULT_TOP_P),
                    use_cache=use_cache,
                    template_id=st.session_state.get("current_template")
                )
                
                # Handle the response based on format
                if json_mode:
                    # For JSON mode (non-streaming)
                    full_response = response.choices[0].message.content
                    message_placeholder.markdown("```json\n" + full_response + "\n```")
                else:
                    # For streaming text response
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                            time.sleep(0.01)
                    
                    # Update with complete response with code highlighting
                    message_placeholder.markdown(process_code_blocks(full_response), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                full_response = f"I apologize, but I encountered an error: {str(e)}"
                message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Check if model comparison is enabled
    if st.session_state.get("compare_models", False) and len(st.session_state.get("comparison_models", [])) >= 1:
        st.markdown("---")
        st.markdown("### Model Comparison")
        
        # Use the same prompt for comparison with other models
        compare_models(
            prompt=user_prompt,
            model_ids=st.session_state.comparison_models,
            temperature=temperature,
            max_tokens=max_tokens
        )

# Footer with dark theme styling
st.divider()
st.markdown(f"""
    <div style="text-align: center; padding: 10px; color: #999999; font-size: 0.8em; margin-top: 30px;">
        <p>© {CURRENT_YEAR} {APP_NAME}</p>
        <p style="margin-top: 10px;">
            <a href="https://groq.com" target="_blank" rel="noopener noreferrer">
                <img 
                    src="https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg" 
                    alt="Powered by Groq for fast inference."
                    style="height: 40px; margin-bottom: 10px;"
                />
            </a>
        </p>
        <p style="font-size: 0.9em; color: #666666;">Dark theme edition | {len(MODEL_INFO)} Groq Models Supported</p>
        <p style="margin-top: 10px;">
            <a href="/Chat_History" target="_self" style="color: #7e57c2; text-decoration: none; font-size: 0.9em; margin-right: 20px;">
                View Chat History 💬
            </a>
            <a href="/AdvancedTools" target="_self" style="color: #7e57c2; text-decoration: none; font-size: 0.9em; margin-right: 20px;">
                Advanced Tools 🛠️
            </a>
            <a href="/PerformanceOptimizations" target="_self" style="color: #7e57c2; text-decoration: none; font-size: 0.9em;">
                Performance ⚡
            </a>
        </p>
    </div>
""", unsafe_allow_html=True)

# Function to clear cached state
def clear_cache():
    logger.info("Clearing cached state...")
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Clear response cache
    if "response_cache" in globals():
        logger.info("Clearing response cache...")
        response_cache.memory_cache = {}
        response_cache.clear(max_age_hours=0)  # Clear all disk cache
    
    # Reset the app to initial state
    st.rerun()

# Add a button to clear cache (for debugging)
if st.button("Clear Cache"):
    clear_cache()

# Add mobile navigation for small screens
device_type = get_device_type()
if device_type == "mobile":
    create_mobile_navigation()
