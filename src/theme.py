"""
Theme module for SilentCodingLegend AI application.
Contains theme-related utility functions.
"""
import streamlit as st
from src.config import DARK_THEME, LIGHT_THEME, PRIMARY_COLOR

def toggle_theme():
    """Toggle between light and dark theme."""
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "dark"
    else:
        # Toggle theme
        st.session_state.ui_theme = "light" if st.session_state.ui_theme == "dark" else "dark"
    
def get_theme_colors():
    """Get the current theme colors."""
    # Initialize theme in session state if not already set
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "dark"
    
    # Get the current theme colors
    current_theme = DARK_THEME if st.session_state.ui_theme == "dark" else LIGHT_THEME
    
    # Extract colors from the current theme
    colors = {
        "background_color": current_theme["background_color"],
        "secondary_background_color": current_theme["secondary_background_color"],
        "text_color": current_theme["text_color"],
        "user_message_color": current_theme["user_message_color"],
        "assistant_message_color": current_theme["assistant_message_color"],
        "border_color": current_theme["border_color"],
        "input_background": current_theme["input_background"],
        "sidebar_color": current_theme["sidebar_color"],
        "code_background": current_theme["code_background"],
        "code_text": current_theme["code_text"],
        "primary_color": current_theme["primary_color"]
    }
    
    return colors

def apply_theme_style():
    """Apply custom styling based on the current theme."""
    colors = get_theme_colors()
    
    st.markdown(f"""
        <style>
        /* Theme base colors */
        :root {{
            --background-color: {colors["background_color"]};
            --secondary-background-color: {colors["secondary_background_color"]};
            --text-color: {colors["text_color"]};
            --user-message-color: {colors["user_message_color"]};
            --assistant-message-color: {colors["assistant_message_color"]};
            --accent-color: {colors["primary_color"]};
            --border-color: {colors["border_color"]};
            --input-background: {colors["input_background"]};
            --sidebar-color: {colors["sidebar_color"]};
            --code-background: {colors["code_background"]};
            --code-text: {colors["code_text"]};
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
            color: var(--text-color);
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
            background-color: var(--code-background);
            color: var(--code-text);
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 0.85em;
        }}
        
        /* Pre blocks */
        pre {{
            background-color: var(--code-background);
            border: 1px solid var(--border-color);
            border-radius: 5px;
            padding: 1em;
            overflow-x: auto;
        }}
        
        /* Syntax highlighting for different languages */
        .language-python {{ color: #4b8bbe; }}
        .language-javascript {{ color: #f7df1e; }}
        .language-html {{ color: #e34c26; }}
        .language-css {{ color: #563d7c; }}
        .language-bash {{ color: #89e051; }}
        .language-json {{ color: #3c4c65; }}
        
        /* Custom styling for the chat input */
        .stChatInput > div {{
            background-color: {colors["input_background"]} !important;
            border: 1px solid {colors["border_color"]} !important;
        }}
        .stChatInput input {{
            color: {colors["text_color"]} !important;
        }}
        .stChatInput button {{
            background-color: {colors["primary_color"]} !important;
        }}
        
        /* Theme toggle button styling */
        .theme-toggle {{
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1000;
            background-color: var(--accent-color);
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        
        /* Response code highlighting */
        .response-code {{
            background-color: var(--code-background);
            border-radius: 5px;
            padding: 1em;
            border-left: 3px solid var(--accent-color);
            margin: 1em 0;
        }}
        
        /* Model comparison container */
        .model-comparison {{
            display: flex;
            flex-wrap: wrap;
            gap: 1em;
        }}
        
        .model-response {{
            flex: 1;
            min-width: 300px;
            background-color: var(--secondary-background-color);
            border-radius: 8px;
            padding: 1em;
            border: 1px solid var(--border-color);
            margin-bottom: 1em;
        }}
        
        .model-header {{
            font-weight: bold;
            color: var(--accent-color);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5em;
            margin-bottom: 1em;
        }}
        
        /* Code copy button */
        .code-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: var(--code-background);
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            padding: 0.5em 1em;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .copy-button {{
            background-color: var(--accent-color);
            color: white;
            border: none;
            border-radius: 3px;
            padding: 0.25em 0.5em;
            font-size: 0.8em;
            cursor: pointer;
        }}
        </style>
    """, unsafe_allow_html=True)
