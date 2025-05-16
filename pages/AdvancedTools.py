"""
Advanced Tools page for SilentCodingLegend AI.
This page contains the code execution environment, batch processing, and API integration tools.
"""

import streamlit as st

# Import configuration and utilities
from src.config import (
    APP_NAME, APP_DESCRIPTION, CURRENT_YEAR
)
from src.utils import apply_custom_style
from src.advanced_features import (
    setup_code_execution, 
    setup_batch_processing
)
from src.api_integration import setup_enhanced_api_integration

# Apply custom styling
apply_custom_style()

# Add enhanced features
from src.enhanced_features import add_theme_toggle
add_theme_toggle()

# Page header
st.title("🛠️ Advanced Tools")
st.markdown("""
    <div style="padding: 10px; border-radius: 10px; background-color: #2d2d2d; margin-bottom: 20px; border-left: 4px solid #7e57c2;">
        <p style="color: #e0e0e0; margin: 0;">Advanced tools for code execution, batch processing, and API integration</p>
    </div>
""", unsafe_allow_html=True)

# Add help section with documentation
with st.expander("ℹ️ How to use these features", expanded=False):
    st.markdown("""
    ## Advanced Features Documentation
    
    This page provides access to advanced tools that extend the capabilities of SilentCodingLegend AI.
    
    - **💻 Code Execution**: Write and run code directly in the browser in Python, JavaScript, or Bash
    - **🔄 Batch Processing**: Process multiple documents or images with a single prompt
    - **🔌 API Integration**: Generate API keys and test API endpoints for integration with other applications
    
    For detailed documentation on how to use these features, please refer to the [Advanced Features Guide](/docs/advanced_features_guide.md).
    """)
    
    # Display key tips for each feature
    st.subheader("Quick Tips")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Code Execution**")
        st.markdown("""
        - Use sample templates to get started
        - Code runs in a sandboxed environment
        - 15 second timeout for safety
        """)
    
    with col2:
        st.markdown("**Batch Processing**")
        st.markdown("""
        - Supports multiple file formats
        - Be specific in your processing prompts
        - Results can be exported as JSON
        """)
        
    with col3:
        st.markdown("**API Integration**")
        st.markdown("""
        - API keys persist between sessions
        - Test endpoints before integration
        - Keep keys secure and rotate periodically
        """)
    
    # Add performance optimization link
    st.markdown("""
    ### ⚡ Performance Optimizations
    
    For advanced performance tools like caching, lazy loading, and background processing, 
    visit the [Performance Optimizations](/PerformanceOptimizations) page.
    """)

# Create tabs for different advanced features
tabs = st.tabs([
    "💻 Code Execution", 
    "🔄 Batch Processing", 
    "🔌 API Integration"
])

# Code Execution Environment
with tabs[0]:
    setup_code_execution()

# Batch Processing
with tabs[1]:
    setup_batch_processing()

# API Integration
with tabs[2]:
    setup_enhanced_api_integration()

# Footer
st.divider()
st.markdown(f"""
    <div style="text-align: center; padding: 10px; color: #999999; font-size: 0.8em; margin-top: 30px;">
        <p>© {CURRENT_YEAR} {APP_NAME} - Advanced Tools</p>
        <p style="font-size: 0.9em; color: #666666;">Powered by Groq API</p>
    </div>
""", unsafe_allow_html=True)
