"""
Performance Optimizations page for SilentCodingLegend AI.
This page provides tools to optimize the application's performance through caching,
lazy loading, and background task management.
"""

import streamlit as st

# Import configuration and utilities
from src.config import (
    APP_NAME, APP_DESCRIPTION, CURRENT_YEAR
)
from src.utils import apply_custom_style

# Import performance optimization modules
from src.performance_opt import (
    setup_performance_optimization,
    response_cache, 
    lazy_load_manager,
    bg_task_manager
)

# Apply custom styling
apply_custom_style()

# Add enhanced features
from src.enhanced_features import add_theme_toggle
add_theme_toggle()

# Page header
st.title("⚡ Performance Optimizations")
st.markdown("""
    <div style="padding: 10px; border-radius: 10px; background-color: #2d2d2d; margin-bottom: 20px; border-left: 4px solid #7e57c2;">
        <p style="color: #e0e0e0; margin: 0;">Tools to optimize performance through caching, lazy loading, and background processing</p>
    </div>
""", unsafe_allow_html=True)

# Add help section with documentation
with st.expander("ℹ️ How to use performance optimizations", expanded=False):
    st.markdown("""
    ## Performance Optimization Documentation
    
    This page provides access to performance optimization tools for SilentCodingLegend AI.
    
    - **⚡ Response Caching**: Cache API responses to reduce redundant calls and improve response time
    - **🚀 Lazy Loading**: Load UI components and resources only when needed
    - **⏱️ Background Processing**: Run tasks in the background without blocking the UI
    
    Using these features will significantly improve the application's performance and responsiveness.
    """)
    
    # Display key tips for each feature
    st.subheader("Quick Tips")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Response Caching**")
        st.markdown("""
        - Responses are cached for 24 hours by default
        - Streaming responses are never cached
        - Great for repetitive queries with same parameters
        """)
    
    with col2:
        st.markdown("**Lazy Loading**")
        st.markdown("""
        - Sections load only when requested
        - Use for heavy UI components
        - Helps reduce initial loading time
        """)
        
    with col3:
        st.markdown("**Background Processing**")
        st.markdown("""
        - Run time-intensive tasks in the background
        - Monitor task progress in real-time
        - Results are stored for later retrieval
        """)

# Set up the performance optimization UI
setup_performance_optimization()

# Footer
st.divider()
st.markdown(f"""
    <div style="text-align: center; padding: 10px; color: #999999; font-size: 0.8em; margin-top: 30px;">
        <p>© {CURRENT_YEAR} {APP_NAME} - Performance Optimizations</p>
        <p style="font-size: 0.9em; color: #666666;">Powered by Groq API</p>
    </div>
""", unsafe_allow_html=True)
