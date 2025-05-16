"""
Mobile optimization utilities for SilentCodingLegend AI.
Provides responsive design utilities and mobile-specific enhancements.
"""

import streamlit as st
from typing import Dict, Any, Tuple, List, Optional, Union, Callable, ContextManager
import platform
import contextlib

def get_device_type() -> str:
    """
    Determine the likely device type based on platform and user agent.
    
    Returns:
        String indicating device type: "mobile", "tablet", or "desktop"
    """
    # Try to get user agent info if available in streamlit
    user_agent = ""
    try:
        if hasattr(st.runtime, "get_instance"):
            user_agent = st.runtime.get_instance()._get_user_info().get("User-Agent", "")
        elif hasattr(st.session_state, "user_agent"):
            user_agent = st.session_state.user_agent
    except:
        # Fall back to platform detection if user agent isn't available
        pass
    
    user_agent = user_agent.lower()
    
    # Check for mobile devices in user agent
    mobile_keywords = ['android', 'iphone', 'mobile', 'tablet']
    if any(keyword in user_agent for keyword in mobile_keywords):
        # Distinguish between tablets and phones (rough estimation)
        if 'ipad' in user_agent or 'tablet' in user_agent:
            return "tablet"
        return "mobile"
    
    # If no user agent info available, try with platform module
    system = platform.system().lower()
    if system in ['android', 'ios']:
        return "mobile"
    
    # Default to desktop
    return "desktop"

def is_mobile() -> bool:
    """
    Simple check if the current device is likely a mobile device.
    
    Returns:
        Boolean indicating if device is mobile
    """
    return get_device_type() == "mobile"

def is_tablet() -> bool:
    """
    Check if the current device is likely a tablet.
    
    Returns:
        Boolean indicating if device is a tablet
    """
    return get_device_type() == "tablet"

def get_viewport_meta_tag() -> str:
    """
    Generate a responsive viewport meta tag for HTML head.
    
    Returns:
        HTML meta tag string
    """
    return """
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    """

def add_swipe_detection() -> None:
    """
    Add JavaScript for detecting swipe gestures on mobile devices.
    This enables swipe-to-open sidebar and other swipe interactions.
    """
    swipe_js = """
    <script>
    // Simple swipe detection for mobile devices
    document.addEventListener('DOMContentLoaded', function() {
        let touchStartX = 0;
        let touchEndX = 0;
        let touchStartY = 0;
        let touchEndY = 0;
        
        // Min distance for a swipe to be recognized
        const minSwipeDistance = 50;
        
        // Track touch start position
        document.addEventListener('touchstart', function(e) {
            touchStartX = e.changedTouches[0].screenX;
            touchStartY = e.changedTouches[0].screenY;
        });
        
        // Track touch end position and handle swipes
        document.addEventListener('touchend', function(e) {
            touchEndX = e.changedTouches[0].screenX;
            touchEndY = e.changedTouches[0].screenY;
            handleSwipeGesture();
        });
        
        function handleSwipeGesture() {
            // Calculate horizontal and vertical distances
            const horizontalDistance = touchEndX - touchStartX;
            const verticalDistance = Math.abs(touchEndY - touchStartY);
            
            // Only trigger if horizontal > vertical (to avoid scroll confusion)
            if (Math.abs(horizontalDistance) > verticalDistance && Math.abs(horizontalDistance) > minSwipeDistance) {
                if (horizontalDistance > 0) {
                    // Right swipe - open sidebar
                    const sidebarButton = document.querySelector('[data-testid="collapsedControl"]');
                    if (sidebarButton) {
                        sidebarButton.click();
                    }
                } else {
                    // Left swipe - close sidebar
                    const expandedControl = document.querySelector('[data-testid="expanderContent"]');
                    if (expandedControl) {
                        const closeButton = document.querySelector('[data-testid="expandedControl"]');
                        if (closeButton) {
                            closeButton.click();
                        }
                    }
                }
            }
        }
    });
    </script>
    """
    
    st.markdown(swipe_js, unsafe_allow_html=True)

def improve_touch_targets() -> None:
    """
    Improve touch targets for better mobile experience.
    Adds specific CSS for better touch interaction.
    """
    st.markdown("""
    <style>
    /* Improved touch targets for mobile */
    @media (pointer: coarse) {
        /* Make radio buttons and checkboxes more tappable */
        .stRadio label, .stCheckbox label {
            padding: 10px 0 !important;
            margin: 5px 0 !important;
        }
        
        /* Improve selectbox touch area */
        .stSelectbox, [data-baseweb="select"] {
            min-height: 44px !important;
        }
        
        /* Better spacing for touch elements */
        .stButton, .stDownloadButton, .stFileUploader {
            margin: 10px 0 !important;
        }
        
        /* Larger touch targets for buttons */
        .stButton button, .stDownloadButton button {
            min-width: 120px !important;
            font-size: 1rem !important;
        }
        
        /* Better spacing between items */
        .stMarkdown, .stText {
            margin: 0.5rem 0 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def apply_responsive_styles() -> None:
    """
    Apply responsive styles for different device sizes.
    This should be called in addition to the main styling function.
    """
    # Add responsive CSS styles
    st.markdown("""
    <style>
    /* Responsive Base Styles */
    html, body, [data-testid="stAppViewContainer"] {
        max-width: 100vw !important;
        overflow-x: hidden !important;
    }
    
    /* Mobile styles (for screens under 768px) */
    @media (max-width: 768px) {
        /* Adjust header sizes */
        h1 {
            font-size: 1.5rem !important;
        }
        h2 {
            font-size: 1.3rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
        
        /* Reduce padding in the main container */
        [data-testid="stAppViewContainer"] > div {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        
        /* Adjust chat message appearance */
        .chat-message {
            padding: 0.8rem !important;
            margin-bottom: 0.6rem !important;
        }
        
        /* Make buttons more tappable */
        .stButton button {
            min-height: 44px !important;
            margin: 0.3rem 0 !important;
        }
        
        /* Improve sidebar usability on mobile */
        section[data-testid="stSidebar"] {
            width: 90vw !important;
            min-width: 90vw !important;
            padding: 1rem !important;
        }
        
        /* Adjust file uploader */
        [data-testid="stFileUploader"] {
            width: 100% !important;
        }
        
        /* Format inputs for touch */
        input, textarea, select {
            font-size: 16px !important; /* Prevents iOS zoom on focus */
        }
        
        /* Adjust column spacing */
        [data-testid="column"] {
            padding: 0.2rem !important;
        }
        
        /* Make tabs more tappable */
        button[role="tab"] {
            padding: 0.6rem 0.8rem !important;
        }
        
        /* Improve touch targets */
        .stCheckbox, .stRadio {
            min-height: 30px !important;
        }
        
        /* Adjust code blocks */
        pre {
            max-width: 100% !important;
            overflow-x: auto !important;
            font-size: 0.8rem !important;
        }
        
        /* Adjust expander components */
        details {
            padding: 0.5rem !important;
        }
        
        /* Minimize markdown padding */
        [data-testid="stMarkdown"] {
            padding-top: 0.2rem !important;
            padding-bottom: 0.2rem !important;
        }
        
        /* Improve mobile chat display */
        .chat-message .avatar {
            width: 32px !important;
            height: 32px !important;
            margin-right: 0.5rem !important;
        }
        
        /* Make dataframes scrollable on mobile */
        .stDataFrame {
            overflow-x: auto !important;
        }
        
        /* Format chat interface for mobile */
        [data-testid="stChatMessageContent"] {
            padding: 0.5rem !important;
        }
        
        /* Better image handling */
        img {
            max-width: 100% !important;
            height: auto !important;
        }
    }
    
    /* Tablet styles (between 768px and 1024px) */
    @media (min-width: 769px) and (max-width: 1024px) {
        /* Slightly adjusted styles for tablets */
        [data-testid="stAppViewContainer"] > div {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        h1 {
            font-size: 1.8rem !important;
        }
        
        /* Make sidebar appropriate for tablets */
        section[data-testid="stSidebar"] {
            width: 320px !important;
            min-width: 320px !important;
        }
    }
    
    /* Touch-optimized styling for both mobile and tablet */
    @media (pointer: coarse) {
        /* Larger touch targets */
        .stButton button, input, select, [role="tab"] {
            min-height: 44px !important;
        }
        
        /* More space between interactive elements */
        .stButton, .stCheckbox, .stRadio, .stSelectbox {
            margin: 0.5rem 0 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def optimize_for_mobile() -> None:
    """
    Apply all mobile optimizations including viewport settings and responsive styles.
    Call this function early in your app to ensure proper mobile display.
    """
    # Add viewport meta tag
    st.markdown(get_viewport_meta_tag(), unsafe_allow_html=True)
    
    # Apply responsive styles
    apply_responsive_styles()
    
    # Store device type in session state for reference
    st.session_state.device_type = get_device_type()
    
    # Apply specific layout adjustments based on device
    device_type = get_device_type()
    if device_type == "mobile":
        # Apply mobile-specific swipe detection script
        add_swipe_detection()
        
        # Improve touch targets for mobile devices
        improve_touch_targets()

def get_adaptive_column_ratio() -> Tuple:
    """
    Get appropriate column ratio based on device type.
    
    Returns:
        A tuple of column ratios to use with st.columns()
    """
    device_type = get_device_type()
    
    if device_type == "mobile":
        # Stack columns on mobile by making first column full width
        return (1, 0)
    elif device_type == "tablet":
        # Slightly adjust ratio for tablet
        return (3, 2)
    else:
        # Default desktop ratio
        return (2, 3)

def create_adaptive_layout(columns: int = 2) -> list:
    """
    Create an adaptive layout that adjusts based on device type.
    
    Args:
        columns: Number of columns for desktop (will be reduced for mobile)
    
    Returns:
        List of column objects
    """
    device_type = get_device_type()
    
    if device_type == "mobile":
        # Either return a single column or fewer columns for mobile
        return st.columns(1)
    elif device_type == "tablet" and columns > 2:
        # Reduce columns on tablet if original count was high
        return st.columns(min(columns, 2))
    else:
        # Use requested columns for desktop
        return st.columns(columns)

def create_adaptive_container(content_type: str = "default", mobile_collapsed: bool = True) -> ContextManager:
    """
    Create an adaptive container based on content type and device.
    
    Args:
        content_type: The type of content (options, settings, content)
        mobile_collapsed: Whether to default to collapsed state on mobile
    
    Returns:
        A container or expander based on device type
    """
    device_type = get_device_type()
    
    # Settings and options often benefit from being collapsible on mobile
    if device_type == "mobile" and mobile_collapsed and content_type in ["options", "settings"]:
        label = "Expand Settings" if content_type == "settings" else "Expand Options"
        return st.expander(label, expanded=False)
    # Main content should generally be expanded
    elif content_type == "content" or not mobile_collapsed:
        return st.container()
    else:
        return st.container()

def adaptive_image_display(image, max_mobile_width: int = 350) -> None:
    """
    Display an image with adaptive sizing based on device type.
    
    Args:
        image: The image to display (PIL Image or path)
        max_mobile_width: Maximum width for mobile devices
    """
    device_type = get_device_type()
    
    if device_type == "mobile":
        # Use a narrower display for mobile
        st.image(image, width=max_mobile_width, use_container_width=False)
    else:
        # For tablet and desktop, use responsive width
        st.image(image, use_container_width=True)

def create_mobile_navigation():
    """Create a mobile-friendly navigation footer that sticks to the bottom of the screen"""
    import streamlit as st
    from src.config import APP_NAME, CURRENT_YEAR
    
    st.markdown(f"""
        <style>
            .mobile-nav {{
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: #1e1e1e;
                padding: 10px 0;
                box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
                z-index: 1000;
                text-align: center;
            }}
            .mobile-nav a {{
                display: inline-block;
                margin: 0 10px;
                color: #7e57c2;
                text-decoration: none;
                font-size: 0.9em;
            }}
            .mobile-nav-icons {{
                margin-bottom: 10px;
            }}
            .mobile-nav-icons a {{
                font-size: 1.2em;
                margin: 0 15px;
            }}
            .groq-badge {{
                margin: 10px auto;
                display: block;
                height: 30px;
            }}
        </style>
        <div class="mobile-nav">
            <div class="mobile-nav-icons">
                <a href="/" target="_self">🏠</a>
                <a href="/VisionAI" target="_self">👁️</a>
                <a href="/Chat_History" target="_self">💬</a>
                <a href="/KnowledgeBase" target="_self">📚</a>
            </div>
            <a href="https://groq.com" target="_blank" rel="noopener noreferrer">
                <img 
                    class="groq-badge"
                    src="https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg" 
                    alt="Powered by Groq for fast inference."
                />
            </a>
            <p style="font-size: 0.8em; color: #777; margin: 5px 0;">© {CURRENT_YEAR} {APP_NAME}</p>
        </div>
    """, unsafe_allow_html=True)

def adaptive_container(mobile_collapsed: bool = True):
    """
    Create an adaptive container that's appropriately sized for the device.
    For mobile, this can be an expander or a regular container.
    
    Args:
        mobile_collapsed: Whether to default to collapsed state on mobile
    
    Returns:
        A container object
    """
    device_type = get_device_type()
    
    if device_type == "mobile" and mobile_collapsed:
        return st.expander("Expand for more options", expanded=False)
    else:
        return st.container()
