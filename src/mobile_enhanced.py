"""
Enhanced mobile optimization utilities for SilentCodingLegend AI application.
This wrapper module ensures proper mobile optimization across all pages.
"""

import streamlit as st
from src.mobile_optimization import (
    get_device_type, 
    optimize_for_mobile, 
    create_adaptive_layout,
    create_adaptive_container,
    create_mobile_navigation,
    adaptive_image_display,
    add_swipe_detection,
    improve_touch_targets
)

def apply_mobile_optimizations():
    """
    Applies all mobile optimizations in the correct order.
    Call this function at the beginning of each page to ensure proper mobile display.
    """
    # Get device type
    device_type = get_device_type()
    
    # Apply core optimizations
    optimize_for_mobile()
    
    # Store device type in session state for reference
    st.session_state.device_type = device_type
    
    # Return the device type for conditional UI adjustments
    return device_type

def create_mobile_friendly_container(title, content_type="settings", mobile_collapsed=True):
    """
    Create a mobile-friendly container with proper title formatting.
    
    Args:
        title: The title to display for the container
        content_type: The type of content (options, settings, content)
        mobile_collapsed: Whether to collapse on mobile
        
    Returns:
        A container appropriate for the current device
    """
    device_type = get_device_type()
    
    # Create the container
    container = create_adaptive_container(
        content_type=content_type, 
        mobile_collapsed=(device_type == "mobile" and mobile_collapsed)
    )
    
    # Return the container for use in a with statement
    return container

def get_mobile_layout_columns(count=2):
    """
    Get appropriate column layout based on device.
    
    Args:
        count: Desired number of columns for desktop
        
    Returns:
        List of column objects
    """
    device_type = get_device_type()
    return create_adaptive_layout(columns=count)

def display_mobile_footer(app_name, year, page_name=""):
    """
    Display a standardized mobile-friendly footer.
    
    Args:
        app_name: Application name
        year: Current year
        page_name: Current page name
    """
    # Standard footer
    st.divider()
    st.markdown(f"""
        <div style="text-align: center; padding: 10px; color: #999999; font-size: 0.8em; margin-top: 30px;">
            <p>© {year} {app_name}{' - ' + page_name if page_name else ''}</p>
            <p style="font-size: 0.9em; color: #666666;">Powered by Groq API</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add mobile navigation for small screens
    device_type = get_device_type()
    if device_type == "mobile":
        create_mobile_navigation()
