"""
Mobile Responsiveness demo page for SilentCodingLegend AI.
This page showcases the mobile-optimized features of the application.
"""

import streamlit as st
import os
import platform
from datetime import datetime

# Import configuration and utilities
from src.config import (
    APP_NAME, APP_DESCRIPTION
)
from src.utils import apply_custom_style
from src.mobile_optimization import (
    get_device_type, 
    adaptive_container,
    create_adaptive_layout,
    get_adaptive_column_ratio
)

# Apply dark theme styling with mobile optimization
apply_custom_style()

# Page header
st.title("📱 Mobile Optimization")
st.markdown("""
    <div style="padding: 10px; border-radius: 10px; background-color: #2d2d2d; margin-bottom: 20px; border-left: 4px solid #7e57c2;">
        <p style="color: #e0e0e0; margin: 0;">Experience SilentCodingLegend AI on any device with optimized layouts and responsive design</p>
    </div>
""", unsafe_allow_html=True)

# Device detection
device_type = get_device_type()
st.markdown(f"### Detected Device: **{device_type.title()}**")

# Get system info for demonstration
system_info = {
    "Operating System": platform.system() + " " + platform.release(),
    "Python Version": platform.python_version(),
    "Current Date": datetime.now().strftime("%Y-%m-%d"),
    "Current Time": datetime.now().strftime("%H:%M:%S"),
    "Device Type": device_type.title()
}

st.markdown("### System Information")
for key, value in system_info.items():
    st.markdown(f"**{key}:** {value}")

# Adaptive layouts demonstration
st.markdown("### Adaptive Layouts")
st.markdown("These layouts automatically adjust based on your device:")

# Example 1: Adaptive columns
st.subheader("Adaptive Columns")
cols = create_adaptive_layout(columns=3)

for i, col in enumerate(cols):
    with col:
        st.button(f"Button {i+1}", key=f"btn_{i}")
        st.markdown(f"Column {i+1}")

# Example 2: Adaptive column ratio
st.subheader("Adaptive Column Ratio")
left_size, right_size = get_adaptive_column_ratio()
left_col, right_col = st.columns([left_size, right_size])

with left_col:
    st.markdown("### Content Column")
    st.write("This column contains the main content and will expand to full width on mobile devices.")
    st.image("https://placehold.co/600x200/2d2d2d/b39ddb?text=Main+Content", use_container_width=True)

with right_col:
    if right_size > 0:  # Only show if column is visible (not mobile)
        st.markdown("### Sidebar Content")
        st.write("This column may be hidden on mobile devices to provide more space for the main content.")
        st.metric("Example Metric", "42%", "4.2% increase")

# Example 3: Adaptive containers
st.subheader("Adaptive Containers")
with adaptive_container(mobile_collapsed=True):
    st.write("This content is automatically collapsed into an expander on mobile devices to save space.")
    st.code("""
    # Example code for demonstration
    def hello_mobile():
        print("Hello, mobile user!")
        return "Optimized for your device"
    """)

# Touch-optimized controls demonstration
st.markdown("### Touch-Optimized Controls")
st.markdown("These controls are optimized for touch interfaces with larger tap targets:")

# Demo options
st.radio(
    "Select an option",
    ["Option 1", "Option 2", "Option 3"],
    horizontal=True
)

col1, col2 = st.columns(2)
with col1:
    st.button("Primary Action", type="primary")
with col2:
    st.button("Secondary Action")

# File uploader with full width
st.markdown("### Responsive File Uploader")
st.file_uploader("Upload a file", type=["jpg", "png", "pdf"])

# Text inputs optimized for mobile
st.markdown("### Mobile-Friendly Inputs")
st.text_input("Enter text", placeholder="Optimized text input")
st.slider("Adjust value", min_value=0, max_value=100, value=50)

# Responsive table example
st.markdown("### Responsive Data Display")
data = {
    "Name": ["Project A", "Project B", "Project C"],
    "Status": ["Complete", "In Progress", "Planned"],
    "Progress": [100, 65, 0]
}
st.dataframe(data, use_container_width=True)

# Additional mobile tips
with st.expander("Tips for Mobile Users"):
    st.markdown("""
    * Use landscape orientation for wider views
    * Tap the top-left menu to access the sidebar
    * Double-tap on code blocks to zoom
    * Long-press on images to save them
    * Use the bottom navigation for quick access
    """)

# Footer
st.divider()
st.markdown(f"""
    <div style="text-align: center; padding: 10px; color: #999999; font-size: 0.8em; margin-top: 20px;">
        <p>© 2025 {APP_NAME} - Mobile Optimized</p>
        <p style="font-size: 0.9em; color: #666666;">{APP_DESCRIPTION}</p>
    </div>
""", unsafe_allow_html=True)
