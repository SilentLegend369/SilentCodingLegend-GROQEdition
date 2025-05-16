"""
Advanced vision utilities for SilentCodingLegend AI.
Includes functions for image comparison, region selection, and annotation.
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import List, Tuple, Dict, Optional, Union
import cv2

def get_image_dimensions(image: Image.Image) -> Tuple[int, int]:
    """
    Get the dimensions of an image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (width, height)
    """
    return image.size

def extract_image_region(
    image: Image.Image, 
    region: str
) -> Image.Image:
    """
    Extract a specific region from an image.
    
    Args:
        image: PIL Image object
        region: Region identifier (e.g., "top-left", "center", "bottom-right")
        
    Returns:
        Cropped image containing only the specified region
    """
    width, height = image.size
    
    # Define the regions as fractions of the image
    regions = {
        "top-left": (0, 0, width//2, height//2),
        "top-center": (width//4, 0, 3*width//4, height//2),
        "top-right": (width//2, 0, width, height//2),
        "middle-left": (0, height//4, width//2, 3*height//4),
        "center": (width//4, height//4, 3*width//4, 3*height//4),
        "middle-right": (width//2, height//4, width, 3*height//4),
        "bottom-left": (0, height//2, width//2, height),
        "bottom-center": (width//4, height//2, 3*width//4, height),
        "bottom-right": (width//2, height//2, width, height),
    }
    
    if region not in regions:
        raise ValueError(f"Unknown region: {region}. Available regions: {list(regions.keys())}")
    
    return image.crop(regions[region])

def highlight_image_region(
    image: Image.Image, 
    region: str,
    color: str = "red",
    alpha: float = 0.3
) -> Image.Image:
    """
    Highlight a specific region in an image.
    
    Args:
        image: PIL Image object
        region: Region identifier (e.g., "top-left", "center", "bottom-right")
        color: Color to use for highlighting
        alpha: Transparency level (0=transparent, 1=opaque)
        
    Returns:
        Image with the specified region highlighted
    """
    width, height = image.size
    
    # Convert to RGBA if not already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create a copy to avoid modifying the original
    highlighted = image.copy()
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Define the regions as fractions of the image
    regions = {
        "top-left": (0, 0, width//2, height//2),
        "top-center": (width//4, 0, 3*width//4, height//2),
        "top-right": (width//2, 0, width, height//2),
        "middle-left": (0, height//4, width//2, 3*height//4),
        "center": (width//4, height//4, 3*width//4, 3*height//4),
        "middle-right": (width//2, height//4, width, 3*height//4),
        "bottom-left": (0, height//2, width//2, height),
        "bottom-center": (width//4, height//2, 3*width//4, height),
        "bottom-right": (width//2, height//2, width, height),
    }
    
    if region not in regions:
        raise ValueError(f"Unknown region: {region}. Available regions: {list(regions.keys())}")
    
    # Convert color string to RGB
    color_map = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    
    rgb_color = color_map.get(color.lower(), (255, 0, 0))  # Default to red if color not found
    
    # Draw the semi-transparent rectangle
    draw.rectangle(regions[region], fill=(*rgb_color, int(255 * alpha)))
    
    # Combine the original image with the overlay
    return Image.alpha_composite(highlighted, overlay)

def compare_images(
    image1: Image.Image, 
    image2: Image.Image,
    method: str = "side_by_side"
) -> Image.Image:
    """
    Compare two images using different methods.
    
    Args:
        image1: First PIL Image object
        image2: Second PIL Image object
        method: Comparison method ("side_by_side", "blend", "difference")
        
    Returns:
        Combined image showing the comparison
    """
    # Ensure both images are the same size
    # Resize the second image to match the first if needed
    if image1.size != image2.size:
        image2 = image2.resize(image1.size, Image.LANCZOS)
    
    width, height = image1.size
    
    if method == "side_by_side":
        # Create a new image with double width
        combined = Image.new('RGB', (width * 2, height))
        combined.paste(image1, (0, 0))
        combined.paste(image2, (width, 0))
        
    elif method == "blend":
        # Convert to RGBA if not already
        if image1.mode != 'RGBA':
            image1 = image1.convert('RGBA')
        if image2.mode != 'RGBA':
            image2 = image2.convert('RGBA')
            
        # Create a blended image (50/50)
        combined = Image.blend(image1, image2, 0.5)
        
    elif method == "difference":
        # Convert to numpy arrays
        arr1 = np.array(image1.convert('RGB'))
        arr2 = np.array(image2.convert('RGB'))
        
        # Calculate absolute difference
        diff = cv2.absdiff(arr1, arr2)
        
        # Enhance the difference for visibility
        diff = cv2.convertScaleAbs(diff, alpha=2.0)
        
        # Convert back to PIL
        combined = Image.fromarray(diff)
        
    else:
        raise ValueError(f"Unknown comparison method: {method}")
    
    return combined

def annotate_image(
    image: Image.Image,
    annotations: List[Dict[str, Union[str, Tuple[int, int, int, int]]]]
) -> Image.Image:
    """
    Add text and shape annotations to an image.
    
    Args:
        image: PIL Image object
        annotations: List of annotation dictionaries, each with:
            - type: "text", "rectangle", "circle", "arrow"
            - content: Text content (for text type)
            - position: (x, y) coordinates or (x1, y1, x2, y2) for shapes
            - color: Color string (optional)
            - size: Font or line size (optional)
        
    Returns:
        Annotated image
    """
    # Create a copy to avoid modifying the original
    annotated = image.copy()
    
    # Convert to RGB if not already (some operations don't work on RGBA)
    if annotated.mode != 'RGB':
        annotated = annotated.convert('RGB')
        
    draw = ImageDraw.Draw(annotated)
    
    # Color mapping
    color_map = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    
    # Process each annotation
    for annotation in annotations:
        annotation_type = annotation.get("type", "")
        color_name = annotation.get("color", "red").lower()
        color = color_map.get(color_name, (255, 0, 0))  # Default to red
        
        if annotation_type == "text":
            text = annotation.get("content", "")
            position = annotation.get("position", (10, 10))
            font_size = annotation.get("size", 20)
            
            # Use default font
            try:
                # Try to use a system font if available
                font = ImageFont.truetype("Arial", font_size)
            except IOError:
                # Fall back to default font
                font = ImageFont.load_default()
                
            # Add text with black outline for visibility
            x, y = position
            # Draw text outline
            for offset_x, offset_y in [(0,1), (1,0), (0,-1), (-1,0)]:
                draw.text((x+offset_x, y+offset_y), text, fill=(0, 0, 0), font=font)
            # Draw main text
            draw.text(position, text, fill=color, font=font)
            
        elif annotation_type == "rectangle":
            position = annotation.get("position", (10, 10, 100, 100))
            width = annotation.get("size", 2)
            draw.rectangle(position, outline=color, width=width)
            
        elif annotation_type == "circle":
            position = annotation.get("position", (100, 100, 150, 150))  # x, y, x+r, y+r
            width = annotation.get("size", 2)
            draw.ellipse(position, outline=color, width=width)
            
        elif annotation_type == "arrow":
            # Simple arrow implementation
            start = annotation.get("start", (10, 10))
            end = annotation.get("end", (100, 100))
            width = annotation.get("size", 2)
            
            # Draw the line
            draw.line([start, end], fill=color, width=width)
            
            # Draw arrowhead
            # This is a simple triangle approximation
            arrow_size = width * 3
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = (dx**2 + dy**2)**0.5
            udx, udy = dx / length, dy / length  # Unit vector
            
            # Perpendicular unit vector
            px, py = -udy, udx
            
            # Calculate the three points of the arrowhead
            arrow_point1 = (
                int(end[0] - arrow_size * udx + arrow_size * px / 2),
                int(end[1] - arrow_size * udy + arrow_size * py / 2)
            )
            arrow_point2 = (
                int(end[0] - arrow_size * udx - arrow_size * px / 2),
                int(end[1] - arrow_size * udy - arrow_size * py / 2)
            )
            
            # Draw the arrowhead
            draw.polygon([end, arrow_point1, arrow_point2], fill=color)
    
    return annotated

def create_selectable_image_regions(image: Image.Image) -> Dict:
    """
    Create a UI for selecting regions of an image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with selected region and highlighted image
    """
    width, height = image.size
    
    # Create a grid layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Select Region")
        regions = [
            "top-left", "top-center", "top-right",
            "middle-left", "center", "middle-right",
            "bottom-left", "bottom-center", "bottom-right"
        ]
        
        selected_region = st.selectbox(
            "Choose a region to analyze",
            regions,
            index=4  # Default to center
        )
        
        highlight_color = st.selectbox(
            "Highlight color",
            ["red", "green", "blue", "yellow", "cyan", "magenta"],
            index=0
        )
        
        transparency = st.slider(
            "Highlight transparency",
            min_value=0.1,
            max_value=0.7,
            value=0.3,
            step=0.1
        )
        
        # Action buttons
        extract = st.button("Extract Region")
        highlight = st.button("Highlight Region")
    
    with col2:
        st.markdown("### Preview")
        if extract:
            # Extract the selected region
            region_image = extract_image_region(image, selected_region)
            st.image(region_image, caption=f"Extracted {selected_region} region", use_container_width=True)
            return {
                "action": "extract",
                "region": selected_region,
                "image": region_image
            }
        elif highlight:
            # Highlight the selected region
            highlighted_image = highlight_image_region(
                image, 
                selected_region,
                color=highlight_color,
                alpha=transparency
            )
            st.image(highlighted_image, caption=f"Highlighted {selected_region} region", use_container_width=True)
            return {
                "action": "highlight",
                "region": selected_region,
                "image": highlighted_image
            }
        else:
            # Show original image with grid overlay
            overlay_image = image.copy()
            if overlay_image.mode != 'RGB':
                overlay_image = overlay_image.convert('RGB')
            
            draw = ImageDraw.Draw(overlay_image)
            
            # Draw grid lines
            grid_color = (255, 255, 255, 128)  # Semi-transparent white
            
            # Vertical lines
            draw.line([(width//2, 0), (width//2, height)], fill=grid_color, width=1)
            draw.line([(width//4, 0), (width//4, height)], fill=grid_color, width=1)
            draw.line([(3*width//4, 0), (3*width//4, height)], fill=grid_color, width=1)
            
            # Horizontal lines
            draw.line([(0, height//2), (width, height//2)], fill=grid_color, width=1)
            draw.line([(0, height//4), (width, height//4)], fill=grid_color, width=1)
            draw.line([(0, 3*height//4), (width, 3*height//4)], fill=grid_color, width=1)
            
            st.image(overlay_image, caption="Image with region grid", use_container_width=True)
            return {
                "action": "none",
                "region": selected_region
            }

def create_image_comparison_ui(image1: Optional[Image.Image] = None, image2: Optional[Image.Image] = None) -> Dict:
    """
    Create a UI for comparing two images.
    
    Args:
        image1: First PIL Image object (optional)
        image2: Second PIL Image object (optional)
        
    Returns:
        Dictionary with comparison method and result image
    """
    st.markdown("### Image Comparison")
    
    # Create upload areas if images not provided
    if image1 is None:
        uploaded_file1 = st.file_uploader(
            "Upload first image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="compare_image1"
        )
        if uploaded_file1:
            image1 = Image.open(uploaded_file1)
        
    if image2 is None:
        uploaded_file2 = st.file_uploader(
            "Upload second image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="compare_image2"
        )
        if uploaded_file2:
            image2 = Image.open(uploaded_file2)
    
    # Show original images if available
    if image1 is not None and image2 is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Image 1")
            st.image(image1, use_container_width=True)
        with col2:
            st.markdown("#### Image 2")
            st.image(image2, use_container_width=True)
        
        # Comparison settings
        st.markdown("#### Comparison Settings")
        comparison_method = st.radio(
            "Select comparison method",
            ["side_by_side", "blend", "difference"],
            horizontal=True
        )
        
        # Apply comparison
        if st.button("Compare Images"):
            result_image = compare_images(image1, image2, method=comparison_method)
            st.markdown("#### Comparison Result")
            st.image(result_image, caption=f"{comparison_method.replace('_', ' ').title()} Comparison", use_container_width=True)
            return {
                "method": comparison_method,
                "result": result_image
            }
    
    return {
        "method": None,
        "result": None
    }

def create_image_annotation_ui(image: Optional[Image.Image] = None) -> Dict:
    """
    Create a UI for annotating images.
    
    Args:
        image: PIL Image object (optional)
        
    Returns:
        Dictionary with annotations and annotated image
    """
    st.markdown("### Image Annotation")
    
    # Create upload area if image not provided
    if image is None:
        uploaded_file = st.file_uploader(
            "Upload image to annotate",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="annotate_image"
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
    
    if image is not None:
        # Display original image
        st.markdown("#### Original Image")
        st.image(image, use_container_width=True)
        
        # Annotation tools
        st.markdown("#### Add Annotations")
        
        # Initialize or get annotations list from session state
        if "annotations" not in st.session_state:
            st.session_state.annotations = []
        
        # UI for adding a new annotation
        annotation_type = st.selectbox(
            "Annotation type",
            ["text", "rectangle", "circle", "arrow"],
            key="annotation_type"
        )
        
        # Different inputs based on annotation type
        if annotation_type == "text":
            text_content = st.text_input("Text content", key="text_content")
            col1, col2 = st.columns(2)
            with col1:
                x_pos = st.number_input("X position", value=10, key="text_x_pos")
            with col2:
                y_pos = st.number_input("Y position", value=10, key="text_y_pos")
            
            text_color = st.selectbox(
                "Text color",
                ["red", "green", "blue", "yellow", "cyan", "magenta", "white", "black"],
                key="text_color"
            )
            
            text_size = st.slider("Text size", min_value=10, max_value=50, value=20, key="text_size")
            
            if st.button("Add Text Annotation"):
                st.session_state.annotations.append({
                    "type": "text",
                    "content": text_content,
                    "position": (int(x_pos), int(y_pos)),
                    "color": text_color,
                    "size": text_size
                })
        
        elif annotation_type == "rectangle":
            col1, col2 = st.columns(2)
            with col1:
                x1 = st.number_input("X1 (left)", value=10, key="rect_x1")
                y1 = st.number_input("Y1 (top)", value=10, key="rect_y1")
            with col2:
                x2 = st.number_input("X2 (right)", value=100, key="rect_x2")
                y2 = st.number_input("Y2 (bottom)", value=100, key="rect_y2")
            
            rect_color = st.selectbox(
                "Rectangle color",
                ["red", "green", "blue", "yellow", "cyan", "magenta", "white", "black"],
                key="rect_color"
            )
            
            rect_width = st.slider("Line width", min_value=1, max_value=10, value=2, key="rect_width")
            
            if st.button("Add Rectangle Annotation"):
                st.session_state.annotations.append({
                    "type": "rectangle",
                    "position": (int(x1), int(y1), int(x2), int(y2)),
                    "color": rect_color,
                    "size": rect_width
                })
        
        elif annotation_type == "circle":
            col1, col2 = st.columns(2)
            with col1:
                cx = st.number_input("Center X", value=100, key="circle_x")
                cy = st.number_input("Center Y", value=100, key="circle_y")
            with col2:
                radius = st.number_input("Radius", value=50, key="circle_radius")
            
            circle_color = st.selectbox(
                "Circle color",
                ["red", "green", "blue", "yellow", "cyan", "magenta", "white", "black"],
                key="circle_color"
            )
            
            circle_width = st.slider("Line width", min_value=1, max_value=10, value=2, key="circle_width")
            
            if st.button("Add Circle Annotation"):
                x1, y1 = int(cx - radius), int(cy - radius)
                x2, y2 = int(cx + radius), int(cy + radius)
                st.session_state.annotations.append({
                    "type": "circle",
                    "position": (x1, y1, x2, y2),
                    "color": circle_color,
                    "size": circle_width
                })
        
        elif annotation_type == "arrow":
            col1, col2 = st.columns(2)
            with col1:
                start_x = st.number_input("Start X", value=10, key="arrow_x1")
                start_y = st.number_input("Start Y", value=10, key="arrow_y1")
            with col2:
                end_x = st.number_input("End X", value=100, key="arrow_x2")
                end_y = st.number_input("End Y", value=100, key="arrow_y2")
            
            arrow_color = st.selectbox(
                "Arrow color",
                ["red", "green", "blue", "yellow", "cyan", "magenta", "white", "black"],
                key="arrow_color"
            )
            
            arrow_width = st.slider("Line width", min_value=1, max_value=10, value=2, key="arrow_width")
            
            if st.button("Add Arrow Annotation"):
                st.session_state.annotations.append({
                    "type": "arrow",
                    "start": (int(start_x), int(start_y)),
                    "end": (int(end_x), int(end_y)),
                    "color": arrow_color,
                    "size": arrow_width
                })
        
        # Display current annotations
        if st.session_state.annotations:
            st.markdown(f"#### Current Annotations ({len(st.session_state.annotations)})")
            for i, anno in enumerate(st.session_state.annotations):
                st.text(f"{i+1}. {anno['type']} - {anno.get('content', '')}")
            
            if st.button("Clear All Annotations"):
                st.session_state.annotations = []
            
            # Apply annotations button
            if st.button("Apply Annotations"):
                annotated_image = annotate_image(image, st.session_state.annotations)
                st.markdown("#### Annotated Image")
                st.image(annotated_image, caption="Annotated Image", use_container_width=True)
                return {
                    "annotations": st.session_state.annotations,
                    "image": annotated_image
                }
    
    return {
        "annotations": st.session_state.get("annotations", []),
        "image": None
    }
