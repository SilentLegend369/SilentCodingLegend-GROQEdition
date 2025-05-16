"""
Export utilities for SilentCodingLegend AI.
Includes functions to export conversations in various formats.
"""

import json
import base64
import io
from typing import List, Dict, Optional
import streamlit as st
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def export_to_markdown(messages: List[Dict], title: Optional[str] = None) -> str:
    """
    Export conversation history to Markdown format.
    
    Args:
        messages: List of message dictionaries with role and content
        title: Optional title for the exported document
        
    Returns:
        Markdown content as a string
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = f"# {title or 'SilentCodingLegend AI Chat'}\n\n"
    md_content += f"*Exported on: {now}*\n\n"
    md_content += "---\n\n"
    
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        
        if role.lower() == "system":
            # Skip system messages in the export
            continue
            
        md_content += f"## {role}\n\n{content}\n\n"
        md_content += "---\n\n"
    
    return md_content

def export_to_json(messages: List[Dict], title: Optional[str] = None) -> str:
    """
    Export conversation history to JSON format.
    
    Args:
        messages: List of message dictionaries with role and content
        title: Optional title for the exported document
        
    Returns:
        JSON content as a string
    """
    export_data = {
        "title": title or "SilentCodingLegend AI Chat",
        "timestamp": datetime.now().isoformat(),
        "messages": [msg for msg in messages if msg["role"].lower() != "system"]
    }
    
    return json.dumps(export_data, indent=2)

def export_to_pdf(messages: List[Dict], title: Optional[str] = None) -> bytes:
    """
    Export conversation history to PDF format.
    
    Args:
        messages: List of message dictionaries with role and content
        title: Optional title for the exported document
        
    Returns:
        PDF content as bytes
    """
    # Create an in-memory PDF file
    buffer = io.BytesIO()
    
    # Set up the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=72)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["Normal"]
    
    # Create custom styles
    user_style = ParagraphStyle(
        'UserStyle',
        parent=normal_style,
        spaceAfter=12,
        spaceBefore=6,
        backColor=colors.lightgrey,
        borderPadding=5,
    )
    
    assistant_style = ParagraphStyle(
        'AssistantStyle',
        parent=normal_style,
        spaceAfter=12,
        spaceBefore=6,
        borderPadding=5,
    )
    
    # Build content
    elements = []
    
    # Add title
    elements.append(Paragraph(title or "SilentCodingLegend AI Chat", title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add timestamp
    timestamp = f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    elements.append(Paragraph(timestamp, normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Process messages
    for msg in messages:
        role = msg["role"].lower()
        content = msg["content"]
        
        if role == "system":
            # Skip system messages
            continue
        
        # Add heading for the role
        role_heading = role.capitalize()
        elements.append(Paragraph(role_heading, heading_style))
        
        # Add message content with appropriate style
        style = user_style if role == "user" else assistant_style
        elements.append(Paragraph(content.replace('\n', '<br/>'), style))
        elements.append(Spacer(1, 0.15*inch))
    
    # Build the PDF
    doc.build(elements)
    
    # Get the value from the buffer
    buffer.seek(0)
    return buffer.getvalue()

def get_download_link(content: str, filename: str, mime_type: str) -> str:
    """
    Create a download link for exporting content.
    
    Args:
        content: String or bytes content to download
        filename: Filename for the download
        mime_type: MIME type of the content
        
    Returns:
        HTML string with download link
    """
    if isinstance(content, str):
        content = content.encode()
        
    b64 = base64.b64encode(content).decode()
    dl_link = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" target="_blank">Download {filename}</a>'
    return dl_link

def create_export_ui(messages: List[Dict], title: Optional[str] = None):
    """
    Create a UI for exporting conversation in various formats.
    
    Args:
        messages: List of message dictionaries with role and content
        title: Optional title for the exported document
    """
    st.subheader("Export Conversation")
    
    # Title input
    export_title = st.text_input("Export Title", value=title or "SilentCodingLegend AI Chat")
    
    # Export format selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export as Markdown"):
            md_content = export_to_markdown(messages, export_title)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"chat_export_{timestamp}.md"
            
            # Display download link
            st.markdown(get_download_link(md_content, filename, "text/markdown"), unsafe_allow_html=True)
            
            # Preview
            with st.expander("Preview", expanded=False):
                st.markdown(md_content)
    
    with col2:
        if st.button("Export as JSON"):
            json_content = export_to_json(messages, export_title)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"chat_export_{timestamp}.json"
            
            # Display download link
            st.markdown(get_download_link(json_content, filename, "application/json"), unsafe_allow_html=True)
            
            # Preview
            with st.expander("Preview", expanded=False):
                st.json(json.loads(json_content))
    
    with col3:
        if st.button("Export as PDF"):
            pdf_content = export_to_pdf(messages, export_title)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"chat_export_{timestamp}.pdf"
            
            # Display download link
            st.markdown(get_download_link(pdf_content, filename, "application/pdf"), unsafe_allow_html=True)
            st.info("PDF ready for download. Click the link above.")
