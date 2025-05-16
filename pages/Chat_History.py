"""
Chat History page for SilentCodingLegend AI.
This page allows users to view, manage, and restore chat history across all documents
"""

import os
import streamlit as st
import pandas as pd
import base64
from datetime import datetime
from collections import defaultdict

# Import configuration and utilities
from src.config import (
    APP_NAME, APP_DESCRIPTION, CHAT_HISTORY_PATH
)
from src.utils import (
    apply_custom_style, get_chat_history_backups, load_chat_history_backup
)
from src.export_utils import (
    export_to_markdown, export_to_json, export_to_pdf, get_download_link, create_export_ui
)
from src.mobile_optimization import (
    get_device_type,
    create_adaptive_layout,
    create_adaptive_container,
    create_mobile_navigation
)

# Apply dark theme styling
apply_custom_style()

# Page header
st.title("💬 Chat History")
st.markdown("""
    <div style="padding: 10px; border-radius: 10px; background-color: #2d2d2d; margin-bottom: 20px; border-left: 4px solid #7e57c2;">
        <p style="color: #e0e0e0; margin: 0;">View and manage your past conversations with documents</p>
    </div>
""", unsafe_allow_html=True)

# Get all chat history backups
chat_backups = get_chat_history_backups()

# Main content area
if not chat_backups:
    st.markdown("""
        <div style="text-align: center; padding: 40px; background-color: #1e1e1e; border-radius: 10px; margin: 20px 0;">
            <h3 style="color: #7e57c2;">No Chat History Found</h3>
            <p style="color: #e0e0e0; margin-top: 15px;">
                You haven't had any conversations with your documents yet.
                Go to the Knowledge Base, upload a document, and start asking questions.
            </p>
            <p style="color: #b39ddb; margin-top: 20px;">
                <strong>Get started:</strong> Upload a document in the Knowledge Base page.
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Display chat history metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Conversations", len(chat_backups))
    
    # Count unique documents and models
    unique_docs = set(backup.get('document', 'Unknown') for backup in chat_backups)
    unique_models = set(backup.get('model', 'Unknown') for backup in chat_backups)
    
    with col2:
        st.metric("Documents", len(unique_docs))
    with col3:
        st.metric("Models Used", len(unique_models))
    
    # Group backups by document
    grouped_backups = defaultdict(list)
    for backup in chat_backups:
        doc = backup.get('document', 'Unknown document')
        grouped_backups[doc].append(backup)
    
    # Create tabs for All History and By Document
    tabs = st.tabs(["All History", "By Document", "Search", "Settings"])
    
    with tabs[0]:
        # All history sorted by date (newest first)
        st.markdown("### All Conversations")
        
        # Create a dataframe for all backups
        backup_data = [{
            'Document': b.get('document', 'Unknown'),
            'Date': b.get('date', 'Unknown'),
            'Time': b.get('time', 'Unknown'),
            'Model': b.get('model', 'Unknown'),
            'Session': b.get('session_id', 'Unknown'),
            'ID': idx  # Add index for reference
        } for idx, b in enumerate(chat_backups)]
        
        df = pd.DataFrame(backup_data)
        st.dataframe(df, use_container_width=True)
        
        # Select a conversation to view
        selected_idx = st.selectbox("Select conversation to view details:", 
                                  range(len(backup_data)),
                                  format_func=lambda x: f"{backup_data[x]['Document']} - {backup_data[x]['Date']} {backup_data[x]['Time']}")
        
        if selected_idx is not None:
            selected_backup = chat_backups[selected_idx]
            
            # Show conversation details
            st.markdown("### Conversation Details")
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                    <h4 style="color: #7e57c2; margin-top: 0;">Document: {selected_backup.get('document', 'Unknown')}</h4>
                    <p style="color: #e0e0e0;">
                        <strong>Date:</strong> {selected_backup.get('date', 'Unknown')}<br>
                        <strong>Time:</strong> {selected_backup.get('time', 'Unknown')}<br>
                        <strong>Model:</strong> {selected_backup.get('model', 'Unknown')}<br>
                        <strong>Session:</strong> {selected_backup.get('session_id', 'Unknown')}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show conversation content
            messages = load_chat_history_backup(selected_backup.get('file_path'))
            st.markdown("### Conversation Content")
            
            for message in messages:
                role = message.get("role", "unknown")
                content = message.get("content", "")
                
                with st.chat_message(role):
                    st.markdown(content)
            
            # Options for this conversation
            st.markdown("### Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                # Create export UI with document name in title
                export_title = f"Conversation with {selected_backup.get('document', 'Unknown')}"
                export_tab1, export_tab2, export_tab3 = st.tabs(["Markdown", "JSON", "PDF"])
                
                with export_tab1:
                    if st.button("Export as Markdown", key=f"export_md_{selected_idx}"):
                        md_content = export_to_markdown(messages, export_title)
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        filename = f"conversation_{selected_backup.get('document', 'doc').replace(' ', '_')}_{timestamp}.md"
                        st.markdown(get_download_link(md_content, filename, "text/markdown"), unsafe_allow_html=True)
                
                with export_tab2:
                    if st.button("Export as JSON", key=f"export_json_{selected_idx}"):
                        json_content = export_to_json(messages, export_title)
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        filename = f"conversation_{selected_backup.get('document', 'doc').replace(' ', '_')}_{timestamp}.json"
                        st.markdown(get_download_link(json_content, filename, "application/json"), unsafe_allow_html=True)
                
                with export_tab3:
                    if st.button("Export as PDF", key=f"export_pdf_{selected_idx}"):
                        pdf_content = export_to_pdf(messages, export_title)
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        filename = f"conversation_{selected_backup.get('document', 'doc').replace(' ', '_')}_{timestamp}.pdf"
                        st.markdown(get_download_link(pdf_content, filename, "application/pdf"), unsafe_allow_html=True)
            
            with col2:
                if st.button("Delete Conversation", key=f"delete_{selected_idx}"):
                    # Confirm deletion
                    if st.session_state.get("confirm_delete") != selected_idx:
                        st.session_state["confirm_delete"] = selected_idx
                        st.warning("Are you sure you want to delete this conversation? Click again to confirm.")
                    else:
                        try:
                            os.remove(selected_backup.get('file_path'))
                            st.session_state["confirm_delete"] = None
                            st.success("Conversation deleted successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting conversation: {str(e)}")
    
    with tabs[1]:
        # Organize by document
        st.markdown("### By Document")
        
        # Create document selection
        selected_doc = st.selectbox(
            "Select Document", 
            options=list(grouped_backups.keys()),
            format_func=lambda x: f"{x} ({len(grouped_backups[x])} conversations)"
        )
        
        if selected_doc:
            st.markdown(f"### Conversations for: {selected_doc}")
            
            # Create a dataframe for the selected document
            doc_data = [{
                'Date': b.get('date', 'Unknown'),
                'Time': b.get('time', 'Unknown'),
                'Model': b.get('model', 'Unknown'),
                'Session': b.get('session_id', 'Unknown'),
                'Path': b.get('file_path', '')
            } for b in grouped_backups[selected_doc]]
            
            df_doc = pd.DataFrame(doc_data)
            st.dataframe(df_doc, use_container_width=True)
            
            # Display conversations
            for i, backup in enumerate(grouped_backups[selected_doc]):
                with st.expander(f"{backup.get('date', 'Unknown')} at {backup.get('time', 'Unknown')} - Model: {backup.get('model', 'Unknown')}"):
                    messages = load_chat_history_backup(backup.get('file_path'))
                    
                    # Show conversation preview
                    user_msg = next((m for m in messages if m.get('role') == 'user'), None)
                    ai_msg = next((m for m in messages if m.get('role') == 'assistant'), None)
                    
                    if user_msg:
                        st.markdown("**User question:**")
                        st.markdown(f"```\n{user_msg.get('content')[:150]}{'...' if len(user_msg.get('content', '')) > 150 else ''}\n```")
                    
                    if ai_msg:
                        st.markdown("**AI response:**")
                        st.markdown(f"```\n{ai_msg.get('content')[:150]}{'...' if len(ai_msg.get('content', '')) > 150 else ''}\n```")
                    
                    # Buttons for this conversation
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("View Full Conversation", key=f"view_doc_{i}"):
                            for msg in messages:
                                role = msg.get("role", "unknown")
                                content = msg.get("content", "")
                                
                                with st.chat_message(role):
                                    st.markdown(content)
                    
                    with col2:
                        # Add export options for this individual conversation
                        export_type = st.selectbox("Export as", ["Markdown", "JSON", "PDF"], key=f"export_type_doc_{i}")
                        if st.button("Export", key=f"export_doc_{i}"):
                            export_title = f"Conversation with {selected_doc}"
                            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                            filename = f"conversation_{selected_doc.replace(' ', '_')}_{timestamp}"
                            
                            if export_type == "Markdown":
                                content = export_to_markdown(messages, export_title)
                                st.markdown(get_download_link(content, f"{filename}.md", "text/markdown"), unsafe_allow_html=True)
                            elif export_type == "JSON":
                                content = export_to_json(messages, export_title)
                                st.markdown(get_download_link(content, f"{filename}.json", "application/json"), unsafe_allow_html=True)
                            else:  # PDF
                                content = export_to_pdf(messages, export_title)
                                st.markdown(get_download_link(content, f"{filename}.pdf", "application/pdf"), unsafe_allow_html=True)
                    
                    with col3:
                        if st.button("Delete", key=f"delete_doc_{i}"):
                            try:
                                os.remove(backup.get('file_path'))
                                st.success("Conversation deleted successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting conversation: {str(e)}")
    
    with tabs[2]:
        # Search functionality
        st.markdown("### Search Conversations")
        
        search_query = st.text_input("Enter search term")
        search_field = st.radio("Search in:", ["Document name", "Content", "Date", "Model"], horizontal=True)
        
        if search_query:
            # Perform search based on the selected field
            search_results = []
            
            for idx, backup in enumerate(chat_backups):
                found = False
                
                if search_field == "Document name":
                    if search_query.lower() in backup.get('document', '').lower():
                        found = True
                
                elif search_field == "Date":
                    if search_query.lower() in backup.get('date', '').lower():
                        found = True
                
                elif search_field == "Model":
                    if search_query.lower() in backup.get('model', '').lower():
                        found = True
                
                elif search_field == "Content":
                    # Search in content requires loading the conversation
                    messages = load_chat_history_backup(backup.get('file_path'))
                    for msg in messages:
                        if search_query.lower() in msg.get('content', '').lower():
                            found = True
                            break
                
                if found:
                    search_results.append((idx, backup))
            
            # Display search results
            st.markdown(f"Found {len(search_results)} conversations matching '{search_query}'")
            
            for idx, backup in search_results:
                with st.expander(f"{backup.get('document', 'Unknown')} - {backup.get('date', 'Unknown')} at {backup.get('time', 'Unknown')}"):
                    st.markdown(f"""
                        <div style="background-color: #1e1e1e; padding: 10px; border-radius: 5px;">
                            <p style="color: #e0e0e0; margin: 0;"><strong>Document:</strong> {backup.get('document', 'Unknown')}</p>
                            <p style="color: #e0e0e0; margin: 0;"><strong>Date:</strong> {backup.get('date', 'Unknown')} at {backup.get('time', 'Unknown')}</p>
                            <p style="color: #e0e0e0; margin: 0;"><strong>Model:</strong> {backup.get('model', 'Unknown')}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("View Conversation", key=f"view_search_{idx}"):
                        messages = load_chat_history_backup(backup.get('file_path'))
                        for msg in messages:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            
                            with st.chat_message(role):
                                if search_field == "Content" and search_query.lower() in content.lower():
                                    # Highlight the search term
                                    highlighted = content.replace(
                                        search_query, 
                                        f"<span style='background-color: yellow; color: black;'>{search_query}</span>"
                                    )
                                    st.markdown(highlighted, unsafe_allow_html=True)
                                else:
                                    st.markdown(content)
    
    with tabs[3]:
        # Settings and management
        st.markdown("### Chat History Settings")
        
        # Directory information
        st.markdown(f"""
            <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: #7e57c2; margin-top: 0;">Storage Information</h4>
                <p style="color: #e0e0e0;">
                    <strong>Storage Directory:</strong> {CHAT_HISTORY_PATH}<br>
                    <strong>Total Conversations:</strong> {len(chat_backups)}<br>
                    <strong>Unique Documents:</strong> {len(unique_docs)}<br>
                    <strong>Storage Format:</strong> JSON files with metadata and messages
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Bulk actions
        st.markdown("### Bulk Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Add export format options for bulk export
            export_format = st.selectbox(
                "Select export format:", 
                ["Markdown", "JSON", "PDF"],
                key="bulk_export_format"
            )
            
            if st.button("Export All Conversations", key="export_all"):
                # Create a zip file containing all conversations
                import zipfile
                import io
                import uuid
                
                # Create a unique ID for this export
                export_id = str(uuid.uuid4())[:8]
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                zip_filename = f"all_conversations_{timestamp}_{export_id}.zip"
                
                # Create an in-memory zip file
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add each conversation to the zip
                    for idx, backup in enumerate(chat_backups):
                        try:
                            # Load the conversation
                            messages = load_chat_history_backup(backup.get('file_path'))
                            if not messages:
                                continue
                                
                            # Create the export title
                            doc_name = backup.get('document', 'Unknown').replace(' ', '_')
                            export_title = f"Conversation with {backup.get('document', 'Unknown')}"
                            date_str = backup.get('date', '').replace('-', '')
                            
                            # Generate filename
                            file_base = f"{doc_name}_{date_str}"
                            
                            # Export the conversation in the selected format
                            if export_format == "Markdown":
                                content = export_to_markdown(messages, export_title)
                                zipf.writestr(f"{file_base}.md", content)
                            elif export_format == "JSON":
                                content = export_to_json(messages, export_title)
                                zipf.writestr(f"{file_base}.json", content)
                            else:  # PDF
                                content = export_to_pdf(messages, export_title)
                                zipf.writestr(f"{file_base}.pdf", content)
                        except Exception as e:
                            st.error(f"Error exporting conversation {idx+1}: {str(e)}")
                            continue
                
                # Create a download link for the zip file
                zip_buffer.seek(0)
                b64 = base64.b64encode(zip_buffer.read()).decode()
                href = f'<a href="data:application/zip;base64,{b64}" download="{zip_filename}">Download All Conversations as {export_format}</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success(f"All conversations exported successfully as {export_format}!")
        
        with col2:
            if st.button("Delete All Conversations", key="delete_all"):
                # Confirm deletion
                if st.session_state.get("confirm_delete_all"):
                    try:
                        for backup in chat_backups:
                            os.remove(backup.get('file_path'))
                        st.session_state["confirm_delete_all"] = False
                        st.success("All conversations deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting conversations: {str(e)}")
                else:
                    st.session_state["confirm_delete_all"] = True
                    st.warning("Are you sure you want to delete ALL conversations? This cannot be undone. Click again to confirm.")
        
        # Filters for data retention
        st.markdown("### Data Retention")
        st.info("Set retention policies to automatically manage chat history storage. This feature will be implemented in a future update.")

# Footer
st.divider()
st.markdown(f"""
    <div style="text-align: center; padding: 10px; color: #999999; font-size: 0.8em; margin-top: 30px;">
        <p>© {datetime.now().year} {APP_NAME} - Chat History</p>
        <p style="font-size: 0.9em; color: #666666;">Manage and export your conversation history</p>
        <p style="margin-top: 10px;">
            <a href="/" target="_self" style="color: #7e57c2; text-decoration: none; font-size: 0.9em; margin-right: 20px;">
                Home 🏠
            </a>
            <a href="/VisionAI" target="_self" style="color: #7e57c2; text-decoration: none; font-size: 0.9em; margin-right: 20px;">
                VisionAI 👁️
            </a>
        </p>
    </div>
""", unsafe_allow_html=True)

# Add mobile navigation for small screens
device_type = get_device_type()
if device_type == "mobile":
    create_mobile_navigation()
