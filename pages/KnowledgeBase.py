"""
Knowledge Base page for SilentCodingLegend AI.
This page allows users to upload documents and query them using the Groq API.
"""

import os
import uuid
import time
import streamlit as st
import pandas as pd
from datetime import datetime

# Import configuration and utilities
from src.config import (
    APP_NAME, APP_DESCRIPTION, ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB,
    UPLOAD_PATH, KB_MODEL_CONFIG, MODEL_INFO
)
from src.utils import (
    apply_custom_style, save_uploaded_file, is_valid_file_extension, query_groq_model, 
    backup_chat_history, get_chat_history_backups, load_chat_history_backup
)
from src.document_processor import process_document, format_chunks_with_metadata

# Apply dark theme styling
apply_custom_style()

# Initialize session state for knowledge base
if "kb_documents" not in st.session_state:
    st.session_state.kb_documents = []
if "kb_chunks" not in st.session_state:
    st.session_state.kb_chunks = []
if "kb_messages" not in st.session_state:
    st.session_state.kb_messages = []
if "kb_selected_doc" not in st.session_state:
    st.session_state.kb_selected_doc = None

# Create necessary directories
from src.config import CHAT_HISTORY_PATH
os.makedirs(UPLOAD_PATH, exist_ok=True)
os.makedirs(CHAT_HISTORY_PATH, exist_ok=True)

# Page header
st.title("📚 Knowledge Base")
st.markdown("""
    <div style="padding: 10px; border-radius: 10px; background-color: #2d2d2d; margin-bottom: 20px; border-left: 4px solid #7e57c2;">
        <p style="color: #e0e0e0; margin: 0;">Upload documents and ask questions based on your content</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for document management
with st.sidebar:
    st.markdown("""
        <h2 style="color: #7e57c2; margin-bottom: 20px; text-align: center;">Document Manager</h2>
    """, unsafe_allow_html=True)
    
    # File uploader
    st.markdown('<p style="color: #b39ddb; font-weight: bold;">Upload Document:</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=ALLOWED_EXTENSIONS,
        help=f"Upload files (max {MAX_FILE_SIZE_MB}MB) to query with the AI. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File size exceeds the {MAX_FILE_SIZE_MB}MB limit.")
        else:
            # Show document details
            st.markdown(f"""
                <div style="background-color: #2d2d2d; padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <p style="color: #e0e0e0; font-size: 0.9em; margin: 0;"><strong>Selected File:</strong></p>
                    <p style="color: #b39ddb; font-size: 0.85em; margin: 5px 0 0 0;">• {uploaded_file.name}</p>
                    <p style="color: #b39ddb; font-size: 0.85em; margin: 2px 0 0 0;">• {uploaded_file.size / 1024:.1f} KB</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Process button
            if st.button("Process Document", type="primary"):
                if not is_valid_file_extension(uploaded_file.name):
                    st.error(f"Invalid file type. Please upload one of these formats: {', '.join(ALLOWED_EXTENSIONS)}")
                else:
                    with st.spinner("Processing document..."):
                        # Save the uploaded file
                        file_path = save_uploaded_file(uploaded_file)
                        file_id = str(uuid.uuid4())
                        
                        # Process the document
                        try:
                            chunks = process_document(file_path)
                            formatted_chunks = format_chunks_with_metadata(chunks, uploaded_file.name)
                            
                            # Add document to session state
                            doc_info = {
                                "id": file_id,
                                "name": uploaded_file.name,
                                "path": file_path,
                                "size": uploaded_file.size,
                                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "chunks": len(chunks)
                            }
                            
                            st.session_state.kb_documents.append(doc_info)
                            st.session_state.kb_chunks.extend(formatted_chunks)
                            st.success(f"Document processed: {len(chunks)} chunks extracted")
                            
                            # Set as selected document
                            st.session_state.kb_selected_doc = doc_info
                            
                            # Force a rerun to update the UI
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error processing document: {str(e)}")
    
    # Document selection
    if st.session_state.kb_documents:
        st.markdown('<p style="color: #b39ddb; font-weight: bold; margin-top: 20px;">Your Documents:</p>', unsafe_allow_html=True)
        
        for i, doc in enumerate(st.session_state.kb_documents):
            doc_name = doc.get("name", "Unknown")
            doc_chunks = doc.get("chunks", 0)
            doc_time = doc.get("upload_time", "Unknown")
            
            # Create a card-like display for each document
            st.markdown(f"""
                <div style="background-color: #2d2d2d; padding: 10px; border-radius: 5px; margin-bottom: 10px; 
                           border-left: 3px solid {'#7e57c2' if st.session_state.kb_selected_doc and doc.get('id') == st.session_state.kb_selected_doc.get('id') else '#555'};">
                    <p style="color: #e0e0e0; font-weight: bold; font-size: 0.9em; margin: 0;">{doc_name}</p>
                    <p style="color: #b39ddb; font-size: 0.75em; margin: 2px 0;">{doc_chunks} chunks • {doc_time}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Create columns for document actions
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(f"Select", key=f"select_{doc['id']}"):
                    st.session_state.kb_selected_doc = doc
                    st.rerun()
            with col2:
                if st.button(f"Delete", key=f"delete_{doc['id']}"):
                    # Remove the document chunks
                    st.session_state.kb_chunks = [
                        c for c in st.session_state.kb_chunks 
                        if c["metadata"]["source"] != doc_name
                    ]
                    # Remove the document from the list
                    st.session_state.kb_documents.remove(doc)
                    if st.session_state.kb_selected_doc and st.session_state.kb_selected_doc.get("id") == doc.get("id"):
                        st.session_state.kb_selected_doc = None
                    st.rerun()

# Main area: Show document content or chat interface
if not st.session_state.kb_documents:
    # Show instructions when no documents are uploaded
    st.markdown("""
        <div style="text-align: center; padding: 40px; background-color: #1e1e1e; border-radius: 10px; margin: 20px 0;">
            <h3 style="color: #7e57c2;">Welcome to the Knowledge Base!</h3>
            <p style="color: #e0e0e0; margin-top: 15px;">
                Upload documents to your knowledge base and ask questions about them.
                SilentCodingLegend will analyze your documents and provide answers based on their content.
            </p>
            <p style="color: #b39ddb; margin-top: 20px;">
                <strong>Get started:</strong> Upload a document using the sidebar.
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Show document details
    selected_doc = st.session_state.kb_selected_doc
    
    if selected_doc:
        # Show chat interface with tabs for document view, chat, and chat history
        tabs = st.tabs(["Document Content", "Chat", "Chat History"])
        
        with tabs[0]:
            # Show document information in a card-like container
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="color: #7e57c2; margin-top: 0;">{selected_doc['name']}</h3>
                    <p style="color: #e0e0e0;">
                        <strong>Upload time:</strong> {selected_doc['upload_time']}<br>
                        <strong>Size:</strong> {selected_doc['size'] / 1024:.1f} KB<br>
                        <strong>Chunks:</strong> {selected_doc['chunks']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show document chunks
            st.markdown('<p style="color: #b39ddb; font-weight: bold;">Document Chunks:</p>', unsafe_allow_html=True)
            
            doc_chunks = [
                c for c in st.session_state.kb_chunks 
                if c["metadata"]["source"] == selected_doc['name']
            ]
            
            for i, chunk in enumerate(doc_chunks):
                with st.expander(f"Chunk {i+1}/{len(doc_chunks)}"):
                    st.text(chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"])
        
        with tabs[1]:
            # Chat interface for document queries
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="color: #7e57c2; margin-top: 0;">Ask about {selected_doc['name']}</h3>
                    <p style="color: #e0e0e0;">
                        Ask questions about the content of this document. The AI will search through the document to find relevant information.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display chat history
            for message in st.session_state.kb_messages:
                role = message["role"]
                content = message["content"]
                
                with st.chat_message(role):
                    st.markdown(content)
            
            # User input
            user_query = st.chat_input("Ask a question about the document...")
            
            if user_query:
                # Add user message to chat
                st.session_state.kb_messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)
                
                # Generate response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Display a spinner while processing
                    with st.spinner("⚡ Searching document and generating response..."):
                        try:
                            # Find relevant chunks for the query
                            # In a real application, this would use embeddings and similarity search
                            # For simplicity, we'll just use the query to filter chunks containing keywords
                            doc_chunks = [
                                c for c in st.session_state.kb_chunks 
                                if c["metadata"]["source"] == selected_doc['name']
                            ]
                            
                            # Simple keyword matching (this would be replaced with proper semantic search)
                            query_terms = user_query.lower().split()
                            matched_chunks = []
                            
                            for chunk in doc_chunks:
                                chunk_text = chunk["text"].lower()
                                match_score = sum(1 for term in query_terms if term in chunk_text)
                                if match_score > 0:
                                    matched_chunks.append((chunk, match_score))
                            
                            # Sort chunks by match score and take the top matches
                            matched_chunks.sort(key=lambda x: x[1], reverse=True)
                            top_chunks = matched_chunks[:min(KB_MODEL_CONFIG["similarity_top_k"], len(matched_chunks))]
                            
                            if top_chunks:
                                # Create a context string from the top chunks
                                context = "\n\n---\n\n".join([chunk[0]["text"] for chunk in top_chunks])
                                
                                # Prepare messages for the API call
                                messages = [
                                    {"role": "system", "content": f"You are SilentCodingLegend, an expert AI assistant specialized in answering questions based on document content. Answer only based on the provided document context. If the answer isn't in the context, say that you don't have enough information."},
                                    {"role": "user", "content": f"Context from document '{selected_doc['name']}':\n\n{context}\n\nQuestion: {user_query}"}
                                ]
                                
                                # Call Groq API
                                model_id = KB_MODEL_CONFIG["default_retrieval_model"]
                                response = query_groq_model(
                                    model_id=model_id,
                                    messages=messages,
                                    temperature=0.5,
                                    max_tokens=1500,
                                    stream=True,
                                    json_mode=False
                                )
                                
                                # Process the streaming response
                                for chunk in response:
                                    if chunk.choices[0].delta.content:
                                        full_response += chunk.choices[0].delta.content
                                        message_placeholder.markdown(full_response + "▌")
                                        time.sleep(0.01)
                                        
                                # Update the placeholder with the complete response
                                message_placeholder.markdown(full_response)
                            else:
                                full_response = "I couldn't find any relevant information in the document to answer your question. Please try rephrasing your question or ask about a different topic covered in the document."
                                message_placeholder.markdown(full_response)
                                
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            full_response = f"I apologize, but I encountered an error while processing your question: {str(e)}"
                            message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.kb_messages.append({"role": "assistant", "content": full_response})
                
                # Backup chat history with document name, date, time, session and model info
                try:
                    backup_path = backup_chat_history(
                        messages=st.session_state.kb_messages,
                        document_name=selected_doc['name'],
                        model_id=model_id
                    )
                    # Show a small notification that the backup was created (using st.toast)
                    st.toast(f"Chat history backed up", icon="✅")
                except Exception as e:
                    st.toast(f"Error backing up chat: {str(e)}", icon="⚠️")
        
        with tabs[2]:
            # Chat history tab for viewing past conversations
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="color: #7e57c2; margin-top: 0;">Chat History</h3>
                    <p style="color: #e0e0e0;">
                        View and load previously saved conversations related to your documents.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Debug information in an expandable section
            with st.expander("Debug Information", expanded=False):
                from src.config import CHAT_HISTORY_PATH
                st.info(f"Looking for chat history in: {CHAT_HISTORY_PATH}")
                
                # Check if directory exists and list files
                if not os.path.exists(CHAT_HISTORY_PATH):
                    st.error(f"Chat history directory does not exist!")
                    os.makedirs(CHAT_HISTORY_PATH, exist_ok=True)
                    st.success("Created chat history directory")
                
                history_files = os.listdir(CHAT_HISTORY_PATH) if os.path.exists(CHAT_HISTORY_PATH) else []
                st.info(f"Found {len(history_files)} files in chat history directory:")
                st.write(history_files)
                
                # Add reset functionality
                if st.button("Reset Knowledge Base State"):
                    st.session_state.kb_messages = []
                    st.success("Chat history reset! Please reload the page.")
                    st.rerun()
                
                # Get all chat history backups
                chat_backups = get_chat_history_backups()
                st.info(f"Found {len(chat_backups)} chat backup records")
            
            # Filter backups for the current document if any
            doc_name = selected_doc['name']
            doc_backups = [b for b in chat_backups if b.get('document') == doc_name]
            
            if doc_backups:
                st.markdown(f'<p style="color: #b39ddb; font-weight: bold;">Saved conversations for "{doc_name}":</p>', unsafe_allow_html=True)
                
                for backup in doc_backups:
                    # Format the date and time nicely
                    backup_date = backup.get('date', 'Unknown date')
                    backup_time = backup.get('time', 'Unknown time')
                    backup_model = backup.get('model', 'Unknown model')
                    session_id = backup.get('session_id', 'Unknown session')
                    
                    # Show the backup entry with a collapsible section
                    with st.expander(f"{backup_date} at {backup_time} - Model: {backup_model}"):
                        # Button to load this conversation
                        if st.button("Load Conversation", key=f"load_{backup.get('timestamp', 'unknown')}"):
                            messages = load_chat_history_backup(backup.get('file_path'))
                            if messages:
                                st.session_state.kb_messages = messages
                                st.success("Conversation loaded successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to load conversation.")
                        
                        # Show conversation preview
                        st.markdown("#### Conversation Preview")
                        messages = load_chat_history_backup(backup.get('file_path'))
                        
                        # Show first user question and first AI response
                        user_msg = next((m for m in messages if m.get('role') == 'user'), None)
                        ai_msg = next((m for m in messages if m.get('role') == 'assistant'), None)
                        
                        if user_msg:
                            st.markdown("**User question:**")
                            st.markdown(f"```\n{user_msg.get('content')[:150]}{'...' if len(user_msg.get('content', '')) > 150 else ''}\n```")
                        
                        if ai_msg:
                            st.markdown("**AI response:**")
                            st.markdown(f"```\n{ai_msg.get('content')[:150]}{'...' if len(ai_msg.get('content', '')) > 150 else ''}\n```")
                        
                        # Option to delete this backup
                        if st.button("Delete Backup", key=f"delete_{backup.get('timestamp', 'unknown')}"):
                            try:
                                os.remove(backup.get('file_path'))
                                st.success("Backup deleted successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting backup: {str(e)}")
            else:
                st.info(f"No saved conversations found for document: {doc_name}")
                
            # Show all backups section
            st.markdown("---")
            with st.expander("View All Saved Conversations"):
                if chat_backups:
                    # Group backups by document
                    from collections import defaultdict
                    grouped_backups = defaultdict(list)
                    
                    for backup in chat_backups:
                        doc = backup.get('document', 'Unknown document')
                        grouped_backups[doc].append(backup)
                    
                    # Show each document group
                    for doc, backups in grouped_backups.items():
                        st.markdown(f"#### {doc}")
                        
                        # Create a dataframe for better display
                        backup_data = [{
                            'Date': b.get('date'),
                            'Time': b.get('time'),
                            'Model': b.get('model'),
                            'Session': b.get('session_id'),
                            'File': b.get('filename')
                        } for b in backups]
                        
                        df = pd.DataFrame(backup_data)
                        st.dataframe(df)
                else:
                    st.info("No saved conversations found.")

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; padding: 10px; color: #999999; font-size: 0.8em; margin-top: 30px;">
        <p>© 2025 SilentCodingLegend AI - Knowledge Base</p>
        <p style="font-size: 0.9em; color: #666666;">Dark theme edition</p>
        <p style="margin-top: 10px;">
            <a href="/Chat_History" target="_self" style="color: #7e57c2; text-decoration: none; font-size: 0.9em;">
                Manage All Chat History 💬
            </a>
        </p>
    </div>
""", unsafe_allow_html=True)