"""
Document processing utilities for the Knowledge Base.
"""

import os
import tempfile
import logging
from typing import List, Dict, Any, Optional
import streamlit as st
import PyPDF2
from src.config import KB_MODEL_CONFIG, MAX_FILE_SIZE_MB

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            logger.info(f"Processing PDF with {len(pdf_reader.pages)} pages")
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
    except Exception as e:
        error_msg = f"Error extracting text from PDF: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return ""
    return text

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various file types."""
    # Get file extension
    extension = file_path.split(".")[-1].lower() if "." in file_path else ""
    
    if extension == "pdf":
        return extract_text_from_pdf(file_path)
    
    # For text files, just read the content
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        try:
            with open(file_path, "r", encoding="latin-1") as file:
                return file.read()
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return ""
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks of specified size."""
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        # If we're not at the end of the text, try to find a good break point
        if end < text_length:
            # Try to break at a paragraph
            paragraph_break = text.rfind("\n\n", start, end)
            if paragraph_break != -1 and paragraph_break > start + 200:  # Ensure minimum chunk size
                end = paragraph_break + 2  # Include the paragraph break
            else:
                # Try to break at a sentence
                sentence_breaks = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
                for break_char in sentence_breaks:
                    sentence_break = text.rfind(break_char, start, end)
                    if sentence_break != -1 and sentence_break > start + 200:  # Ensure minimum chunk size
                        end = sentence_break + len(break_char)
                        break
        
        # Add the chunk to our list
        chunks.append(text[start:end])
        
        # Set the new start with overlap
        start = end - chunk_overlap
        
    logger.info(f"Chunked text into {len(chunks)} parts")
    return chunks

def process_document(file_path: str) -> List[str]:
    """Process a document into chunks ready for embedding."""
    # Extract text from file
    text = extract_text_from_file(file_path)
    
    # Chunk the text
    chunks = chunk_text(
        text, 
        chunk_size=KB_MODEL_CONFIG["chunk_size"], 
        chunk_overlap=KB_MODEL_CONFIG["chunk_overlap"]
    )
    
    return chunks

def format_chunks_with_metadata(chunks: List[str], file_name: str) -> List[Dict[str, Any]]:
    """Format text chunks with metadata."""
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        formatted_chunks.append({
            "text": chunk,
            "metadata": {
                "source": file_name,
                "chunk_id": i,
                "total_chunks": len(chunks)
            }
        })
    return formatted_chunks
