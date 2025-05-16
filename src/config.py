"""
Configuration module for SilentCodingLegend AI application.
Contains model configurations, app settings, and other global configurations.
"""

import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Application metadata
APP_NAME = "SilentCodingLegend AI"
APP_VERSION = "1.0.0"
APP_ICON = "🧠"
APP_DESCRIPTION = "Your AI coding assistant powered by Groq"
CURRENT_YEAR = datetime.now().year

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_SYSTEM_PROMPT = """You are SilentCodingLegend, an expert AI coding assistant. You provide clear, concise, and accurate programming help. Your specialties include problem-solving, debugging code, explaining programming concepts, and generating code snippets. You should always aim to provide complete and working solutions. Be friendly, professional, and straightforward."""

JSON_SYSTEM_PROMPT_ADDITION = """\n\nIMPORTANT: You must respond in valid JSON format only. Structure your responses as follows:
{
  "answer": "Your detailed answer goes here",
  "code_snippet": "Any code example goes here",
  "language": "The programming language of the code",
  "explanation": "Explanation of the code or concept"
}"""

# UI Configuration
UI_THEME = "dark"  # Default theme
PRIMARY_COLOR = "#7e57c2"

# Dark theme colors
DARK_THEME = {
    "primary_color": "#7e57c2",
    "background_color": "#121212",
    "secondary_background_color": "#1e1e1e",
    "text_color": "#e0e0e0",
    "user_message_color": "#1e3a5f",
    "assistant_message_color": "#2d2d2d",
    "border_color": "#333333",
    "input_background": "#2d2d2d",
    "sidebar_color": "#1a1a1a",
    "code_background": "#2b2b2b",
    "code_text": "#e6e6e6"
}

# Light theme colors
LIGHT_THEME = {
    "primary_color": "#7e57c2",
    "background_color": "#f5f5f5",
    "secondary_background_color": "#ffffff",
    "text_color": "#333333",
    "user_message_color": "#e3f2fd",
    "assistant_message_color": "#f9f9f9",
    "border_color": "#dddddd",
    "input_background": "#ffffff",
    "sidebar_color": "#efefef", 
    "code_background": "#f0f0f0",
    "code_text": "#333333"
}
LIGHT_THEME = {
    "primary_color": "#673ab7",
    "background_color": "#ffffff",
    "secondary_background_color": "#f5f5f5",
    "text_color": "#333333",
    "user_message_color": "#e3f2fd",
    "assistant_message_color": "#f5f5f5",
    "border_color": "#e0e0e0",
    "input_background": "#f9f9f9",
    "sidebar_color": "#f0f0f0",
    "code_background": "#f5f5f5",
    "code_text": "#24292e"
}

# Set default theme colors (from dark theme)
BACKGROUND_COLOR = DARK_THEME["background_color"]
SECONDARY_BACKGROUND_COLOR = DARK_THEME["secondary_background_color"]
TEXT_COLOR = DARK_THEME["text_color"]
USER_MESSAGE_COLOR = DARK_THEME["user_message_color"]
ASSISTANT_MESSAGE_COLOR = DARK_THEME["assistant_message_color"]
BORDER_COLOR = DARK_THEME["border_color"]
INPUT_BACKGROUND = DARK_THEME["input_background"]
SIDEBAR_COLOR = DARK_THEME["sidebar_color"]

# File paths
ASSETS_PATH = "assets"
UPLOAD_PATH = "uploads"
KNOWLEDGE_PATH = "knowledge"
CHAT_HISTORY_PATH = os.path.join(KNOWLEDGE_PATH, "chat_history")
IMAGE_UPLOAD_PATH = os.path.join(UPLOAD_PATH, "images")

# Knowledge Base Configuration
MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSIONS = ["pdf", "txt", "md", "py", "js", "html", "css", "java", "c", "cpp", "json", "xml", "csv"]
VECTOR_DB_PATH = os.path.join(KNOWLEDGE_PATH, "vector_db")

# Default values
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_CATEGORY = "Featured Models"

# Model Information
MODEL_INFO = {
    "llama-3.1-70b-vision": {
        "display_name": "Llama 3.1 70B Vision",
        "req_limit": "500 requests/day",
        "token_limit": "6,000 tokens/minute",
        "recommended_max": 2048,
        "supports_vision": True
    },
    "groq-rewind-llama-4-mvr-17b-128e-instruct": {
        "display_name": "Groq Rewind Llama 4 Mvr Vision",
        "req_limit": "500 requests/day",
        "token_limit": "6,000 tokens/minute",
        "recommended_max": 2048,
        "supports_vision": True
    },
    "allam-2-7b": {
        "display_name": "Allam 2 7B",
        "req_limit": "7,000 requests/day",
        "token_limit": "6,000 tokens/minute",
        "recommended_max": 1500
    },
    "deepseek-r1-distill-llama-70b": {
        "display_name": "DeepSeek R1 Distill Llama 70B",
        "req_limit": "1,000 requests/day",
        "token_limit": "6,000 tokens/minute",
        "recommended_max": 1500
    },
    "gemma-2-9b-instruct": {
        "display_name": "Gemma 2 9B Instruct",
        "req_limit": "14,400 requests/day",
        "token_limit": "15,000 tokens/minute", 
        "recommended_max": 3000
    },
    "groq-compound-beta": {
        "display_name": "Groq Compound Beta",
        "req_limit": "200 requests/day",
        "token_limit": "70,000 tokens/minute",
        "recommended_max": 4000
    },
    "groq-compound-beta-mini": {
        "display_name": "Groq Compound Beta Mini",
        "req_limit": "200 requests/day",
        "token_limit": "70,000 tokens/minute",
        "recommended_max": 4000
    },
    "llama3-70b-8192": {
        "display_name": "Llama 3 70B",
        "req_limit": "14,400 requests/day",
        "token_limit": "6,000 tokens/minute",
        "recommended_max": 2048
    },
    "llama3-8b-8192": {
        "display_name": "Llama 3 8B",
        "req_limit": "14,400 requests/day",
        "token_limit": "6,000 tokens/minute",
        "recommended_max": 2048
    },
    "llama-3.1-8b-instant": {
        "display_name": "Llama 3.1 8B",
        "req_limit": "14,400 requests/day",
        "token_limit": "6,000 tokens/minute",
        "recommended_max": 2048
    },
    "llama-3.3-70b-versatile": {
        "display_name": "Llama 3.3 70B",
        "req_limit": "1,000 requests/day",
        "token_limit": "12,000 tokens/minute",
        "recommended_max": 2500
    },
    "llama-4-maverick-17b-128e-instruct": {
        "display_name": "Llama 4 Maverick 17B",
        "req_limit": "1,000 requests/day",
        "token_limit": "6,000 tokens/minute",
        "recommended_max": 1500
    },
    "meta-llama/llama-4-maverick-17b-128e-instruct": {
        "display_name": "Meta Llama 4 Maverick 17B",
        "req_limit": "1,000 requests/day",
        "token_limit": "6,000 tokens/minute",
        "recommended_max": 1500,
        "supports_vision": True
    },
    "llama-4-scout-instruct": {
        "display_name": "Llama 4 Scout Instruct",
        "req_limit": "1,000 requests/day",
        "token_limit": "30,000 tokens/minute",
        "recommended_max": 3500
    },
    "llama-guard-3-8b": {
        "display_name": "Llama Guard 3 8B",
        "req_limit": "14,400 requests/day",
        "token_limit": "15,000 tokens/minute",
        "recommended_max": 3000
    },
    "mistral-saba-24b": {
        "display_name": "Mistral Saba 24B",
        "req_limit": "1,000 requests/day",
        "token_limit": "6,000 tokens/minute",
        "recommended_max": 1500
    },
    "qwen-qwq-32b": {
        "display_name": "Qwen QwQ 32B",
        "req_limit": "1,000 requests/day",
        "token_limit": "6,000 tokens/minute", 
        "recommended_max": 1500
    },
    "meta-llama/llama-guard-4-12b": {
        "display_name": "Meta Llama Guard 4 12B",
        "req_limit": "14,400 requests/day", 
        "token_limit": "15,000 tokens/minute",
        "recommended_max": 3000
    },
}

# Vision Models 
VISION_MODELS = {
    "llama-3.1-70b-vision": {
        "display_name": "Llama 3.1 70B Vision",
        "description": "Advanced multimodal model that can process both text and images",
        "recommended_temp": 0.7,
        "req_limit": "500 requests/day",
        "token_limit": "6,000 tokens/minute",
        "recommended_max": 2048,
        "supported_image_types": ["png", "jpg", "jpeg", "webp"]
    },
    "groq-rewind-llama-4-mvr-17b-128e-instruct": {
        "display_name": "Groq Rewind Llama 4 Mvr Vision",
        "description": "Vision-capable model optimized for image understanding",
        "recommended_temp": 0.5, 
        "req_limit": "500 requests/day",
        "token_limit": "6,000 tokens/minute",
        "recommended_max": 2048,
        "supported_image_types": ["png", "jpg", "jpeg", "webp"]
    },
    "meta-llama/llama-4-maverick-17b-128e-instruct": {
        "display_name": "Llama 4 Maverick 17B",
        "description": "Meta's multimodal model with strong vision capabilities",
        "recommended_temp": 0.6,
        "req_limit": "1,000 requests/day",
        "token_limit": "6,000 tokens/minute",
        "recommended_max": 2048,
        "supported_image_types": ["png", "jpg", "jpeg", "webp"]
    }
}

# Allowed Image Extensions
ALLOWED_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "webp"]
MAX_IMAGE_SIZE_MB = 20

# Model Categories
MODEL_CATEGORIES = {
    "Featured Models": ["llama3-70b-8192", "llama3-8b-8192", "mistral-saba-24b", "groq-compound-beta"],
    "Llama Family": ["llama3-70b-8192", "llama3-8b-8192", "llama-3.1-8b-instant", "llama-3.3-70b-versatile", 
                     "llama-4-scout-instruct"],
    "Safety & Moderation": ["llama-guard-3-8b", "meta-llama/llama-guard-4-12b"],
    "Vision Models": ["llama-3.1-70b-vision", "groq-rewind-llama-4-mvr-17b-128e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct"],
    "Other Models": ["allam-2-7b", "deepseek-r1-distill-llama-70b", "gemma-2-9b-instruct", 
                    "groq-compound-beta", "groq-compound-beta-mini", "qwen-qwq-32b"]
}

# Tool usage configurations for different models
TOOL_USAGE_CONFIG = {
    "default": {
        "supports_tools": False,
    },
    "llama-4-scout-instruct": {
        "supports_tools": True,
        "supports_json_mode": True,
    },
}

# Knowledge Base specific model configurations
KB_MODEL_CONFIG = {
    "default_embedding_model": "llama3-8b-8192",
    "default_retrieval_model": "llama3-70b-8192",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "similarity_top_k": 5,
}

# Function to get recommended max tokens based on model
def get_recommended_max_tokens(model_id):
    """Get the recommended max token value for a specific model."""
    if model_id in MODEL_INFO:
        return MODEL_INFO[model_id]["recommended_max"]
    return DEFAULT_MAX_TOKENS

# Function to check if a model supports tools
def model_supports_tools(model_id):
    """Check if a model supports tool usage."""
    if model_id in TOOL_USAGE_CONFIG:
        return TOOL_USAGE_CONFIG[model_id]["supports_tools"]
    return TOOL_USAGE_CONFIG["default"]["supports_tools"]