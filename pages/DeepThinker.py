"""
DeepThinker page for SilentCodingLegend AI.
This page utilizes Groq's advanced reasoning capabilities to solve complex problems with explicit reasoning chains.
"""

import streamlit as st
import os
import time
import json
import pandas as pd
from datetime import datetime

# Import configuration and utilities
from src.config import (
    APP_NAME, APP_DESCRIPTION, CURRENT_YEAR,
    MODEL_INFO
)
from src.utils import (
    apply_custom_style, get_groq_client, query_groq_model, get_system_prompt, backup_chat_history
)
from src.performance_opt import cached_query_groq_model

# Models that support reasoning capabilities
REASONING_MODELS = {
    "deepseek-r1-distill-llama-70b": {
        "display_name": "DeepSeek R1 Distill Llama 70B",
        "description": "Advanced reasoning model with explicit reasoning chains",
        "recommended_temp": 0.6
    },
    "qwen-qwq-32b": {
        "display_name": "Qwen QwQ 32B",
        "description": "Specialized for mathematical reasoning and logic tasks",
        "recommended_temp": 0.5
    }
}

# Reasoning formats
REASONING_FORMATS = {
    "raw": "Shows reasoning within <think> tags in the content",
    "parsed": "Separates reasoning into a dedicated field while keeping the response concise",
    "hidden": "Returns only the final answer without showing reasoning steps"
}

# Example problem categories
EXAMPLE_PROBLEMS = {
    "Mathematical Reasoning": [
        "Find all values of x that satisfy the equation: 3x² + 6x - 24 = 0",
        "What is the probability of drawing exactly 2 aces from a standard deck of 52 cards when drawing 5 cards?",
        "If f(x) = 2x² - 3x + 1, find the values of x where f'(x) = 0"
    ],
    "Logical Deduction": [
        "Alice, Bob, and Charlie each have either a red, blue, or green hat. No two people have the same color hat. Alice says, 'I don't see a red hat.' Bob says, 'I see a green hat.' What color hat does each person have?",
        "In a certain town, there are two types of people: knights who always tell the truth, and knaves who always lie. You meet three people, A, B, and C. A says, 'B is a knight.' B says, 'C is a knave.' C says, 'A is a knave.' Who is a knight and who is a knave?",
        "There are 5 houses in a row, each painted a different color. In each house lives a person of different nationality. Each person drinks a different beverage, smokes a different brand of cigarettes, and keeps a different pet. Given the clues: The Brit lives in the red house. The Swede keeps dogs as pets. The Dane drinks tea. The green house is on the left of the white house. The green house owner drinks coffee. The person who smokes Pall Mall keeps birds. The owner of the yellow house smokes Dunhill. The man living in the center house drinks milk. The Norwegian lives in the first house. The man who smokes Blend lives next to the one who keeps cats. The man who keeps horses lives next to the man who smokes Dunhill. The owner who smokes Blue Master drinks beer. The German smokes Prince. The Norwegian lives next to the blue house. The man who smokes Blend has a neighbor who drinks water. Who owns the fish?"
    ],
    "Computer Science Problems": [
        "Explain how the quicksort algorithm works and analyze its time complexity in best, average, and worst cases",
        "Describe a solution to the n-queens problem using backtracking",
        "Design a system to detect cycles in a directed graph and explain the algorithm"
    ],
    "Critical Analysis": [
        "Analyze the ethical implications of using large language models in automated decision-making systems",
        "Compare and contrast declarative and imperative programming paradigms",
        "Evaluate the tradeoffs between privacy and functionality in modern smartphone applications"
    ]
}

# Apply dark theme styling
apply_custom_style()

# Initialize session state for deep thinker history
if "dt_messages" not in st.session_state:
    st.session_state.dt_messages = []
if "dt_selected_model" not in st.session_state:
    st.session_state.dt_selected_model = "deepseek-r1-distill-llama-70b"
if "dt_reasoning_format" not in st.session_state:
    st.session_state.dt_reasoning_format = "raw"
if "dt_temperature" not in st.session_state:
    st.session_state.dt_temperature = 0.6

# Page header
st.title("🧠 DeepThinker")
st.markdown("""
    <div style="padding: 10px; border-radius: 10px; background-color: #2d2d2d; margin-bottom: 20px; border-left: 4px solid #7e57c2;">
        <p style="color: #e0e0e0; margin: 0;">Solve complex problems with step-by-step reasoning and explicit thinking chains</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for model settings and options
with st.sidebar:
    st.markdown("""
        <h2 style="color: #7e57c2; margin-bottom: 20px; text-align: center;">DeepThinker Settings</h2>
    """, unsafe_allow_html=True)
    
    # Model selection with custom styling
    st.markdown('<p style="color: #b39ddb; font-weight: bold;">Select Reasoning Model:</p>', unsafe_allow_html=True)
    
    model_options = list(REASONING_MODELS.keys())
    model_display_names = [REASONING_MODELS[m]["display_name"] for m in model_options]
    
    model_index = model_options.index(st.session_state.dt_selected_model) if st.session_state.dt_selected_model in model_options else 0
    
    selected_display = st.selectbox(
        "Reasoning Model",
        model_display_names,
        index=model_index,
        help="Select a model optimized for reasoning tasks",
        label_visibility="collapsed"
    )
    
    # Get selected model key
    selected_index = model_display_names.index(selected_display)
    model = model_options[selected_index]
    st.session_state.dt_selected_model = model
    
    # Show model description
    st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <p style="color: #e0e0e0; font-size: 0.9em; margin: 0;"><strong>{REASONING_MODELS[model]['display_name']}</strong></p>
            <p style="color: #b39ddb; font-size: 0.85em; margin: 5px 0 0 0;">{REASONING_MODELS[model]['description']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Reasoning format selection
    st.markdown('<p style="color: #b39ddb; font-weight: bold; margin-top: 15px;">Reasoning Format:</p>', unsafe_allow_html=True)
    
    reasoning_format = st.radio(
        "Reasoning Format",
        list(REASONING_FORMATS.keys()),
        index=list(REASONING_FORMATS.keys()).index(st.session_state.dt_reasoning_format),
        format_func=lambda x: x.capitalize(),
        help="Controls how the model's reasoning process is presented",
        label_visibility="collapsed"
    )
    st.session_state.dt_reasoning_format = reasoning_format
    
    # Show format description
    st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 8px; border-radius: 5px; margin-top: 5px;">
            <p style="color: #e0e0e0; font-size: 0.85em; margin: 0;">{REASONING_FORMATS[reasoning_format]}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Temperature control with custom styling
    st.markdown('<p style="color: #b39ddb; font-weight: bold; margin-top: 15px;">Temperature:</p>', unsafe_allow_html=True)
    
    temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.dt_temperature, 
        step=0.1,
        help="Controls randomness in responses. Lower values (0.5-0.6) recommended for reasoning tasks.",
        label_visibility="collapsed"
    )
    st.session_state.dt_temperature = temperature
    
    # Temperature recommendation
    recommended_temp = REASONING_MODELS[model]["recommended_temp"]
    if abs(temperature - recommended_temp) > 0.2:
        st.markdown(f"""
            <div style="background-color: #2d2d2d; padding: 8px; border-radius: 5px; margin-top: 5px; border-left: 2px solid #ff9800;">
                <p style="color: #e0e0e0; font-size: 0.85em; margin: 0;">⚠️ Recommended temperature for this model is {recommended_temp}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Advanced options section
    with st.expander("🔧 Advanced Options", expanded=False):
        # Max tokens setting
        st.markdown('<p style="color: #b39ddb; font-weight: bold;">Max Tokens:</p>', unsafe_allow_html=True)
        
        if "dt_max_tokens" not in st.session_state:
            st.session_state.dt_max_tokens = 2048
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=512,
            max_value=4096,
            value=st.session_state.dt_max_tokens,
            step=256,
            help="Maximum length of model's response. Higher values needed for complex reasoning tasks."
        )
        st.session_state.dt_max_tokens = max_tokens
        
        # Seed setting for reproducibility
        st.markdown('<p style="color: #b39ddb; font-weight: bold; margin-top: 10px;">Random Seed:</p>', unsafe_allow_html=True)
        
        if "dt_seed" not in st.session_state:
            st.session_state.dt_seed = None
        
        use_seed = st.checkbox("Use fixed random seed", value=st.session_state.dt_seed is not None)
        
        if use_seed:
            seed = st.number_input(
                "Seed value",
                value=st.session_state.dt_seed if st.session_state.dt_seed is not None else 42,
                min_value=0,
                max_value=10000,
                help="Fixed seed ensures reproducible results"
            )
            st.session_state.dt_seed = seed
        else:
            st.session_state.dt_seed = None
    
    # About section
    st.divider()
    st.markdown(f"""
        <div style="background-color: #2d2d2d; padding: 15px; border-radius: 10px; border-left: 3px solid #7e57c2;">
            <h3 style="color: #7e57c2; margin-top: 0;">About DeepThinker</h3>
            <p style="color: #e0e0e0; font-size: 0.9em;">DeepThinker leverages Groq's advanced reasoning capabilities to solve complex problems with explicit step-by-step reasoning chains.</p>
            <p style="color: #e0e0e0; font-size: 0.9em;">Best for:</p>
            <ul style="color: #e0e0e0; font-size: 0.85em;">
                <li>Mathematical problems</li>
                <li>Logical reasoning</li>
                <li>Complex analysis</li>
                <li>Step-by-step solutions</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Main content area with tabs
tabs = st.tabs(["Problem Solver", "Example Problems", "History"])

# Check if we should pre-fill a problem from an example
if "dt_current_problem" in st.session_state and "user_input_text" not in st.session_state:
    st.session_state.user_input_text = st.session_state.dt_current_problem

with tabs[0]:
    # Problem input area
    st.markdown("### Enter Your Problem")
    st.markdown("""
        <p style="color: #e0e0e0; font-size: 0.9em;">
            Describe your problem clearly and specify what kind of reasoning you want.
            For best results, ask for step-by-step explanations.
        </p>
    """, unsafe_allow_html=True)
    
    # Initialize the session state variable for user input if it doesn't exist
    if "user_input_text" not in st.session_state:
        st.session_state.user_input_text = ""
    
    user_prompt = st.text_area(
        "Problem description",
        value=st.session_state.user_input_text,
        height=150,
        placeholder="e.g., Solve the equation 3x² + 6x - 24 = 0 step by step, showing all your work and reasoning",
        help="Describe the problem you want to solve with explicit reasoning steps"
    )
    
    # Additional guidance for better results
    with st.expander("Tips for better results", expanded=False):
        st.markdown("""
            ### Tips for Effective Problem Statements:
            
            1. **Be specific about the problem**: Clearly define what you're asking the model to solve.
            
            2. **Request step-by-step reasoning**: Explicitly ask for detailed steps in the solution.
            
            3. **Use precise terminology**: Use correct technical terms for your domain.
            
            4. **Request verification**: Ask the model to check its work at the end.
            
            5. **Avoid ambiguity**: Make sure your problem statement has a clear objective.
            
            6. **Specify the format**: If you want the answer in a specific format, mention it.
            
            **Example of a good problem statement:**
            ```
            Solve the equation 3x² + 6x - 24 = 0 step by step.
            Show your reasoning for each step.
            Verify your solution by substituting back into the original equation.
            ```
        """)
    
    # Process button with customizations
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("Solve with DeepThinker", type="primary", use_container_width=True):
            if not user_prompt:
                st.warning("Please enter a problem to solve.")
            else:
                # Add user message to chat
                st.session_state.dt_messages.append({"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.markdown(user_prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Display spinner while processing
                    with st.spinner("⚡ DeepThinker is reasoning..."):
                        try:
                            # Get selected model params
                            model_id = st.session_state.dt_selected_model
                            reasoning_format = st.session_state.dt_reasoning_format
                            temperature = st.session_state.dt_temperature
                            max_tokens = st.session_state.dt_max_tokens
                            seed = st.session_state.dt_seed
                            
                            # Create a special system prompt for reasoning
                            system_prompt = """You are DeepThinker, an expert AI problem solver that specializes in step-by-step reasoning to solve complex problems.
                            When solving problems, always:
                            1. Break down the problem into manageable steps
                            2. Show your complete reasoning process
                            3. Explain each step clearly and concisely
                            4. Verify your solution at the end
                            5. Be precise and methodical in your approach
                            Avoid skipping steps - the explicit reasoning process is critical for understanding."""
                            
                            # Prepare messages for the API call
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ]
                            
                            # Custom parameters for reasoning queries
                            client = get_groq_client()
                            
                            # Create custom API parameters for reasoning
                            api_params = {
                                "model": model_id,
                                "messages": messages,
                                "temperature": temperature,
                                "max_completion_tokens": max_tokens,
                                "stream": True
                            }
                            
                            # Add reasoning format to system prompt instead of as API parameter
                            if reasoning_format == "raw":
                                api_params["messages"][0]["content"] += "\nPlease show your reasoning within <think> tags."
                            elif reasoning_format == "parsed":
                                api_params["messages"][0]["content"] += "\nPlease provide your reasoning first, then your final answer separately."
                            elif reasoning_format == "hidden":
                                api_params["messages"][0]["content"] += "\nPlease provide only the final answer without showing your reasoning steps."
                            
                            # Add seed if specified
                            if seed is not None:
                                api_params["seed"] = seed
                            
                            # Check if caching is enabled and appropriate
                            use_cache = st.session_state.get("cache_enabled", True) and not api_params["stream"]
                            
                            # Use cached version if streaming is disabled, otherwise direct API call
                            if use_cache and not api_params["stream"]:
                                response = cached_query_groq_model(
                                    model_id=model_id,
                                    messages=messages,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    stream=False,
                                    use_cache=True
                                )
                            else:
                                # Call Groq API with reasoning parameters (no caching for streaming)
                                response = client.chat.completions.create(**api_params)
                            
                            # Process the streaming response
                            for chunk in response:
                                if chunk.choices[0].delta.content:
                                    full_response += chunk.choices[0].delta.content
                                    message_placeholder.markdown(full_response + "▌")
                                    time.sleep(0.01)
                            
                            # Update the placeholder with the complete response
                            message_placeholder.markdown(full_response)
                            
                        except Exception as e:
                            error_message = str(e)
                            st.error(f"Error: {error_message}")
                            
                            # Add more detailed debugging information
                            st.expander("Debug Information", expanded=False).json({
                                "error": error_message,
                                "model": model_id,
                                "api_params": {k: v for k, v in api_params.items() if k != "messages"}
                            })
                            
                            full_response = f"I apologize, but I encountered an error while solving this problem: {error_message}"
                            message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.dt_messages.append({"role": "assistant", "content": full_response})
                
                # Backup chat history with timestamp and model info
                try:
                    backup_path = backup_chat_history(
                        messages=st.session_state.dt_messages,
                        document_name=f"DeepThinker_{model_id}",
                        model_id=model_id
                    )
                    st.toast(f"Reasoning solution backed up", icon="✅")
                except Exception as e:
                    st.toast(f"Error backing up solution: {str(e)}", icon="⚠️")

    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.dt_messages = []
            st.session_state.user_input_text = ""  # Clear the input text
            st.rerun()
    
    # Display previous chat history for this session
    if st.session_state.dt_messages:
        st.markdown("### Current Session")
        
        for i in range(0, len(st.session_state.dt_messages), 2):
            if i < len(st.session_state.dt_messages):
                user_msg = st.session_state.dt_messages[i]
                with st.chat_message("user"):
                    st.markdown(user_msg["content"])
            
            if i+1 < len(st.session_state.dt_messages):
                ai_msg = st.session_state.dt_messages[i+1]
                with st.chat_message("assistant"):
                    st.markdown(ai_msg["content"])

with tabs[1]:
    # Example problems interface
    st.markdown("### Example Problems")
    st.markdown("""
        <p style="color: #e0e0e0; font-size: 0.9em;">
            Select from these example problems to see how DeepThinker uses reasoning to solve them.
        </p>
    """, unsafe_allow_html=True)
    
    # Category selection
    category = st.selectbox(
        "Problem Category",
        options=list(EXAMPLE_PROBLEMS.keys()),
        help="Select a category of problems"
    )
    
    # Problem selection within category
    selected_problem = st.selectbox(
        "Select Problem",
        options=EXAMPLE_PROBLEMS[category],
        help="Choose a specific problem to solve"
    )
    
    # Use example button
    if st.button("Use This Example", type="primary"):
        # Set as current problem and store for input field
        st.session_state.dt_current_problem = selected_problem
        st.session_state.user_input_text = selected_problem
        
        # Switch to the first tab
        st.rerun()
        
    # Preview selected problem
    st.markdown("### Problem Preview")
    st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-top: 10px;">
            <p style="color: #e0e0e0; font-size: 0.95em; margin: 0;">{selected_problem}</p>
        </div>
    """, unsafe_allow_html=True)

with tabs[2]:
    # History of previously solved problems
    st.markdown("### Previously Solved Problems")
    
    from src.utils import get_chat_history_backups, load_chat_history_backup
    
    # Get all chat history backups for DeepThinker
    chat_backups = get_chat_history_backups()
    dt_backups = [b for b in chat_backups if b.get('document', '').startswith('DeepThinker_')]
    
    if not dt_backups:
        st.info("No previous DeepThinker problems found. Solve some problems to build your history!")
    else:
        # Group by date
        from collections import defaultdict
        grouped_backups = defaultdict(list)
        
        for backup in dt_backups:
            date = backup.get('date', 'Unknown date')
            grouped_backups[date].append(backup)
        
        # Sort dates (newest first)
        sorted_dates = sorted(grouped_backups.keys(), reverse=True)
        
        # Create tabs for each date
        date_tabs = st.tabs(sorted_dates)
        
        for i, date in enumerate(sorted_dates):
            with date_tabs[i]:
                # Sort by time (newest first)
                day_backups = sorted(grouped_backups[date], key=lambda x: x.get('time', ''), reverse=True)
                
                for j, backup in enumerate(day_backups):
                    # Extract problem from first message
                    messages = load_chat_history_backup(backup.get('file_path'))
                    problem = messages[0]['content'] if messages and len(messages) > 0 else "Unknown problem"
                    model = backup.get('model', 'Unknown model')
                    time = backup.get('time', 'Unknown time')
                    
                    # Display in an expander
                    with st.expander(f"{time} - {problem[:80]}{'...' if len(problem) > 80 else ''}", expanded=False):
                        st.markdown(f"**Model:** {model}")
                        st.markdown(f"**Time:** {time}")
                        
                        # Display the conversation
                        for msg in messages:
                            role = msg['role']
                            content = msg['content']
                            
                            with st.chat_message(role):
                                st.markdown(content)
                        
                        # Delete button
                        if st.button("Delete this solution", key=f"delete_{date}_{j}"):
                            try:
                                os.remove(backup.get('file_path'))
                                st.success("Solution deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting solution: {str(e)}")

# Footer
st.divider()
st.markdown(f"""
    <div style="text-align: center; padding: 10px; color: #999999; font-size: 0.8em; margin-top: 30px;">
        <p>© {CURRENT_YEAR} {APP_NAME} - DeepThinker</p>
        <p style="font-size: 0.9em; color: #666666;">Powered by Groq Reasoning API</p>
        <p style="margin-top: 10px;">
            <a href="/Chat_History" target="_self" style="color: #7e57c2; text-decoration: none; font-size: 0.9em;">
                View All Chat History 💬
            </a>
        </p>
    </div>
""", unsafe_allow_html=True)
