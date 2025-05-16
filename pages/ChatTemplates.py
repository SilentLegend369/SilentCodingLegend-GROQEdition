"""
Chat Templates management page for SilentCodingLegend AI.
This page allows users to view, manage, and create custom chat templates.
"""

import streamlit as st

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Chat Templates | SilentCodingLegend AI",
    page_icon="📝",
    layout="wide",
)

import json
import os
from datetime import datetime
import pandas as pd

from src.chat_templates import (
    ChatTemplate, get_template_by_id, get_all_templates,
    get_templates_by_category, get_all_categories
)
from src.theme import apply_theme_style

# Apply custom theme
apply_theme_style()

# User custom templates directory
USER_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'templates')
os.makedirs(USER_TEMPLATES_DIR, exist_ok=True)

# Function to load user custom templates
def load_user_templates():
    templates = {}
    if not os.path.exists(USER_TEMPLATES_DIR):
        return templates
    
    for filename in os.listdir(USER_TEMPLATES_DIR):
        if not filename.endswith('.json'):
            continue
        
        try:
            with open(os.path.join(USER_TEMPLATES_DIR, filename), 'r') as f:
                template_data = json.load(f)
                template_id = os.path.splitext(filename)[0]
                templates[template_id] = ChatTemplate.from_dict(template_data)
        except Exception as e:
            st.error(f"Error loading template {filename}: {str(e)}")
    
    return templates

# Function to save a user custom template
def save_user_template(template_id, template):
    template_path = os.path.join(USER_TEMPLATES_DIR, f"{template_id}.json")
    with open(template_path, 'w') as f:
        json.dump(template.to_dict(), f, indent=2)
    return template_path

# Function to delete a user custom template
def delete_user_template(template_id):
    template_path = os.path.join(USER_TEMPLATES_DIR, f"{template_id}.json")
    if os.path.exists(template_path):
        os.remove(template_path)
        return True
    return False

# Header
st.title("📝 Chat Templates")
st.markdown("""
This page allows you to view, manage, and create custom chat templates. 
Templates help you quickly start conversations with optimized prompts for specific tasks.
""")

# Create tabs for Browse and Create
browse_tab, create_tab, manage_tab = st.tabs(["Browse Templates", "Create Template", "Manage Custom Templates"])

with browse_tab:
    st.header("Browse Available Templates")
    
    # Filter options
    col1, col2 = st.columns([1, 3])
    with col1:
        category_filter = st.selectbox(
            "Filter by category",
            ["All Categories"] + get_all_categories(),
            index=0
        )
    
    with col2:
        search_query = st.text_input("Search templates", placeholder="Enter keywords...")
    
    # Get templates based on filter
    if category_filter == "All Categories":
        templates = get_all_templates()
    else:
        templates = get_templates_by_category(category_filter)
    
    # Add user templates
    user_templates = load_user_templates()
    all_templates = {**templates, **user_templates}
    
    # Apply search filter if provided
    if search_query:
        search_query = search_query.lower()
        all_templates = {
            tid: temp for tid, temp in all_templates.items() 
            if search_query in temp.title.lower() or 
               search_query in temp.description.lower() or
               search_query in temp.category.lower() or
               search_query in temp.system_prompt.lower()
        }
    
    # Display templates in a grid
    if not all_templates:
        st.info("No templates found matching your criteria.")
    else:
        # Sort templates by category and title
        sorted_templates = sorted(
            all_templates.items(), 
            key=lambda x: (x[1].category, x[1].title)
        )
        
        # Display templates grouped by category
        current_category = None
        
        for template_id, template in sorted_templates:
            # Add category header if new category
            if template.category != current_category:
                st.subheader(f"{template.category.capitalize()}")
                current_category = template.category
            
            # Template card
            with st.expander(f"{template.icon} {template.title}", expanded=False):
                st.markdown(f"**Description:** {template.description}")
                st.markdown(f"**Category:** {template.category}")
                
                # Check if it's a user template
                is_user_template = template_id in user_templates
                if is_user_template:
                    st.info("This is a custom template created by you.")
                
                # Model recommendation
                if template.recommended_model:
                    st.markdown(f"**Recommended Model:** {template.recommended_model}")
                    st.markdown(f"**Recommended Temperature:** {template.recommended_temperature}")
                
                # System prompt (not using nested expander)
                st.markdown("**System Prompt:**")
                st.code(template.system_prompt)
                
                # Suggested first prompt (not using nested expander)
                if template.suggested_first_prompt:
                    st.markdown("**Suggested First Prompt:**")
                    st.markdown(f"```\n{template.suggested_first_prompt}\n```")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use This Template", key=f"use_{template_id}"):
                        # Store the template in session state for the main app to use
                        st.session_state.current_template = template_id
                        st.success(f"Selected template: {template.title}. Please go to the home page to start using it.")
                
                with col2:
                    if is_user_template:
                        if st.button("Delete Template", key=f"delete_{template_id}"):
                            if delete_user_template(template_id):
                                st.success(f"Template '{template.title}' deleted.")
                                st.rerun()
                            else:
                                st.error("Failed to delete template.")
                    else:
                        if st.button("Duplicate & Edit", key=f"duplicate_{template_id}"):
                            # Setup the create tab with this template's values
                            st.session_state.template_to_duplicate = template.to_dict()
                            st.session_state.active_tab = "create"
                            st.rerun()

with create_tab:
    st.header("Create Custom Template")
    st.markdown("""
    Create your own custom chat template. These templates will be saved for your use and appear alongside 
    the predefined templates in the main application.
    """)
    
    # Check if we're duplicating a template
    if "template_to_duplicate" in st.session_state:
        template_data = st.session_state.template_to_duplicate
        # Clear after use
        del st.session_state.template_to_duplicate
    else:
        template_data = {
            "title": "",
            "description": "",
            "system_prompt": "",
            "category": "",
            "icon": "💬",
            "suggested_first_prompt": "",
            "recommended_model": None,
            "recommended_temperature": 0.7
        }
    
    # Input fields
    template_title = st.text_input("Template Title", value=template_data["title"])
    template_description = st.text_area("Description", value=template_data["description"], height=100)
    
    col1, col2 = st.columns(2)
    with col1:
        # Either use existing categories or create new
        existing_categories = get_all_categories()
        custom_category = st.checkbox("Create new category", value=template_data["category"] not in existing_categories)
        
        if custom_category:
            template_category = st.text_input("Category", value=template_data["category"] if template_data["category"] not in existing_categories else "")
        else:
            template_category = st.selectbox("Category", existing_categories, index=existing_categories.index(template_data["category"]) if template_data["category"] in existing_categories else 0)
    
    with col2:
        # Icon selection
        icons = ["💬", "🤖", "💻", "📊", "🔍", "📝", "🧠", "📚", "⚙️", "🛠️", "👨‍💻", "🔬", "🎨", "🎯", "💡", "🧩", "📱", "🌐", "📈", "🔒", "🧪"]
        template_icon = st.selectbox("Icon", icons, index=icons.index(template_data["icon"]) if template_data["icon"] in icons else 0)
    
    # System prompt
    st.markdown("### System Prompt (Instructions for the AI)")
    system_prompt = st.text_area("System Prompt", value=template_data["system_prompt"], height=200, help="This is the instruction that guides how the AI responds")
    
    # Optional fields
    st.markdown("### Optional Settings")
    
    suggested_prompt = st.text_area("Suggested First Prompt", value=template_data["suggested_first_prompt"], height=100, help="A suggested first message for the user to send")
    
    col1, col2 = st.columns(2)
    with col1:
        # Model recommendation
        recommended_model = st.text_input("Recommended Model (optional)", value=template_data["recommended_model"] if template_data["recommended_model"] else "")
    
    with col2:
        # Temperature recommendation
        recommended_temp = st.slider("Recommended Temperature", 0.0, 1.0, value=float(template_data["recommended_temperature"]), step=0.1)
    
    # Save button
    if st.button("Save Template"):
        if not template_title:
            st.error("Template title is required.")
        elif not system_prompt:
            st.error("System prompt is required.")
        elif not template_category:
            st.error("Category is required.")
        else:
            try:
                # Create template object
                template = ChatTemplate(
                    title=template_title,
                    description=template_description,
                    system_prompt=system_prompt,
                    category=template_category,
                    icon=template_icon,
                    suggested_first_prompt=suggested_prompt if suggested_prompt else None,
                    recommended_model=recommended_model if recommended_model else None,
                    recommended_temperature=recommended_temp
                )
                
                # Generate a unique ID
                template_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}_{template_title.replace(' ', '_').lower()}"
                
                # Save template
                save_path = save_user_template(template_id, template)
                
                st.success(f"Template saved successfully: {template_title}")
                st.info(f"Your template is now available in the main application and in the Browse Templates tab.")
                
                # Reset form
                for key in ["template_title", "template_description", "template_category", 
                           "template_icon", "system_prompt", "suggested_prompt", 
                           "recommended_model", "recommended_temp"]:
                    if key in st.session_state:
                        del st.session_state[key]
                
            except Exception as e:
                st.error(f"Error saving template: {str(e)}")

with manage_tab:
    st.header("Manage Custom Templates")
    
    user_templates = load_user_templates()
    
    if not user_templates:
        st.info("You don't have any custom templates yet. Create one in the 'Create Template' tab.")
    else:
        # Display user templates in a table
        template_data = []
        for template_id, template in user_templates.items():
            template_data.append({
                "ID": template_id,
                "Title": template.title,
                "Category": template.category,
                "Description": template.description
            })
        
        df = pd.DataFrame(template_data)
        st.dataframe(df, use_container_width=True)
        
        # Export/Import section
        st.subheader("Export/Import Templates")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Export Templates")
            if st.button("Export All Custom Templates"):
                # Create a JSON with all templates
                export_data = {template_id: template.to_dict() for template_id, template in user_templates.items()}
                export_json = json.dumps(export_data, indent=2)
                
                st.download_button(
                    label="Download Templates JSON",
                    data=export_json,
                    file_name="custom_chat_templates.json",
                    mime="application/json"
                )
        
        with col2:
            st.markdown("#### Import Templates")
            uploaded_file = st.file_uploader("Upload Templates JSON", type=["json"])
            
            if uploaded_file is not None:
                try:
                    templates_to_import = json.load(uploaded_file)
                    if not templates_to_import:
                        st.warning("The uploaded file doesn't contain any templates.")
                    else:
                        # Count imported templates
                        import_count = 0
                        
                        # Process each template
                        for template_id, template_data in templates_to_import.items():
                            # Make sure it's a user template ID
                            if not template_id.startswith("user_"):
                                template_id = f"user_imported_{datetime.now().strftime('%Y%m%d%H%M%S')}_{import_count}"
                            
                            # Create and save the template
                            template = ChatTemplate.from_dict(template_data)
                            save_user_template(template_id, template)
                            import_count += 1
                        
                        st.success(f"Successfully imported {import_count} templates.")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error importing templates: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>Templates help you get more consistent and specialized results from the AI.</p>
</div>
""", unsafe_allow_html=True)
