"""
Tests for chat templates functionality.
"""

import pytest
from src.chat_templates import (
    ChatTemplate, get_template_by_id, get_all_templates,
    get_templates_by_category, get_all_categories
)

class TestChatTemplates:
    
    def test_chat_template_creation(self):
        """Test ChatTemplate class creation works correctly."""
        template = ChatTemplate(
            title="Test Template",
            description="Test description",
            system_prompt="You are a test assistant.",
            category="test",
            icon="🧪",
            suggested_first_prompt="This is a test prompt.",
            recommended_model="test-model",
            recommended_temperature=0.5
        )
        
        assert template.title == "Test Template"
        assert template.description == "Test description"
        assert template.system_prompt == "You are a test assistant."
        assert template.category == "test"
        assert template.icon == "🧪"
        assert template.suggested_first_prompt == "This is a test prompt."
        assert template.recommended_model == "test-model"
        assert template.recommended_temperature == 0.5
    
    def test_template_conversion_to_dict(self):
        """Test template conversion to dictionary."""
        template = ChatTemplate(
            title="Test Template",
            description="Test description",
            system_prompt="Test prompt",
            category="test"
        )
        
        template_dict = template.to_dict()
        
        assert template_dict["title"] == "Test Template"
        assert template_dict["description"] == "Test description"
        assert template_dict["system_prompt"] == "Test prompt"
        assert template_dict["category"] == "test"
        assert "icon" in template_dict
        assert "recommended_temperature" in template_dict
    
    def test_template_from_dict(self):
        """Test creating template from dictionary."""
        template_dict = {
            "title": "Dict Template",
            "description": "From dictionary",
            "system_prompt": "Dict prompt",
            "category": "dict",
            "icon": "📝",
            "recommended_temperature": 0.8
        }
        
        template = ChatTemplate.from_dict(template_dict)
        
        assert template.title == "Dict Template"
        assert template.description == "From dictionary"
        assert template.system_prompt == "Dict prompt"
        assert template.category == "dict"
        assert template.icon == "📝"
        assert template.recommended_temperature == 0.8
    
    def test_get_template_by_id(self):
        """Test getting a template by ID."""
        template = get_template_by_id("code_debugging")
        
        assert template is not None
        assert template.title == "Code Debugging Assistant"
        assert template.category == "coding"
        assert "debug" in template.system_prompt.lower()
    
    def test_get_nonexistent_template(self):
        """Test getting a template that doesn't exist."""
        template = get_template_by_id("nonexistent_template")
        
        assert template is None
    
    def test_get_all_templates(self):
        """Test getting all templates."""
        templates = get_all_templates()
        
        assert templates is not None
        assert len(templates) > 0
        assert "code_debugging" in templates
        assert "creative_writing" in templates
    
    def test_get_templates_by_category(self):
        """Test filtering templates by category."""
        coding_templates = get_templates_by_category("coding")
        
        assert coding_templates is not None
        assert len(coding_templates) > 0
        assert all(t.category == "coding" for t in coding_templates.values())
        
        # Check that the code_debugging template is in the coding category
        assert "code_debugging" in coding_templates
    
    def test_get_all_categories(self):
        """Test getting all template categories."""
        categories = get_all_categories()
        
        assert categories is not None
        assert len(categories) > 0
        assert "coding" in categories
        assert "creativity" in categories
