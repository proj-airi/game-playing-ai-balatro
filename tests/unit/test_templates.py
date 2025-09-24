"""Unit tests for prompt template system."""

import pytest
from src.ai_balatro.ai.templates.prompt_template import (
    PromptTemplate,
    TemplateSection,
    TemplateType,
    PromptTemplateManager,
    render_prompt,
    get_system_prompt
)


class TestTemplateSection:
    """Test TemplateSection class."""

    def test_section_creation(self):
        """Test creating template sections."""
        section = TemplateSection(
            name="test_section",
            content="Test content with {variable}",
            required=True,
            conditions=["var_exists:variable"]
        )

        assert section.name == "test_section"
        assert section.content == "Test content with {variable}"
        assert section.required is True
        assert "var_exists:variable" in section.conditions

    def test_section_defaults(self):
        """Test section default values."""
        section = TemplateSection("basic", "Basic content")

        assert section.required is True
        assert section.conditions == []


class TestPromptTemplate:
    """Test PromptTemplate class."""

    def test_template_creation(self):
        """Test creating prompt templates."""
        template = PromptTemplate(
            name="test_template",
            template_type=TemplateType.ANALYSIS,
            variables={"test_var": "test_value"}
        )

        assert template.name == "test_template"
        assert template.template_type == TemplateType.ANALYSIS
        assert template.variables["test_var"] == "test_value"
        assert len(template.sections) == 0

    def test_add_section(self):
        """Test adding sections to template."""
        template = PromptTemplate("test", TemplateType.USER)

        template.add_section("intro", "Introduction: {topic}")
        template.add_section(
            "details",
            "Details: {details}",
            required=False,
            conditions=["var_exists:details"]
        )

        assert len(template.sections) == 2
        assert template.sections[0].name == "intro"
        assert template.sections[1].required is False

    def test_basic_rendering(self):
        """Test basic template rendering."""
        template = PromptTemplate("test", TemplateType.USER)
        template.add_section("greeting", "Hello {name}!")
        template.add_section("message", "Your message: {message}")

        context = {"name": "Alice", "message": "How are you?"}
        result = template.render(context)

        assert "Hello Alice!" in result
        assert "Your message: How are you?" in result
        assert result.count("\n\n") == 1  # Sections separated by double newlines

    def test_missing_required_variables(self):
        """Test handling missing required variables."""
        template = PromptTemplate("test", TemplateType.USER)
        template.add_section("required_section", "Value: {required_var}", required=True)

        with pytest.raises(ValueError, match="Required template variable missing"):
            template.render({})

    def test_missing_optional_variables(self):
        """Test handling missing optional variables."""
        template = PromptTemplate("test", TemplateType.USER)
        template.add_section("optional", "Optional: {optional_var}", required=False)

        # Should not raise error, section should be skipped
        result = template.render({})
        assert result == ""

    def test_list_variable_formatting(self):
        """Test formatting list variables."""
        template = PromptTemplate("test", TemplateType.USER)
        template.add_section("items", "Items:\n{item_list}")

        context = {"item_list": ["Item 1", "Item 2", "Item 3"]}
        result = template.render(context)

        assert "- Item 1" in result
        assert "- Item 2" in result
        assert "- Item 3" in result

    def test_dict_variable_formatting(self):
        """Test formatting dict variables."""
        template = PromptTemplate("test", TemplateType.USER)
        template.add_section("data", "Data:\n{data_dict}")

        context = {"data_dict": {"key1": "value1", "key2": "value2"}}
        result = template.render(context)

        assert "- key1: value1" in result
        assert "- key2: value2" in result

    def test_condition_evaluation(self):
        """Test condition evaluation."""
        template = PromptTemplate("test", TemplateType.USER)

        # var_exists condition
        template.add_section(
            "conditional1",
            "Exists: {var1}",
            conditions=["var_exists:var1"]
        )

        # var_equals condition
        template.add_section(
            "conditional2",
            "Equals test: {var2}",
            conditions=["var_equals:var2:test_value"]
        )

        # var_not_empty condition
        template.add_section(
            "conditional3",
            "Not empty: {var3}",
            conditions=["var_not_empty:var3"]
        )

        # Test with conditions met
        context = {
            "var1": "present",
            "var2": "test_value",
            "var3": "not empty"
        }
        result = template.render(context)

        assert "Exists: present" in result
        assert "Equals test: test_value" in result
        assert "Not empty: not empty" in result

        # Test with conditions not met
        context_fail = {
            "var2": "wrong_value",
            "var3": ""
        }
        result_fail = template.render(context_fail)

        assert "Exists:" not in result_fail
        assert "Equals test:" not in result_fail
        assert "Not empty:" not in result_fail

    def test_multiple_conditions(self):
        """Test multiple conditions on one section."""
        template = PromptTemplate("test", TemplateType.USER)
        template.add_section(
            "multi_condition",
            "Both exist: {var1} and {var2}",
            conditions=["var_exists:var1", "var_exists:var2"]
        )

        # Both conditions met
        result1 = template.render({"var1": "a", "var2": "b"})
        assert "Both exist: a and b" in result1

        # One condition not met
        result2 = template.render({"var1": "a"})
        assert "Both exist:" not in result2

    def test_empty_sections_excluded(self):
        """Test that empty sections are excluded."""
        template = PromptTemplate("test", TemplateType.USER)
        template.add_section("empty", "")
        template.add_section("whitespace", "   \n  ")
        template.add_section("content", "Real content")

        result = template.render({})
        assert "Real content" in result
        assert result.strip() == "Real content"


class TestPromptTemplateManager:
    """Test PromptTemplateManager class."""

    def test_manager_initialization(self):
        """Test manager initialization with default templates."""
        manager = PromptTemplateManager()

        # Should have default templates
        templates = manager.list_templates()
        expected_templates = [
            "balatro_system",
            "game_state_analysis",
            "strategic_planning",
            "action_execution"
        ]

        for template_name in expected_templates:
            assert template_name in templates

    def test_register_template(self):
        """Test registering custom templates."""
        manager = PromptTemplateManager()

        custom_template = PromptTemplate("custom", TemplateType.USER)
        custom_template.add_section("test", "Test content")

        manager.register_template(custom_template)

        assert "custom" in manager.list_templates()
        retrieved = manager.get_template("custom")
        assert retrieved is custom_template

    def test_render_by_name(self):
        """Test rendering templates by name."""
        manager = PromptTemplateManager()

        context = {
            "game_state_summary": "Test state",
            "card_list": ["Card A", "Card B"]
        }

        result = manager.render_template("game_state_analysis", context)

        assert "Test state" in result
        assert "Card A" in result

    def test_nonexistent_template(self):
        """Test handling nonexistent templates."""
        manager = PromptTemplateManager()

        with pytest.raises(ValueError, match="Template 'nonexistent' not found"):
            manager.render_template("nonexistent", {})

        assert manager.get_template("nonexistent") is None


class TestDefaultTemplates:
    """Test default Balatro templates."""

    def test_balatro_system_template(self):
        """Test Balatro system template."""
        result = render_prompt("balatro_system", {})

        assert "Balatro" in result
        assert "poker-themed" in result
        assert "analyze" in result.lower()

    def test_game_state_analysis_template(self):
        """Test game state analysis template."""
        context = {
            "game_state_summary": "Round 3, Score: 5000",
            "card_list": ["Ace of Hearts", "King of Spades"]
        }

        result = render_prompt("game_state_analysis", context)

        assert "CURRENT GAME STATE" in result
        assert "Round 3, Score: 5000" in result
        assert "DETECTED CARDS" in result
        assert "Ace of Hearts" in result

    def test_strategic_planning_template(self):
        """Test strategic planning template."""
        context = {
            "available_cards": ["Ace", "King", "Queen"],
            "objective": "Score 10,000 points"
        }

        result = render_prompt("strategic_planning", context)

        assert "AVAILABLE CARDS" in result
        assert "Ace" in result
        assert "strategic plan" in result.lower()

    def test_action_execution_template(self):
        """Test action execution template."""
        context = {
            "situation_summary": "Need to play cards",
            "available_actions": ["play_cards", "discard_cards"]
        }

        result = render_prompt("action_execution", context)

        assert "CURRENT SITUATION" in result
        assert "Need to play cards" in result
        assert "AVAILABLE ACTIONS" in result

    def test_template_with_optional_sections(self):
        """Test templates with optional sections."""
        # Test without optional context
        context_minimal = {
            "game_state_summary": "Basic state",
            "card_list": ["Card 1"]
        }

        result_minimal = render_prompt("game_state_analysis", context_minimal)
        assert "CARD DESCRIPTIONS" not in result_minimal

        # Test with optional context
        context_full = {
            "game_state_summary": "Full state",
            "card_list": ["Card 1"],
            "card_descriptions": {"0": "Powerful card"}
        }

        result_full = render_prompt("game_state_analysis", context_full)
        assert "CARD DESCRIPTIONS" in result_full
        assert "Powerful card" in result_full


class TestUtilityFunctions:
    """Test utility functions."""

    def test_render_prompt_function(self):
        """Test render_prompt convenience function."""
        context = {"game_state_summary": "Test", "card_list": ["Card"]}
        result = render_prompt("game_state_analysis", context)

        assert isinstance(result, str)
        assert "Test" in result

    def test_get_system_prompt_function(self):
        """Test get_system_prompt function."""
        balatro_prompt = get_system_prompt("balatro")
        default_prompt = get_system_prompt("unknown")

        assert "Balatro" in balatro_prompt
        assert "helpful AI assistant" in default_prompt

    def test_template_types(self):
        """Test all template types are valid."""
        types_to_test = [
            TemplateType.SYSTEM,
            TemplateType.USER,
            TemplateType.ANALYSIS,
            TemplateType.PLANNING,
            TemplateType.EXECUTION
        ]

        for template_type in types_to_test:
            template = PromptTemplate("test", template_type)
            assert template.template_type == template_type


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_circular_variable_references(self):
        """Test handling of variable references that might cause issues."""
        template = PromptTemplate("test", TemplateType.USER)
        template.add_section("circular", "{var1} references {var2}")

        context = {"var1": "Value1", "var2": "{var1}"}
        result = template.render(context)

        # Should not cause infinite recursion
        assert "Value1" in result
        assert "{var1}" in result  # var2's value contains literal {var1}

    def test_special_characters_in_variables(self):
        """Test handling special characters in variable values."""
        template = PromptTemplate("test", TemplateType.USER)
        template.add_section("special", "Content: {special_var}")

        context = {"special_var": "Line 1\nLine 2\nSpecial chars: !@#$%"}
        result = template.render(context)

        assert "Line 1\nLine 2" in result
        assert "!@#$%" in result

    def test_none_values(self):
        """Test handling None values in context."""
        template = PromptTemplate("test", TemplateType.USER)
        template.add_section("none_test", "Value: {none_var}", required=False)

        context = {"none_var": None}

        # Section should be skipped due to None value
        result = template.render(context)
        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])