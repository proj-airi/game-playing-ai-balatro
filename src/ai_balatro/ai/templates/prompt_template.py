"""Simple prompt template system for structured LLM prompts."""

from typing import Dict, Any, List, Optional, Union
import re
from dataclasses import dataclass, field
from enum import Enum


class TemplateType(Enum):
    """Types of prompt templates."""
    SYSTEM = "system"
    USER = "user"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"


@dataclass
class TemplateSection:
    """A section of a prompt template."""
    name: str
    content: str
    required: bool = True
    conditions: List[str] = field(default_factory=list)


@dataclass
class PromptTemplate:
    """Template for generating structured prompts."""
    name: str
    template_type: TemplateType
    sections: List[TemplateSection] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_section(self, name: str, content: str, required: bool = True, conditions: List[str] = None):
        """Add a section to the template."""
        section = TemplateSection(
            name=name,
            content=content,
            required=required,
            conditions=conditions or []
        )
        self.sections.append(section)

    def render(self, context: Dict[str, Any]) -> str:
        """Render the template with given context."""
        rendered_sections = []

        for section in self.sections:
            # Check if section should be included
            if not self._should_include_section(section, context):
                continue

            # Render section content
            try:
                rendered_content = self._render_content(section.content, context)
                if rendered_content.strip():  # Only add non-empty sections
                    rendered_sections.append(rendered_content)
            except KeyError as e:
                if section.required:
                    raise ValueError(f"Required template variable missing: {e}")
                # Skip optional sections with missing variables

        return "\n\n".join(rendered_sections)

    def _should_include_section(self, section: TemplateSection, context: Dict[str, Any]) -> bool:
        """Check if section should be included based on conditions."""
        if not section.conditions:
            return True

        for condition in section.conditions:
            if not self._evaluate_condition(condition, context):
                return False
        return True

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a simple condition."""
        # Simple conditions: "var_exists:variable_name" or "var_equals:variable_name:value"
        if condition.startswith("var_exists:"):
            var_name = condition.split(":", 1)[1]
            return var_name in context and context[var_name] is not None

        if condition.startswith("var_equals:"):
            parts = condition.split(":", 2)
            if len(parts) == 3:
                var_name, expected_value = parts[1], parts[2]
                return context.get(var_name) == expected_value

        if condition.startswith("var_not_empty:"):
            var_name = condition.split(":", 1)[1]
            value = context.get(var_name)
            return value is not None and str(value).strip() != ""

        return True

    def _render_content(self, content: str, context: Dict[str, Any]) -> str:
        """Render content with variable substitution."""
        # Simple variable substitution: {variable_name}
        def replace_var(match):
            var_name = match.group(1)
            if var_name in context:
                value = context[var_name]
                if isinstance(value, (list, dict)):
                    # Format complex types nicely
                    if isinstance(value, list):
                        return "\n".join(f"- {item}" for item in value)
                    else:
                        return "\n".join(f"- {k}: {v}" for k, v in value.items())
                return str(value)
            else:
                return match.group(0)  # Keep original if not found

        return re.sub(r'\{([^}]+)\}', replace_var, content)


class PromptTemplateManager:
    """Manages a collection of prompt templates."""

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._register_default_templates()

    def register_template(self, template: PromptTemplate):
        """Register a template."""
        self.templates[template.name] = template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        return self.templates.get(name)

    def render_template(self, name: str, context: Dict[str, Any]) -> str:
        """Render template by name."""
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        return template.render(context)

    def list_templates(self) -> List[str]:
        """List all template names."""
        return list(self.templates.keys())

    def _register_default_templates(self):
        """Register default templates for Balatro agent."""

        # System template for agent initialization
        system_template = PromptTemplate(
            name="balatro_system",
            template_type=TemplateType.SYSTEM,
        )
        system_template.add_section(
            "role",
            "You are an AI agent playing Balatro, a poker-themed roguelike deckbuilder game. "
            "Your goal is to analyze the game state, make strategic decisions, and execute actions through function calls."
        )
        system_template.add_section(
            "capabilities",
            "You can:\n"
            "- Analyze detected cards and game elements\n"
            "- Plan strategic moves for optimal scoring\n"
            "- Execute actions like selecting cards, playing hands, or using special cards\n"
            "- Learn from game outcomes to improve decision-making"
        )
        system_template.add_section(
            "instructions",
            "Always think step by step:\n"
            "1. Analyze the current situation\n"
            "2. Consider available options and their outcomes\n"
            "3. Choose the action that maximizes long-term success\n"
            "4. Execute the chosen action via function calls"
        )

        # Game state analysis template
        analysis_template = PromptTemplate(
            name="game_state_analysis",
            template_type=TemplateType.ANALYSIS,
        )
        analysis_template.add_section(
            "current_state",
            "CURRENT GAME STATE:\n{game_state_summary}"
        )
        analysis_template.add_section(
            "detected_cards",
            "DETECTED CARDS:\n{card_list}",
            conditions=["var_not_empty:card_list"]
        )
        analysis_template.add_section(
            "card_descriptions",
            "CARD DESCRIPTIONS:\n{card_descriptions}",
            required=False,
            conditions=["var_not_empty:card_descriptions"]
        )
        analysis_template.add_section(
            "question",
            "Analyze this game state and determine what information you need or what action to take."
        )

        # Strategic planning template
        planning_template = PromptTemplate(
            name="strategic_planning",
            template_type=TemplateType.PLANNING,
        )
        planning_template.add_section(
            "context",
            "GAME CONTEXT:\n"
            "Current Score: {current_score}\n"
            "Round: {current_round}\n"
            "Blind Requirement: {blind_requirement}",
            required=False
        )
        planning_template.add_section(
            "available_cards",
            "AVAILABLE CARDS:\n{available_cards}"
        )
        planning_template.add_section(
            "objective",
            "OBJECTIVE: {objective}",
            required=False
        )
        planning_template.add_section(
            "request",
            "Create a strategic plan to achieve the best outcome. "
            "Consider poker hand rankings, special card effects, and scoring multipliers. "
            "Provide your reasoning and the specific action to take."
        )

        # Action execution template
        execution_template = PromptTemplate(
            name="action_execution",
            template_type=TemplateType.EXECUTION,
        )
        execution_template.add_section(
            "situation",
            "CURRENT SITUATION:\n{situation_summary}"
        )
        execution_template.add_section(
            "decision",
            "PLANNED ACTION: {planned_action}",
            required=False
        )
        execution_template.add_section(
            "available_actions",
            "AVAILABLE ACTIONS:\n{available_actions}"
        )
        execution_template.add_section(
            "instruction",
            "Execute the most appropriate action based on the current situation. "
            "Use function calls to perform the action."
        )

        # Register all templates
        for template in [system_template, analysis_template, planning_template, execution_template]:
            self.register_template(template)


# Global template manager instance
template_manager = PromptTemplateManager()


def render_prompt(template_name: str, context: Dict[str, Any]) -> str:
    """Convenience function to render a template."""
    return template_manager.render_template(template_name, context)


def get_system_prompt(agent_type: str = "balatro") -> str:
    """Get system prompt for agent type."""
    if agent_type == "balatro":
        return template_manager.render_template("balatro_system", {})
    return "You are a helpful AI assistant."