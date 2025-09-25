"""Integration tests for the agent framework."""

import pytest
import os
from unittest.mock import Mock
from typing import Dict

from src.ai_balatro.ai.providers import OpenRouterProvider
from src.ai_balatro.ai.memory import ConversationMemory, MessageRole
from src.ai_balatro.ai.templates import render_prompt
from src.ai_balatro.ai.agents import (
    BaseAgent,
    AgentContext,
    AgentResult,
    AgentOrchestrator,
)
from src.ai_balatro.ai.actions import GAME_ACTIONS


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses=None, function_calls=None):
        self.responses = responses or ['Mock response']
        self.function_calls = function_calls or []
        self.response_index = 0
        self.is_initialized = False

    def initialize(self) -> bool:
        self.is_initialized = True
        return True

    def generate_text(self, prompt: str, context: Dict = None):
        from src.ai_balatro.ai.llm.base import ProcessingResult

        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
        else:
            response = 'Default mock response'

        return ProcessingResult(
            success=True, data={'content': response}, metadata={'model': 'mock'}
        )

    def function_call(self, prompt: str, functions: list, context: Dict = None):
        from src.ai_balatro.ai.llm.base import ProcessingResult

        if self.function_calls:
            function_call = self.function_calls[0] if self.function_calls else {}
        else:
            function_call = {'name': 'select_card', 'arguments': {'index': 0}}

        return ProcessingResult(
            success=True,
            data={
                'content': "I'll select the first card",
                'function_calls': [function_call],
            },
            metadata={'model': 'mock'},
        )

    def shutdown(self):
        self.is_initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class TestAgent(BaseAgent):
    """Test agent implementation."""

    def __init__(self, llm_provider, test_responses=None):
        super().__init__('TestAgent', llm_provider)
        self.test_responses = test_responses or []
        self.analysis_calls = []
        self.planning_calls = []
        self.execution_calls = []

    def analyze_situation(self, context: AgentContext) -> AgentResult:
        """Mock analysis."""
        self.analysis_calls.append(context)

        prompt = render_prompt(
            'game_state_analysis',
            {
                'game_state_summary': 'Test game state',
                'card_list': ['Test card 1', 'Test card 2'],
            },
        )

        result = self._llm_query(prompt)
        return AgentResult(
            success=result.success,
            reasoning=result.reasoning or 'Analyzed test situation',
            context=context,
            errors=result.errors,
        )

    def plan_action(self, context: AgentContext) -> AgentResult:
        """Mock planning."""
        self.planning_calls.append(context)

        prompt = render_prompt(
            'strategic_planning',
            {'available_cards': ['Card 1', 'Card 2'], 'objective': 'Test objective'},
        )

        result = self._llm_query(prompt, use_functions=True, functions=GAME_ACTIONS)
        return AgentResult(
            success=result.success,
            action=result.action,
            reasoning=result.reasoning or 'Planned test action',
            context=context,
            errors=result.errors,
        )

    def execute_action(self, action: dict, context: AgentContext) -> AgentResult:
        """Mock execution."""
        self.execution_calls.append((action, context))

        return AgentResult(
            success=True,
            action=action,
            reasoning=f'Executed {action.get("name", "unknown")} action',
            context=context,
            metadata={'should_stop': True},  # Stop after first execution
        )


@pytest.fixture
def mock_provider():
    """Mock LLM provider fixture."""
    return MockLLMProvider(
        responses=[
            'The game state shows 2 cards available for selection.',
            'I should select the first card as it has the highest value.',
            'Action executed successfully.',
        ],
        function_calls=[{'name': 'select_card', 'arguments': {'index': 0}}],
    )


@pytest.fixture
def test_agent(mock_provider):
    """Test agent fixture."""
    return TestAgent(mock_provider)


@pytest.fixture
def sample_context():
    """Sample agent context."""
    return AgentContext(
        session_id='test_session',
        game_state={'cards_visible': 2, 'score': 1000, 'round': 1},
        metadata={'test': True},
    )


class TestConversationMemory:
    """Test conversation memory functionality."""

    def test_create_conversation(self):
        """Test creating a new conversation."""
        memory = ConversationMemory()
        conversation = memory.create_conversation(
            'test_conv',
            system_message='Test system message',
            max_messages=10,
            max_tokens=500,
        )

        assert conversation is not None
        assert conversation.system_message == 'Test system message'
        assert conversation.max_messages == 10
        assert conversation.max_tokens == 500

    def test_message_management(self):
        """Test message addition and limits."""
        memory = ConversationMemory()
        conversation = memory.create_conversation('test', max_messages=3)

        # Add messages
        conversation.add_user_message('Message 1')
        conversation.add_assistant_message('Response 1')
        conversation.add_user_message('Message 2')
        conversation.add_assistant_message('Response 2')

        # Should have max 3 messages (excluding system)
        non_system_messages = [
            m for m in conversation.messages if m.role != MessageRole.SYSTEM
        ]
        assert len(non_system_messages) <= 3

    def test_api_message_format(self):
        """Test API message formatting."""
        memory = ConversationMemory()
        conversation = memory.create_conversation('test', system_message='System msg')

        conversation.add_user_message('User message')
        conversation.add_assistant_message('Assistant response')

        api_messages = conversation.get_messages_for_api()

        assert len(api_messages) >= 2
        assert api_messages[0]['role'] == 'system'
        assert api_messages[1]['role'] == 'user'


class TestPromptTemplates:
    """Test prompt template functionality."""

    def test_template_rendering(self):
        """Test basic template rendering."""
        context = {
            'game_state_summary': 'Test state',
            'card_list': ['Card A', 'Card B'],
        }

        prompt = render_prompt('game_state_analysis', context)

        assert 'Test state' in prompt
        assert 'Card A' in prompt
        assert 'Card B' in prompt

    def test_conditional_sections(self):
        """Test conditional section inclusion."""
        # Context with card descriptions
        context_with_desc = {
            'game_state_summary': 'Test state',
            'card_list': ['Card A'],
            'card_descriptions': {'0': 'Description A'},
        }

        prompt_with_desc = render_prompt('game_state_analysis', context_with_desc)
        assert 'CARD DESCRIPTIONS' in prompt_with_desc

        # Context without card descriptions
        context_no_desc = {'game_state_summary': 'Test state', 'card_list': ['Card A']}

        prompt_no_desc = render_prompt('game_state_analysis', context_no_desc)
        assert 'CARD DESCRIPTIONS' not in prompt_no_desc

    def test_variable_substitution(self):
        """Test variable substitution in templates."""
        context = {
            'available_cards': ['Ace', 'King', 'Queen'],
            'objective': 'Win the round',
        }

        prompt = render_prompt('strategic_planning', context)

        assert 'Ace' in prompt
        assert 'King' in prompt
        assert 'Queen' in prompt
        assert 'Win the round' in prompt


class TestBaseAgent:
    """Test base agent functionality."""

    def test_agent_initialization(self, mock_provider):
        """Test agent initialization."""
        agent = TestAgent(mock_provider)

        assert agent.name == 'TestAgent'
        assert agent.llm_provider == mock_provider
        assert agent.conversation_memory is not None
        assert agent.current_state.value == 'idle'

    def test_conversation_setup(self, mock_provider):
        """Test conversation initialization."""
        agent = TestAgent(mock_provider)
        conversation_id = agent.get_conversation_id()

        assert conversation_id.startswith('TestAgent_')

        conversation = agent.conversation_memory.get_conversation(conversation_id)
        assert conversation is not None

    def test_llm_query(self, test_agent, mock_provider):
        """Test LLM querying functionality."""
        with mock_provider:
            result = test_agent._llm_query('Test prompt')

            assert result.success
            assert result.reasoning is not None

            # Check conversation was updated
            conversation = test_agent.conversation_memory.get_active()
            assert len(conversation.messages) >= 2  # User + Assistant

    def test_agent_execution_flow(self, test_agent, mock_provider, sample_context):
        """Test complete agent execution flow."""
        with mock_provider:
            result = test_agent.run(sample_context)

            assert result.success
            assert len(test_agent.analysis_calls) == 1
            assert len(test_agent.planning_calls) == 1
            assert len(test_agent.execution_calls) == 1

    def test_agent_error_handling(self, mock_provider):
        """Test agent error handling."""
        # Create provider that fails
        failing_provider = MockLLMProvider()
        failing_provider.generate_text = Mock(side_effect=Exception('Test error'))

        agent = TestAgent(failing_provider)

        with failing_provider:
            result = agent.run(AgentContext(session_id='error_test'))

            assert not result.success
            assert 'Test error' in str(result.errors)


class TestAgentOrchestrator:
    """Test agent orchestration."""

    def test_agent_registration(self, mock_provider):
        """Test agent registration."""
        orchestrator = AgentOrchestrator()
        agent1 = TestAgent(mock_provider)
        agent2 = TestAgent(mock_provider)
        agent2.name = 'TestAgent2'

        orchestrator.register_agent(agent1, 0)
        orchestrator.register_agent(agent2, 1)

        assert len(orchestrator.agents) == 2
        assert orchestrator.execution_order == ['TestAgent', 'TestAgent2']

    def test_multi_agent_execution(self, mock_provider, sample_context):
        """Test multi-agent execution."""
        orchestrator = AgentOrchestrator()

        agent1 = TestAgent(mock_provider)
        agent1.name = 'Analyzer'
        agent2 = TestAgent(mock_provider)
        agent2.name = 'Planner'

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        with mock_provider:
            results = orchestrator.run_agents(sample_context)

            assert len(results) == 2
            assert all(result.success for result in results)

    def test_shared_context_updates(self, mock_provider, sample_context):
        """Test shared context updates between agents."""
        orchestrator = AgentOrchestrator()

        agent1 = TestAgent(mock_provider)
        agent1.name = 'First'
        agent2 = TestAgent(mock_provider)
        agent2.name = 'Second'

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        with mock_provider:
            results = orchestrator.run_agents(sample_context)

            # Second agent should have access to first agent's results
            shared_context = orchestrator.shared_context
            assert 'previous_results' in shared_context.metadata
            assert len(shared_context.metadata['previous_results']) > 0


@pytest.mark.integration
class TestRealProviderIntegration:
    """Integration tests with real LLM provider (requires API key)."""

    @pytest.mark.skipif(
        not os.getenv('OPENROUTER_API_KEY'), reason='OPENROUTER_API_KEY not set'
    )
    def test_openrouter_provider_integration(self):
        """Test real OpenRouter provider."""
        provider = OpenRouterProvider(model_name='anthropic/claude-3.5-sonnet')

        with provider:
            result = provider.generate_text(
                "Hello, respond with 'Integration test successful'"
            )

            assert result.success
            assert result.data['content'] is not None

    @pytest.mark.skipif(
        not os.getenv('OPENROUTER_API_KEY'), reason='OPENROUTER_API_KEY not set'
    )
    def test_real_agent_execution(self, sample_context):
        """Test agent with real LLM provider."""
        provider = OpenRouterProvider(model_name='anthropic/claude-3.5-sonnet')
        agent = TestAgent(provider)

        with provider:
            result = agent.run(sample_context)

            assert result.success
            assert result.reasoning is not None

            # Check that conversation has meaningful content
            conversation = agent.conversation_memory.get_active()
            messages = conversation.get_messages_for_api()
            assert len(messages) >= 3  # System + User + Assistant


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
