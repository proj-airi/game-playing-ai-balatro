"""Unit tests for AI providers."""

import pytest
from unittest.mock import Mock, patch

from src.ai_balatro.ai.providers.openrouter import OpenRouterProvider
from src.ai_balatro.ai.providers.base import ProviderConfig, ProviderType
from src.ai_balatro.ai.engines.api_provider_engine import APIProviderEngine
from src.ai_balatro.ai.llm.base import ProcessingResult


class TestOpenRouterProvider:
    """Test OpenRouterProvider class."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = OpenRouterProvider(
            model_name='anthropic/claude-3.5-sonnet', api_key='test_key', timeout=60
        )

        assert provider.config.model_name == 'anthropic/claude-3.5-sonnet'
        assert provider.config.api_key == 'test_key'
        assert provider.engine.config.timeout == 60
        assert not provider.is_initialized

    def test_initialization_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict('os.environ', {}, clear=True):
            provider = OpenRouterProvider()

            # Should log error but not raise exception
            assert provider.config.api_key is None

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'env_test_key'})
    def test_initialization_with_env_api_key(self):
        """Test initialization with environment API key."""
        provider = OpenRouterProvider()
        assert provider.config.api_key == 'env_test_key'

    def test_context_manager(self):
        """Test provider as context manager."""
        provider = OpenRouterProvider(api_key='test_key')

        # Mock the initialization and engine
        provider.engine = Mock()
        provider.engine.initialize.return_value = True
        provider._make_chat_request = Mock(
            return_value=ProcessingResult(
                success=True, data={'choices': [{'message': {'content': 'test'}}]}
            )
        )

        with provider:
            assert provider.is_initialized

        provider.engine.shutdown.assert_called_once()

    def test_available_models(self):
        """Test getting available models."""
        provider = OpenRouterProvider()
        models = provider.get_available_models()

        expected_models = [
            'anthropic/claude-3.5-sonnet',
            'openai/gpt-4o',
            'meta-llama/llama-3.1-405b-instruct',
        ]

        for model in expected_models:
            assert model in models

    def test_model_setting(self):
        """Test setting model."""
        provider = OpenRouterProvider(model_name='original_model')
        assert provider.config.model_name == 'original_model'

        # Mock successful model change
        provider.engine = Mock()
        provider.engine.initialize.return_value = True
        provider._make_chat_request = Mock(
            return_value=ProcessingResult(
                success=True, data={'choices': [{'message': {'content': 'test'}}]}
            )
        )

        result = provider.set_model('new_model')
        assert result is True
        assert provider.config.model_name == 'new_model'


class TestChatRequestHandling:
    """Test chat request handling in OpenRouterProvider."""

    def setup_method(self):
        """Set up test provider."""
        self.provider = OpenRouterProvider(model_name='test_model', api_key='test_key')
        self.provider.engine = Mock(spec=APIProviderEngine)

    def test_successful_chat_request(self):
        """Test successful chat completion request."""
        # Mock successful API response
        mock_response = {
            'choices': [
                {
                    'message': {'content': 'Test response', 'role': 'assistant'},
                    'finish_reason': 'stop',
                }
            ],
            'usage': {'total_tokens': 50},
        }

        self.provider.engine.make_request.return_value = ProcessingResult(
            success=True, data={'response': mock_response}, processing_time=0.5
        )

        result = self.provider._make_chat_request(
            messages=[{'role': 'user', 'content': 'Test'}], max_tokens=100
        )

        assert result.success
        assert result.data['content'] == 'Test response'
        assert result.data['usage']['total_tokens'] == 50

    def test_chat_request_with_function_calls(self):
        """Test chat request with function calling."""
        # Mock response with function calls
        mock_response = {
            'choices': [
                {
                    'message': {
                        'content': "I'll select a card",
                        'role': 'assistant',
                        'tool_calls': [
                            {
                                'id': 'call_123',
                                'type': 'function',
                                'function': {
                                    'name': 'select_card',
                                    'arguments': '{"index": 0}',
                                },
                            }
                        ],
                    },
                    'finish_reason': 'tool_calls',
                }
            ]
        }

        self.provider.engine.make_request.return_value = ProcessingResult(
            success=True, data={'response': mock_response}
        )

        functions = [{'name': 'select_card', 'parameters': {}}]
        result = self.provider._make_chat_request(
            messages=[{'role': 'user', 'content': 'Select a card'}],
            tools=[{'type': 'function', 'function': func} for func in functions],
        )

        assert result.success
        assert len(result.data['function_calls']) == 1
        assert result.data['function_calls'][0]['name'] == 'select_card'
        assert result.data['function_calls'][0]['arguments']['index'] == 0

    def test_malformed_function_arguments(self):
        """Test handling of malformed function arguments."""
        # Mock response with invalid JSON in function arguments
        mock_response = {
            'choices': [
                {
                    'message': {
                        'content': 'Test',
                        'tool_calls': [
                            {
                                'id': 'call_123',
                                'type': 'function',
                                'function': {
                                    'name': 'test_function',
                                    'arguments': '{"invalid": json}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        self.provider.engine.make_request.return_value = ProcessingResult(
            success=True, data={'response': mock_response}
        )

        with patch('src.ai_balatro.ai.providers.openrouter.logger') as mock_logger:
            result = self.provider._make_chat_request(
                messages=[{'role': 'user', 'content': 'Test'}]
            )

            assert result.success
            assert len(result.data['function_calls']) == 0
            mock_logger.warning.assert_called()

    def test_empty_response(self):
        """Test handling of empty response."""
        mock_response = {'choices': []}

        self.provider.engine.make_request.return_value = ProcessingResult(
            success=True, data={'response': mock_response}
        )

        result = self.provider._make_chat_request(
            messages=[{'role': 'user', 'content': 'Test'}]
        )

        assert not result.success
        assert 'No choices in response' in result.errors

    def test_engine_failure(self):
        """Test handling of engine failure."""
        self.provider.engine.make_request.return_value = ProcessingResult(
            success=False, data=None, errors=['Network error']
        )

        result = self.provider._make_chat_request(
            messages=[{'role': 'user', 'content': 'Test'}]
        )

        assert not result.success
        assert 'Network error' in result.errors


class TestProviderMethods:
    """Test high-level provider methods."""

    def setup_method(self):
        """Set up test provider."""
        self.provider = OpenRouterProvider(api_key='test_key')
        self.provider.is_initialized = True

        # Mock the internal chat request method
        self.provider._make_chat_request = Mock()

    def test_generate_text(self):
        """Test text generation."""
        self.provider._make_chat_request.return_value = ProcessingResult(
            success=True, data={'content': 'Generated text response'}
        )

        result = self.provider.generate_text('Test prompt')

        assert result.success
        assert result.data['content'] == 'Generated text response'

        # Verify correct message format was passed
        call_args = self.provider._make_chat_request.call_args
        messages = call_args[1]['messages']
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == 'Test prompt'

    def test_generate_text_with_context(self):
        """Test text generation with context."""
        context = {
            'system_message': 'You are a helpful assistant',
            'history': [
                {'role': 'user', 'content': 'Previous message'},
                {'role': 'assistant', 'content': 'Previous response'},
            ],
            'max_tokens': 500,
            'temperature': 0.8,
        }

        self.provider._make_chat_request.return_value = ProcessingResult(
            success=True, data={'content': 'Response with context'}
        )

        result = self.provider.generate_text('Current prompt', context)

        assert result.success

        # Verify context was properly included
        call_args = self.provider._make_chat_request.call_args
        messages = call_args[1]['messages']

        # Should have system message + history + current prompt
        assert any(msg['role'] == 'system' for msg in messages)
        assert any(msg['content'] == 'Previous message' for msg in messages)
        assert any(msg['content'] == 'Current prompt' for msg in messages)

    def test_function_call(self):
        """Test function calling."""
        functions = [{'name': 'test_function', 'parameters': {'type': 'object'}}]

        self.provider._make_chat_request.return_value = ProcessingResult(
            success=True,
            data={
                'content': "I'll call the function",
                'function_calls': [{'name': 'test_function', 'arguments': {}}],
            },
        )

        result = self.provider.function_call('Call a function', functions)

        assert result.success
        assert len(result.data['function_calls']) == 1

        # Verify tools were passed correctly
        call_args = self.provider._make_chat_request.call_args
        tools = call_args[1]['tools']
        assert len(tools) == 1
        assert tools[0]['function']['name'] == 'test_function'

    def test_uninitialized_provider(self):
        """Test calling methods on uninitialized provider."""
        uninitialized_provider = OpenRouterProvider()

        result = uninitialized_provider.generate_text('Test')
        assert not result.success
        assert 'not initialized' in result.errors[0].lower()

        result = uninitialized_provider.function_call('Test', [])
        assert not result.success
        assert 'not initialized' in result.errors[0].lower()

    def test_provider_shutdown(self):
        """Test provider shutdown."""
        self.provider.engine = Mock()

        self.provider.shutdown()

        assert not self.provider.is_initialized
        self.provider.engine.shutdown.assert_called_once()


class TestProviderConfig:
    """Test provider configuration."""

    def test_config_creation(self):
        """Test creating provider config."""
        config = ProviderConfig(
            provider_type=ProviderType.LLM,
            model_name='test_model',
            api_key='test_key',
            max_tokens=1000,
        )

        assert config.provider_type == ProviderType.LLM
        assert config.model_name == 'test_model'
        assert config.api_key == 'test_key'
        assert config.max_tokens == 1000
        assert config.metadata == {}

    def test_config_defaults(self):
        """Test config default values."""
        config = ProviderConfig(provider_type=ProviderType.LLM, model_name='test_model')

        assert config.api_key is None
        assert config.base_url is None
        assert config.max_tokens == 1000
        assert config.temperature == 0.7
        assert isinstance(config.metadata, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
