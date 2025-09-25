"""OpenRouter provider using APIProviderEngine."""

import os
import json
from typing import Dict, List, Optional

from .base import LLMProvider, ProviderConfig, ProviderType
from ..engines.api_provider_engine import APIProviderEngine
from ..llm.base import ProcessingResult
from ...utils.logger import get_logger

logger = get_logger(__name__)


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider."""

    def __init__(
        self,
        model_name: str = 'anthropic/claude-3.5-sonnet',
        api_key: Optional[str] = None,
        base_url: str = 'https://openrouter.ai/api/v1',
        timeout: int = 30,
        max_retries: int = 3,
        **kwargs,
    ):
        """
        Initialize OpenRouter provider.

        Args:
            model_name: Model identifier (e.g., "anthropic/claude-3.5-sonnet")
            api_key: OpenRouter API key (from env if None)
            base_url: OpenRouter API base URL
            timeout: Request timeout
            max_retries: Max retry attempts
        """
        config = ProviderConfig(
            provider_type=ProviderType.LLM,
            model_name=model_name,
            api_key=api_key or os.getenv('OPENROUTER_API_KEY'),
            base_url=base_url,
            **kwargs,
        )

        engine = APIProviderEngine(
            name='OpenRouter', timeout=timeout, max_retries=max_retries
        )

        super().__init__('OpenRouter', config, engine)

        if not self.config.api_key:
            logger.error('OPENROUTER_API_KEY not found in environment variables')

    def initialize(self) -> bool:
        """Initialize the provider and engine."""
        try:
            if not self.config.api_key:
                logger.error('Cannot initialize: OPENROUTER_API_KEY not provided')
                return False

            # Initialize the engine
            if not self.engine.initialize():
                logger.error('Failed to initialize APIProviderEngine')
                return False

            # Test connection
            test_result = self._make_chat_request(
                messages=[{'role': 'user', 'content': 'Hello'}], max_tokens=10
            )

            if test_result.success:
                logger.info(
                    f'OpenRouter provider initialized with {self.config.model_name}'
                )
                self.is_initialized = True
                return True
            else:
                logger.error(f'OpenRouter test call failed: {test_result.errors}')
                return False

        except Exception as e:
            logger.error(f'Failed to initialize OpenRouter provider: {e}')
            return False

    def generate_text(
        self, prompt: str, context: Optional[Dict] = None
    ) -> ProcessingResult:
        """Generate text response."""
        if not self.is_initialized:
            return ProcessingResult(
                success=False, data=None, errors=['Provider not initialized']
            )

        try:
            # Prepare messages
            messages = [{'role': 'user', 'content': prompt}]

            # Add system message if provided
            if context and 'system_message' in context:
                messages.insert(
                    0, {'role': 'system', 'content': context['system_message']}
                )

            # Add conversation history if provided
            if context and 'history' in context:
                for msg in context['history']:
                    messages.append(msg)

            # Get parameters
            max_tokens = (
                context.get('max_tokens', self.config.max_tokens)
                if context
                else self.config.max_tokens
            )
            temperature = (
                context.get('temperature', self.config.temperature)
                if context
                else self.config.temperature
            )

            return self._make_chat_request(
                messages=messages, max_tokens=max_tokens, temperature=temperature
            )

        except Exception as e:
            logger.error(f'Text generation failed: {e}')
            return ProcessingResult(
                success=False, data=None, errors=[f'Generation failed: {e}']
            )

    def function_call(
        self, prompt: str, functions: List[Dict], context: Optional[Dict] = None
    ) -> ProcessingResult:
        """Generate function calls."""
        if not self.is_initialized:
            return ProcessingResult(
                success=False, data=None, errors=['Provider not initialized']
            )

        try:
            # Prepare messages
            messages = [{'role': 'user', 'content': prompt}]

            # Add system message if provided
            if context and 'system_message' in context:
                messages.insert(
                    0, {'role': 'system', 'content': context['system_message']}
                )

            # Add conversation history if provided
            if context and 'history' in context:
                for msg in context['history']:
                    messages.append(msg)

            # Get parameters
            max_tokens = (
                context.get('max_tokens', self.config.max_tokens)
                if context
                else self.config.max_tokens
            )
            temperature = (
                context.get('temperature', self.config.temperature)
                if context
                else self.config.temperature
            )

            return self._make_chat_request(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=[{'type': 'function', 'function': func} for func in functions],
                tool_choice='auto',
            )

        except Exception as e:
            logger.error(f'Function call failed: {e}')
            return ProcessingResult(
                success=False, data=None, errors=[f'Function call failed: {e}']
            )

    def _make_chat_request(
        self,
        messages: List[Dict],
        max_tokens: int = None,
        temperature: float = None,
        tools: List[Dict] = None,
        tool_choice: str = None,
    ) -> ProcessingResult:
        """Make chat completion request to OpenRouter."""
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json',
        }

        payload = {
            'model': self.config.model_name,
            'messages': messages,
            'max_tokens': max_tokens or self.config.max_tokens,
            'temperature': temperature or self.config.temperature,
        }

        if tools:
            payload['tools'] = tools
            if tool_choice:
                payload['tool_choice'] = tool_choice

        url = f'{self.config.base_url}/chat/completions'

        result = self.engine.make_request(
            method='POST', url=url, headers=headers, json_data=payload
        )

        if not result.success:
            return result

        try:
            response_data = result.data['response']

            if 'choices' not in response_data or not response_data['choices']:
                return ProcessingResult(
                    success=False,
                    data=None,
                    errors=['No choices in response'],
                    processing_time=result.processing_time,
                )

            choice = response_data['choices'][0]
            message = choice['message']

            # Parse function calls if present
            function_calls = []
            if 'tool_calls' in message and message['tool_calls']:
                for tool_call in message['tool_calls']:
                    if tool_call['type'] == 'function':
                        try:
                            arguments = json.loads(tool_call['function']['arguments'])
                            function_calls.append(
                                {
                                    'id': tool_call['id'],
                                    'name': tool_call['function']['name'],
                                    'arguments': arguments,
                                }
                            )
                        except json.JSONDecodeError as e:
                            logger.warning(f'Failed to parse function arguments: {e}')

            return ProcessingResult(
                success=True,
                data={
                    'content': message.get('content', ''),
                    'function_calls': function_calls,
                    'model': self.config.model_name,
                    'usage': response_data.get('usage', {}),
                    'finish_reason': choice.get('finish_reason', ''),
                },
                processing_time=result.processing_time,
                metadata=result.metadata,
            )

        except Exception as e:
            logger.error(f'Failed to parse OpenRouter response: {e}')
            return ProcessingResult(
                success=False,
                data=None,
                errors=[f'Response parsing failed: {e}'],
                processing_time=result.processing_time,
            )

    def get_available_models(self) -> List[str]:
        """Get list of available models from OpenRouter."""
        return [
            'anthropic/claude-3.5-sonnet',
            'anthropic/claude-3-sonnet',
            'anthropic/claude-3-haiku',
            'openai/gpt-4o',
            'openai/gpt-4',
            'openai/gpt-3.5-turbo',
            'meta-llama/llama-3.1-405b-instruct',
            'meta-llama/llama-3.1-70b-instruct',
            'google/gemini-pro',
            'mistralai/mistral-7b-instruct',
        ]

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.engine:
            self.engine.shutdown()
        self.is_initialized = False
