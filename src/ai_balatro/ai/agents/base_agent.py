"""Base agent framework for extensible AI agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

from ..providers.base import LLMProvider
from ..memory.conversation import ConversationMemory
from ..templates.prompt_template import PromptTemplateManager
from ...utils.logger import get_logger

logger = get_logger(__name__)


class AgentState(Enum):
    """Agent execution states."""

    IDLE = 'idle'
    THINKING = 'thinking'
    ACTING = 'acting'
    WAITING = 'waiting'
    ERROR = 'error'
    STOPPED = 'stopped'


@dataclass
class AgentContext:
    """Context passed between agent methods."""

    session_id: str
    game_state: Dict[str, Any] = field(default_factory=dict)
    previous_actions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """Result from agent execution."""

    success: bool
    action: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    state: AgentState = AgentState.IDLE
    context: Optional[AgentContext] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all AI agents."""

    def __init__(
        self,
        name: str,
        llm_provider: LLMProvider,
        conversation_memory: Optional[ConversationMemory] = None,
        template_manager: Optional[PromptTemplateManager] = None,
        max_iterations: int = 10,
    ):
        """
        Initialize base agent.

        Args:
            name: Agent identifier
            llm_provider: LLM provider for inference
            conversation_memory: Memory manager (creates new if None)
            template_manager: Template manager (creates new if None)
            max_iterations: Maximum iterations per execution
        """
        self.name = name
        self.agent_id = str(uuid.uuid4())
        self.llm_provider = llm_provider
        self.conversation_memory = conversation_memory or ConversationMemory()
        self.template_manager = template_manager or PromptTemplateManager()

        self.max_iterations = max_iterations
        self.current_state = AgentState.IDLE
        self.current_context: Optional[AgentContext] = None
        self.execution_history: List[AgentResult] = []

        # Initialize conversation
        self._init_conversation()

        logger.info(f"Agent '{name}' initialized with ID: {self.agent_id}")

    @abstractmethod
    def analyze_situation(self, context: AgentContext) -> AgentResult:
        """Analyze current situation and determine next steps."""
        pass

    @abstractmethod
    def plan_action(self, context: AgentContext) -> AgentResult:
        """Plan the next action based on analysis."""
        pass

    @abstractmethod
    def execute_action(
        self, action: Dict[str, Any], context: AgentContext
    ) -> AgentResult:
        """Execute the planned action."""
        pass

    def run(self, initial_context: Optional[AgentContext] = None) -> AgentResult:
        """Run the agent's main execution loop."""
        self.current_state = AgentState.THINKING

        # Initialize context
        if initial_context is None:
            initial_context = AgentContext(session_id=str(uuid.uuid4()))

        self.current_context = initial_context
        iteration = 0

        try:
            while iteration < self.max_iterations:
                logger.info(f'Agent {self.name} - Iteration {iteration + 1}')

                # Analyze situation
                analysis_result = self.analyze_situation(self.current_context)
                self.execution_history.append(analysis_result)

                if not analysis_result.success:
                    self.current_state = AgentState.ERROR
                    return analysis_result

                # Plan action
                self.current_state = AgentState.THINKING
                planning_result = self.plan_action(self.current_context)
                self.execution_history.append(planning_result)

                if not planning_result.success:
                    self.current_state = AgentState.ERROR
                    return planning_result

                # Execute action if planned
                if planning_result.action:
                    self.current_state = AgentState.ACTING
                    execution_result = self.execute_action(
                        planning_result.action, self.current_context
                    )
                    self.execution_history.append(execution_result)

                    # Update context with action
                    if self.current_context:
                        self.current_context.previous_actions.append(
                            planning_result.action
                        )

                    if not execution_result.success:
                        self.current_state = AgentState.ERROR
                        return execution_result

                    # Check if we should continue
                    if self._should_stop(execution_result):
                        self.current_state = AgentState.IDLE
                        return execution_result

                iteration += 1

            # Max iterations reached
            self.current_state = AgentState.IDLE
            return AgentResult(
                success=True,
                reasoning='Maximum iterations reached',
                state=self.current_state,
                context=self.current_context,
                metadata={'iterations': iteration},
            )

        except Exception as e:
            self.current_state = AgentState.ERROR
            logger.error(f'Agent {self.name} execution failed: {e}')

            return AgentResult(
                success=False,
                state=self.current_state,
                context=self.current_context,
                errors=[f'Execution failed: {e}'],
            )

    def stop(self):
        """Stop agent execution."""
        self.current_state = AgentState.STOPPED
        logger.info(f'Agent {self.name} stopped')

    def reset(self):
        """Reset agent state."""
        self.current_state = AgentState.IDLE
        self.current_context = None
        self.execution_history = []
        logger.info(f'Agent {self.name} reset')

    def get_conversation_id(self) -> str:
        """Get conversation ID for this agent session."""
        return f'{self.name}_{self.agent_id}'

    def _init_conversation(self):
        """Initialize conversation memory."""
        conversation_id = self.get_conversation_id()
        system_message = self._get_system_message()

        self.conversation_memory.create_conversation(
            conversation_id=conversation_id, system_message=system_message
        )
        self.conversation_memory.set_active(conversation_id)

    def _get_system_message(self) -> str:
        """Get system message for this agent."""
        # Override in subclasses for specific system messages
        return f'You are {self.name}, an AI agent designed to analyze situations and take actions.'

    def _should_stop(self, result: AgentResult) -> bool:
        """Determine if agent should stop execution."""
        # Override in subclasses for specific stop conditions
        return result.metadata.get('should_stop', False)

    def _llm_query(
        self,
        prompt: str,
        use_functions: bool = False,
        functions: Optional[List[Dict]] = None,
    ) -> AgentResult:
        """Query LLM with conversation context."""
        try:
            conversation = self.conversation_memory.get_active()
            if not conversation:
                return AgentResult(
                    success=False, errors=['No active conversation found']
                )

            # Add user message
            conversation.add_user_message(prompt)

            # Prepare context
            context = {
                'history': conversation.get_messages_for_api()[
                    :-1
                ],  # Exclude current message
                'max_tokens': 1000,
                'temperature': 0.3,
            }

            if use_functions and functions:
                logger.info(f"   Function Count: {len(functions)}")
                logger.info(f"   Available Functions: {[f.get('name', 'unknown') for f in functions]}")

            # Log conversation history length
            history_length = len(context['history'])
            logger.info(f"   Conversation History: {history_length} messages")

            # Log prompt preview (truncated for readability)
            prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            logger.info(f"   Prompt Preview: {prompt_preview}")

            # Log full prompt for detailed debugging (can be disabled if too verbose)
            if logger.isEnabledFor(10):  # DEBUG level
                logger.debug(f"   Full Prompt:\n{prompt}")
                if history_length > 0:
                    logger.debug(f"   Conversation History:\n{context['history']}")

            # Query LLM
            if use_functions and functions:
                logger.info("   Calling LLM with function calling...")
                result = self.llm_provider.function_call(prompt, functions, context)
            else:
                logger.info("   Calling LLM for text generation...")
                result = self.llm_provider.generate_text(prompt, context)

            # Log response details
            logger.info(f"   LLM Response Success: {result.success}")
            if result.success:
                response_data = result.data
                if response_data:
                    logger.info(f"   Response Content Length: {len(response_data.get('content', ''))}")
                    if 'function_calls' in response_data and response_data['function_calls']:
                        logger.info(f"   Function Calls: {len(response_data['function_calls'])}")
                        for i, fc in enumerate(response_data['function_calls']):
                            logger.info(f"     {i+1}. {fc.get('name', 'unknown')}({fc.get('arguments', {})})")
                    if 'usage' in response_data:
                        usage = response_data['usage']
                        logger.info(f"   Token Usage: {usage}")
            else:
                logger.error(f"   LLM Error: {result.errors}")

            if result.success:
                # Add assistant response
                content = result.data.get('content', '')
                if content:
                    conversation.add_assistant_message(content)

                return AgentResult(
                    success=True,
                    action=result.data.get('function_calls', [{}])[0]
                    if use_functions
                    else None,
                    reasoning=content,
                    metadata=result.metadata,
                )
            else:
                return AgentResult(
                    success=False, errors=result.errors, metadata=result.metadata
                )

        except Exception as e:
            logger.error(f'LLM query failed: {e}')
            return AgentResult(
                success=False,
                errors=[f'LLM query failed: {e}'],
                context=context
            )

    def _render_prompt(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render prompt using template."""
        try:
            return self.template_manager.render_template(template_name, context)
        except Exception as e:
            logger.error(f'Template rendering failed: {e}')
            return f"Error rendering template '{template_name}': {e}"


class AgentOrchestrator:
    """Orchestrates multiple agents working together."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_order: List[str] = []
        self.shared_context: AgentContext = AgentContext(session_id=str(uuid.uuid4()))

    def register_agent(self, agent: BaseAgent, execution_order: Optional[int] = None):
        """Register an agent."""
        self.agents[agent.name] = agent

        if execution_order is not None:
            # Insert at specific position
            if execution_order >= len(self.execution_order):
                self.execution_order.append(agent.name)
            else:
                self.execution_order.insert(execution_order, agent.name)
        else:
            # Add to end
            self.execution_order.append(agent.name)

        logger.info(f'Registered agent: {agent.name}')

    def run_agents(self, context: Optional[AgentContext] = None) -> List[AgentResult]:
        """Run all agents in execution order."""
        if context:
            self.shared_context = context

        results = []

        for agent_name in self.execution_order:
            agent = self.agents.get(agent_name)
            if not agent:
                continue

            logger.info(f'Running agent: {agent_name}')

            # Update context with previous results
            self.shared_context.metadata['previous_results'] = results

            result = agent.run(self.shared_context)
            results.append(result)

            # Update shared context
            if result.context:
                self.shared_context.game_state.update(result.context.game_state)
                self.shared_context.previous_actions.extend(
                    result.context.previous_actions
                )

            # Stop on error if configured
            if not result.success:
                logger.error(f'Agent {agent_name} failed: {result.errors}')
                break

        return results

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """List registered agent names."""
        return list(self.agents.keys())
