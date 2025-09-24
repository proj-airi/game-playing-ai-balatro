"""Conversation memory management for LLM agents."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import json


class MessageRole(Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class Message:
    """Single conversation message."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API calls."""
        return {
            "role": self.role.value,
            "content": self.content
        }

    def estimate_tokens(self) -> int:
        """Rough token estimation (4 chars = 1 token)."""
        if self.token_count is not None:
            return self.token_count
        return len(self.content) // 4 + 1


@dataclass
class ConversationState:
    """Current state of conversation."""
    messages: List[Message] = field(default_factory=list)
    max_messages: int = 50
    max_tokens: int = 4000
    system_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: MessageRole, content: str, **kwargs) -> Message:
        """Add message to conversation."""
        message = Message(role=role, content=content, metadata=kwargs)
        self.messages.append(message)
        self._manage_limits()
        return message

    def add_system_message(self, content: str) -> Message:
        """Add or update system message."""
        self.system_message = content
        return self.add_message(MessageRole.SYSTEM, content)

    def add_user_message(self, content: str, **kwargs) -> Message:
        """Add user message."""
        return self.add_message(MessageRole.USER, content, **kwargs)

    def add_assistant_message(self, content: str, **kwargs) -> Message:
        """Add assistant message."""
        return self.add_message(MessageRole.ASSISTANT, content, **kwargs)

    def get_total_tokens(self) -> int:
        """Get total estimated tokens."""
        return sum(msg.estimate_tokens() for msg in self.messages)

    def get_messages_for_api(self) -> List[Dict[str, Any]]:
        """Get messages formatted for API calls."""
        messages = []

        # Add system message first if exists
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })

        # Add conversation messages (skip system messages in history)
        for msg in self.messages:
            if msg.role != MessageRole.SYSTEM:
                messages.append(msg.to_dict())

        return messages

    def _manage_limits(self):
        """Manage message count and token limits."""
        # Remove oldest non-system messages if over limits
        while len(self.messages) > self.max_messages or self.get_total_tokens() > self.max_tokens:
            if len(self.messages) <= 1:  # Keep at least one message
                break

            # Find first non-system message to remove
            for i, msg in enumerate(self.messages):
                if msg.role != MessageRole.SYSTEM:
                    self.messages.pop(i)
                    break
            else:
                break  # No non-system messages to remove

    def clear_history(self, keep_system: bool = True):
        """Clear conversation history."""
        if keep_system and self.system_message:
            self.messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
        else:
            self.messages = []
            self.system_message = None

    def get_recent_context(self, max_messages: int = 10) -> str:
        """Get recent conversation context as string."""
        recent_messages = self.messages[-max_messages:] if self.messages else []
        context_lines = []

        for msg in recent_messages:
            if msg.role != MessageRole.SYSTEM:
                role_name = msg.role.value.upper()
                context_lines.append(f"{role_name}: {msg.content[:200]}...")

        return "\n".join(context_lines)

    def save_to_file(self, filepath: str):
        """Save conversation to file."""
        data = {
            "system_message": self.system_message,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata
                }
                for msg in self.messages
            ],
            "metadata": self.metadata
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'ConversationState':
        """Load conversation from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        state = cls()
        state.system_message = data.get("system_message")
        state.metadata = data.get("metadata", {})

        for msg_data in data.get("messages", []):
            message = Message(
                role=MessageRole(msg_data["role"]),
                content=msg_data["content"],
                timestamp=msg_data.get("timestamp", time.time()),
                metadata=msg_data.get("metadata", {})
            )
            state.messages.append(message)

        return state


class ConversationMemory:
    """Manages multiple conversation states."""

    def __init__(self):
        self.conversations: Dict[str, ConversationState] = {}
        self.active_conversation: Optional[str] = None

    def create_conversation(
        self,
        conversation_id: str,
        system_message: Optional[str] = None,
        max_messages: int = 50,
        max_tokens: int = 4000
    ) -> ConversationState:
        """Create new conversation."""
        state = ConversationState(
            max_messages=max_messages,
            max_tokens=max_tokens,
            system_message=system_message
        )

        if system_message:
            state.add_system_message(system_message)

        self.conversations[conversation_id] = state
        return state

    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """Get existing conversation."""
        return self.conversations.get(conversation_id)

    def set_active(self, conversation_id: str):
        """Set active conversation."""
        if conversation_id in self.conversations:
            self.active_conversation = conversation_id

    def get_active(self) -> Optional[ConversationState]:
        """Get active conversation."""
        if self.active_conversation:
            return self.conversations.get(self.active_conversation)
        return None

    def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        return list(self.conversations.keys())

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            if self.active_conversation == conversation_id:
                self.active_conversation = None
            return True
        return False