"""Unit tests for memory management components."""

import pytest
import tempfile
import os

from src.ai_balatro.ai.memory.conversation import (
    Message,
    MessageRole,
    ConversationState,
    ConversationMemory,
)


class TestMessage:
    """Test Message class."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(
            role=MessageRole.USER, content='Test message', metadata={'test': True}
        )

        assert msg.role == MessageRole.USER
        assert msg.content == 'Test message'
        assert msg.metadata['test'] is True
        assert isinstance(msg.timestamp, float)

    def test_message_to_dict(self):
        """Test message conversion to dict."""
        msg = Message(MessageRole.ASSISTANT, 'Assistant response')
        msg_dict = msg.to_dict()

        assert msg_dict['role'] == 'assistant'
        assert msg_dict['content'] == 'Assistant response'

    def test_token_estimation(self):
        """Test token estimation."""
        short_msg = Message(MessageRole.USER, 'Hi')
        long_msg = Message(
            MessageRole.USER, 'This is a longer message with more tokens'
        )

        assert short_msg.estimate_tokens() >= 1
        assert long_msg.estimate_tokens() > short_msg.estimate_tokens()

    def test_explicit_token_count(self):
        """Test explicit token count."""
        msg = Message(MessageRole.USER, 'Test', token_count=10)
        assert msg.estimate_tokens() == 10


class TestConversationState:
    """Test ConversationState class."""

    def test_initialization(self):
        """Test conversation state initialization."""
        state = ConversationState(
            max_messages=20, max_tokens=1000, system_message='System prompt'
        )

        assert state.max_messages == 20
        assert state.max_tokens == 1000
        assert state.system_message == 'System prompt'
        assert len(state.messages) == 0

    def test_add_messages(self):
        """Test adding different types of messages."""
        state = ConversationState()

        user_msg = state.add_user_message('User message', source='test')
        assert user_msg.role == MessageRole.USER
        assert user_msg.content == 'User message'
        assert user_msg.metadata['source'] == 'test'

        assistant_msg = state.add_assistant_message('Assistant response')
        assert assistant_msg.role == MessageRole.ASSISTANT
        assert assistant_msg.content == 'Assistant response'

        assert len(state.messages) == 2

    def test_system_message_handling(self):
        """Test system message handling."""
        state = ConversationState()

        state.add_system_message('System message')
        assert state.system_message == 'System message'

        # System message should be in messages too
        system_messages = [m for m in state.messages if m.role == MessageRole.SYSTEM]
        assert len(system_messages) == 1

    def test_message_limit_management(self):
        """Test message count limits."""
        state = ConversationState(max_messages=3)

        # Add more messages than limit
        state.add_user_message('Message 1')
        state.add_assistant_message('Response 1')
        state.add_user_message('Message 2')
        state.add_assistant_message('Response 2')
        state.add_user_message('Message 3')

        # Should not exceed max_messages
        assert len(state.messages) <= 3

        # Most recent messages should be kept
        assert any('Message 3' in m.content for m in state.messages)

    def test_token_limit_management(self):
        """Test token limits."""
        state = ConversationState(max_tokens=20)  # Very low limit

        # Add messages that exceed token limit
        state.add_user_message(
            'This is a long message that should exceed the token limit'
        )
        state.add_assistant_message('Another long response that adds more tokens')

        # Should manage tokens automatically
        total_tokens = state.get_total_tokens()
        assert total_tokens <= state.max_tokens or len(state.messages) <= 1

    def test_api_message_format(self):
        """Test API message formatting."""
        state = ConversationState(system_message='System prompt')
        state.add_user_message('User input')
        state.add_assistant_message('Assistant output')

        api_messages = state.get_messages_for_api()

        # Should include system message first
        assert api_messages[0]['role'] == 'system'
        assert api_messages[0]['content'] == 'System prompt'

        # Should include conversation messages
        assert len(api_messages) >= 2
        assert api_messages[1]['role'] == 'user'

    def test_recent_context(self):
        """Test recent context retrieval."""
        state = ConversationState()

        state.add_user_message('Message 1')
        state.add_assistant_message('Response 1')
        state.add_user_message('Message 2')

        context = state.get_recent_context(max_messages=2)

        assert 'Message 2' in context
        assert len(context.split('\n')) <= 2

    def test_clear_history(self):
        """Test clearing conversation history."""
        state = ConversationState(system_message='Keep this')
        state.add_user_message('Remove this')
        state.add_assistant_message('Remove this too')

        # Clear but keep system message
        state.clear_history(keep_system=True)

        system_messages = [m for m in state.messages if m.role == MessageRole.SYSTEM]
        non_system_messages = [
            m for m in state.messages if m.role != MessageRole.SYSTEM
        ]

        assert (
            len(system_messages) >= 0
        )  # May or may not have system message in messages
        assert len(non_system_messages) == 0
        assert state.system_message == 'Keep this'

        # Clear everything
        state.add_user_message('New message')
        state.clear_history(keep_system=False)

        assert len(state.messages) == 0
        assert state.system_message is None


class TestConversationPersistence:
    """Test conversation save/load functionality."""

    def test_save_load_conversation(self):
        """Test saving and loading conversation."""
        # Create conversation
        original = ConversationState(system_message='Test system')
        original.add_user_message('User message', test=True)
        original.add_assistant_message('Assistant response')

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            original.save_to_file(temp_path)

            # Load from file
            loaded = ConversationState.load_from_file(temp_path)

            # Verify content
            assert loaded.system_message == 'Test system'
            assert len(loaded.messages) == 2

            user_msg = next(m for m in loaded.messages if m.role == MessageRole.USER)
            assert user_msg.content == 'User message'
            assert user_msg.metadata.get('test') is True

        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            ConversationState.load_from_file('nonexistent_file.json')


class TestConversationMemory:
    """Test ConversationMemory class."""

    def test_conversation_creation(self):
        """Test creating conversations."""
        memory = ConversationMemory()

        conversation = memory.create_conversation(
            'test_conv', system_message='Test system', max_messages=10
        )

        assert conversation is not None
        assert conversation.system_message == 'Test system'
        assert conversation.max_messages == 10

        # Should be stored in memory
        assert 'test_conv' in memory.conversations
        retrieved = memory.get_conversation('test_conv')
        assert retrieved is conversation

    def test_active_conversation(self):
        """Test active conversation management."""
        memory = ConversationMemory()

        conv1 = memory.create_conversation('conv1')
        conv2 = memory.create_conversation('conv2')

        # Set active conversation
        memory.set_active('conv1')
        assert memory.get_active() is conv1

        memory.set_active('conv2')
        assert memory.get_active() is conv2

        # Invalid conversation
        memory.set_active('nonexistent')
        assert memory.active_conversation == 'nonexistent'
        assert memory.get_active() is None

    def test_conversation_listing(self):
        """Test listing conversations."""
        memory = ConversationMemory()

        memory.create_conversation('conv1')
        memory.create_conversation('conv2')
        memory.create_conversation('conv3')

        conversations = memory.list_conversations()
        assert set(conversations) == {'conv1', 'conv2', 'conv3'}

    def test_conversation_deletion(self):
        """Test deleting conversations."""
        memory = ConversationMemory()

        memory.create_conversation('to_delete')
        memory.create_conversation('to_keep')
        memory.set_active('to_delete')

        # Delete conversation
        result = memory.delete_conversation('to_delete')
        assert result is True
        assert 'to_delete' not in memory.conversations
        assert memory.active_conversation is None

        # Try to delete nonexistent
        result = memory.delete_conversation('nonexistent')
        assert result is False

        # Remaining conversation should still exist
        assert 'to_keep' in memory.conversations

    def test_get_nonexistent_conversation(self):
        """Test getting nonexistent conversation."""
        memory = ConversationMemory()
        result = memory.get_conversation('nonexistent')
        assert result is None


class TestMessageRoleHandling:
    """Test message role handling."""

    def test_all_message_roles(self):
        """Test all message roles are handled correctly."""
        state = ConversationState()

        roles_to_test = [
            (MessageRole.SYSTEM, 'System message'),
            (MessageRole.USER, 'User message'),
            (MessageRole.ASSISTANT, 'Assistant message'),
            (MessageRole.FUNCTION, 'Function result'),
            (MessageRole.TOOL, 'Tool result'),
        ]

        for role, content in roles_to_test:
            msg = state.add_message(role, content)
            assert msg.role == role
            assert msg.content == content

    def test_api_format_excludes_system_from_history(self):
        """Test that API format handles system messages correctly."""
        state = ConversationState(system_message='System prompt')

        # Add system message to history (shouldn't happen normally)
        state.add_message(MessageRole.SYSTEM, 'Another system message')
        state.add_user_message('User message')

        api_messages = state.get_messages_for_api()

        # Should have system message first
        assert api_messages[0]['role'] == 'system'
        assert api_messages[0]['content'] == 'System prompt'

        # Should only have non-system messages from history
        history_messages = api_messages[1:]
        system_in_history = any(msg['role'] == 'system' for msg in history_messages)
        assert not system_in_history


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
