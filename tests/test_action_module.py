"""Tests for the action module."""

from unittest.mock import Mock

from src.ai_balatro.ai.actions import CardAction, ActionType, ActionExecutor
from src.ai_balatro.ai.actions.card_actions import (
    CardPositionDetector,
)
from src.ai_balatro.ai.actions.schemas import GAME_ACTIONS
from src.ai_balatro.core.detection import Detection


class TestCardAction:
    """Test CardAction data structure."""

    def test_create_from_play_array(self):
        """Test creating CardAction from play array."""
        positions = [1, 1, 1, 0]
        action = CardAction.from_array(positions, '出前三张牌')

        assert action.action_type == ActionType.PLAY_CARDS
        assert action.positions == positions
        assert action.is_play_action
        assert not action.is_discard_action
        assert action.selected_indices == [0, 1, 2]
        assert action.discard_indices == []

    def test_create_from_discard_array(self):
        """Test creating CardAction from discard array."""
        positions = [-1, -1, 0, 0]
        action = CardAction.from_array(positions, '弃掉前两张牌')

        assert action.action_type == ActionType.DISCARD_CARDS
        assert action.positions == positions
        assert not action.is_play_action
        assert action.is_discard_action
        assert action.selected_indices == []
        assert action.discard_indices == [0, 1]

    def test_create_from_neutral_array(self):
        """Test creating CardAction from neutral array."""
        positions = [0, 0, 0, 0]
        action = CardAction.from_array(positions)

        assert action.action_type == ActionType.SELECT_CARDS
        assert not action.is_play_action
        assert not action.is_discard_action


class TestCardPositionDetector:
    """Test CardPositionDetector."""

    def setUp(self):
        self.detector = CardPositionDetector()

    def test_filter_playable_cards(self):
        """Test filtering playable cards from detections."""
        # Create mock detections
        detections = [
            Detection(0, 'poker_card_front', 0.9, (100, 100, 150, 200)),
            Detection(1, 'joker_card', 0.8, (200, 100, 250, 200)),
            Detection(2, 'card_description', 0.7, (50, 300, 100, 350)),  # 非可玩牌
            Detection(
                3, 'poker_card_back', 0.6, (300, 100, 350, 200)
            ),  # 背面，非可玩牌
            Detection(4, 'tarot_card', 0.85, (50, 100, 100, 200)),
        ]

        detector = CardPositionDetector()
        hand_cards = detector.get_hand_cards(detections)

        # 应该检测到3张可玩牌，按x坐标排序
        assert len(hand_cards) == 3
        assert hand_cards[0].class_name == 'tarot_card'  # x=50
        assert hand_cards[1].class_name == 'poker_card_front'  # x=100
        assert hand_cards[2].class_name == 'joker_card'  # x=200


class TestActionExecutor:
    """Test ActionExecutor."""

    def setUp(self):
        # Mock dependencies
        self.mock_yolo = Mock()
        self.mock_screen_capture = Mock()

        self.executor = ActionExecutor(self.mock_yolo, self.mock_screen_capture)
        self.executor.initialize()

    def test_initialization(self):
        """Test executor initialization."""
        executor = ActionExecutor(Mock(), Mock())
        assert executor.initialize()
        assert executor.is_initialized

    def test_get_available_actions(self):
        """Test getting available actions."""
        executor = ActionExecutor(Mock(), Mock())
        actions = executor.get_available_actions()

        assert len(actions) == len(GAME_ACTIONS)
        action_names = [action['name'] for action in actions]
        assert 'select_cards_by_position' in action_names
        assert 'hover_card' in action_names
        assert 'click_button' in action_names

    def test_process_card_positions(self):
        """Test processing card position actions."""
        executor = ActionExecutor(Mock(), Mock())
        executor.initialize()

        # Mock card engine
        executor.card_engine = Mock()
        executor.card_engine.execute_card_action.return_value = True

        # Test play action
        result = executor.process(
            {'positions': [1, 1, 0, 0], 'description': '出前两张牌'}
        )

        assert result.success
        assert result.data['action'] == 'card_positions'
        assert result.data['positions'] == [1, 1, 0, 0]
        assert result.data['executed']

        # Verify card engine was called
        executor.card_engine.execute_card_action.assert_called_with(
            [1, 1, 0, 0], '出前两张牌'
        )

    def test_process_invalid_positions(self):
        """Test processing invalid position arrays."""
        executor = ActionExecutor(Mock(), Mock())
        executor.initialize()

        # Invalid values in positions
        result = executor.process(
            {
                'positions': [2, 1, 0, -2]  # 2 and -2 are invalid
            }
        )

        assert not result.success
        assert '位置数组只能包含-1, 0, 1' in result.errors

    def test_execute_from_array_convenience_method(self):
        """Test convenience method for direct array execution."""
        executor = ActionExecutor(Mock(), Mock())
        executor.initialize()

        # Mock card engine
        executor.card_engine = Mock()
        executor.card_engine.execute_card_action.return_value = True

        # Test convenience method
        success = executor.execute_from_array([1, 1, 1, 0], '测试出牌')

        assert success
        executor.card_engine.execute_card_action.assert_called_with(
            [1, 1, 1, 0], '测试出牌'
        )


class TestGameActionSchemas:
    """Test game action schemas."""

    def test_schemas_structure(self):
        """Test that all schemas have required structure."""
        for action in GAME_ACTIONS:
            assert 'name' in action
            assert 'description' in action
            assert 'parameters' in action
            assert 'type' in action['parameters']
            assert 'properties' in action['parameters']

    def test_select_cards_schema(self):
        """Test select_cards_by_position schema."""
        action = next(
            a for a in GAME_ACTIONS if a['name'] == 'select_cards_by_position'
        )

        params = action['parameters']['properties']
        assert 'positions' in params
        assert params['positions']['type'] == 'array'
        assert params['positions']['items']['enum'] == [-1, 0, 1]


def test_integration_example():
    """Integration test example (requires manual verification)."""
    # This would be run manually with actual game instance
    #
    # # Example usage:
    # from src.ai_balatro.core.yolo_detector import YOLODetector
    # from src.ai_balatro.core.screen_capture import ScreenCapture
    #
    # # Initialize components
    # detector = YOLODetector()
    # capture = ScreenCapture()
    # executor = ActionExecutor(detector, capture)
    #
    # # Execute card actions
    # success = executor.execute_from_array([1, 1, 1, 0], "出前三张牌")
    # print(f"出牌操作: {'成功' if success else '失败'}")
    #
    # success = executor.execute_from_array([-1, -1, 0, 0], "弃掉前两张牌")
    # print(f"弃牌操作: {'成功' if success else '失败'}")

    pass  # Placeholder for integration test
