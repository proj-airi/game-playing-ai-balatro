"""Action schemas and data structures for Balatro game actions."""

from enum import Enum
from typing import List
from dataclasses import dataclass


class ActionType(Enum):
    """Types of game actions."""

    SELECT_CARDS = 'select_cards'
    PLAY_CARDS = 'play_cards'
    DISCARD_CARDS = 'discard_cards'
    HOVER_CARD = 'hover_card'
    CLICK_BUTTON = 'click_button'


@dataclass
class CardAction:
    """Represents an action on cards using position array."""

    action_type: ActionType
    positions: List[int]  # Array like [1, 1, 1, 0] or [-1, -1, 0, 0]
    description: str = ''

    @property
    def selected_indices(self) -> List[int]:
        """Get indices of cards to select (1 values)."""
        return [i for i, val in enumerate(self.positions) if val == 1]

    @property
    def discard_indices(self) -> List[int]:
        """Get indices of cards to discard (-1 values)."""
        return [i for i, val in enumerate(self.positions) if val == -1]

    @property
    def is_play_action(self) -> bool:
        """Check if this is a play action (has positive values)."""
        return any(val > 0 for val in self.positions)

    @property
    def is_discard_action(self) -> bool:
        """Check if this is a discard action (has negative values)."""
        return any(val < 0 for val in self.positions)

    @classmethod
    def from_array(cls, positions: List[int], description: str = '') -> 'CardAction':
        """Create CardAction from position array."""
        if any(val > 0 for val in positions):
            action_type = ActionType.PLAY_CARDS
        elif any(val < 0 for val in positions):
            action_type = ActionType.DISCARD_CARDS
        else:
            action_type = ActionType.SELECT_CARDS

        return cls(
            action_type=action_type, positions=positions, description=description
        )


# Function calling schemas for LLM integration
GAME_ACTIONS = [
    {
        'name': 'select_cards_by_position',
        'description': 'Select cards by position using array notation for play/discard actions',
        'parameters': {
            'type': 'object',
            'properties': {
                'positions': {
                    'type': 'array',
                    'items': {'type': 'integer', 'enum': [-1, 0, 1]},
                    'description': 'Position array: 1=play card, -1=discard card, 0=no action',
                },
                'description': {'type': 'string', 'description': 'Action description'},
            },
            'required': ['positions'],
        },
    },
    {
        'name': 'click_button',
        'description': 'Click game interface button',
        'parameters': {
            'type': 'object',
            'properties': {
                'button_type': {
                    'type': 'string',
                    'enum': [
                        'play',
                        'discard',
                        'skip',
                        'shop',
                        'next',
                        'sort_hand_rank',
                        'sort_hand_suits',
                    ],
                    'description': 'Button type to click',
                }
            },
            'required': ['button_type'],
        },
    },
]


# Button positions and identifiers for common game buttons
BUTTON_CONFIG = {
    'play': {'keywords': ['play', '出牌', '确认'], 'description': '出牌按钮'},
    'discard': {'keywords': ['discard', '弃牌', '丢弃'], 'description': '弃牌按钮'},
    'skip': {'keywords': ['skip', '跳过', 'pass'], 'description': '跳过按钮'},
    'shop': {'keywords': ['shop', '商店', 'store'], 'description': '商店按钮'},
    'next': {
        'keywords': ['next', '下一个', 'continue'],
        'description': '继续/下一步按钮',
    },
    'button_sort_hand_rank': {
        'keywords': ['sort_hand_rank', 'sort', 'rank'],
        'description': '按牌面大小排序按钮',
    },
    'button_sort_hand_suits': {
        'keywords': ['sort_hand_suits', 'sort', 'suits'],
        'description': '按花色排序按钮',
    },
    'button_run_info': {
        'keywords': ['run_info', 'info', 'information'],
        'description': '游戏信息按钮',
    },
    'button_options': {
        'keywords': ['options', 'settings', '设置'],
        'description': '选项设置按钮',
    },
}
