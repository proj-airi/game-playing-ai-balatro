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
        'description': '选择指定位置的牌，使用位置数组表示',
        'parameters': {
            'type': 'object',
            'properties': {
                'positions': {
                    'type': 'array',
                    'items': {'type': 'integer', 'enum': [-1, 0, 1]},
                    'description': '位置数组：1表示选择出牌，-1表示弃牌，0表示不操作',
                },
                'description': {'type': 'string', 'description': '操作描述'},
            },
            'required': ['positions'],
        },
    },
    {
        'name': 'hover_card',
        'description': '悬停在指定牌上查看详情',
        'parameters': {
            'type': 'object',
            'properties': {
                'card_index': {
                    'type': 'integer',
                    'description': '牌的位置索引（从0开始）',
                },
                'duration': {
                    'type': 'number',
                    'description': '悬停持续时间（秒）',
                    'default': 1.0,
                },
            },
            'required': ['card_index'],
        },
    },
    {
        'name': 'click_button',
        'description': '点击游戏界面按钮',
        'parameters': {
            'type': 'object',
            'properties': {
                'button_type': {
                    'type': 'string',
                    'enum': ['play', 'discard', 'skip', 'shop', 'next'],
                    'description': '按钮类型',
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
}
