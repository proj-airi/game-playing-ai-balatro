"""
游戏状态分析模块
解析YOLO检测结果，构建小丑牌游戏的当前状态

Author: RainbowBird
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from yolo_detector import Detection


class CardType(Enum):
    """卡牌类型枚举"""
    POKER_CARD_FRONT = "poker_card_front"      # 正面扑克牌
    POKER_CARD_BACK = "poker_card_back"        # 背面扑克牌
    POKER_CARD_STACK = "poker_card_stack"      # 扑克牌堆
    JOKER_CARD = "joker_card"                  # 小丑牌
    TAROT_CARD = "tarot_card"                  # 塔罗牌
    PLANET_CARD = "planet_card"                # 行星牌
    SPECTRAL_CARD = "spectral_card"            # 幽灵牌
    CARD_PACK = "card_pack"                    # 卡包
    CARD_DESCRIPTION = "card_description"      # 卡牌描述
    POKER_CARD_DESCRIPTION = "poker_card_description"  # 扑克牌描述


@dataclass
class Card:
    """卡牌数据类"""
    card_type: CardType
    position: Tuple[int, int]  # 中心位置
    bbox: Tuple[int, int, int, int]  # 边界框 (x1, y1, x2, y2)
    confidence: float
    detection_id: int  # 对应的检测ID
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class GameRegion:
    """游戏区域"""
    name: str
    cards: List[Card]
    bbox: Optional[Tuple[int, int, int, int]] = None  # 区域边界框
    
    def get_cards_by_type(self, card_type: CardType) -> List[Card]:
        """获取指定类型的卡牌"""
        return [card for card in self.cards if card.card_type == card_type]
    
    def get_card_count(self, card_type: CardType) -> int:
        """获取指定类型卡牌数量"""
        return len(self.get_cards_by_type(card_type))


class GameState:
    """游戏状态类"""
    
    def __init__(self):
        self.hand_region = GameRegion("hand", [])           # 手牌区域
        self.played_region = GameRegion("played", [])       # 已打出区域
        self.joker_region = GameRegion("jokers", [])        # 小丑牌区域
        self.shop_region = GameRegion("shop", [])           # 商店区域
        self.deck_region = GameRegion("deck", [])           # 牌堆区域
        self.discard_region = GameRegion("discard", [])     # 弃牌区域
        
        # 游戏信息
        self.round_info = {}
        self.score_info = {}
        self.money_info = {}
        
        # 可交互元素
        self.clickable_cards: List[Card] = []
        self.buttons: List[Dict] = []
    
    def get_all_regions(self) -> List[GameRegion]:
        """获取所有游戏区域"""
        return [
            self.hand_region, self.played_region, self.joker_region,
            self.shop_region, self.deck_region, self.discard_region
        ]
    
    def get_total_card_count(self) -> int:
        """获取总卡牌数量"""
        return sum(len(region.cards) for region in self.get_all_regions())
    
    def get_hand_cards(self) -> List[Card]:
        """获取手牌"""
        return self.hand_region.cards
    
    def get_joker_cards(self) -> List[Card]:
        """获取小丑牌"""
        return self.joker_region.cards
    
    def get_playable_cards(self) -> List[Card]:
        """获取可打出的卡牌（通常是手牌中的正面扑克牌）"""
        return self.hand_region.get_cards_by_type(CardType.POKER_CARD_FRONT)
    
    def to_dict(self) -> Dict:
        """转换为字典格式，便于AI分析"""
        return {
            "hand_cards": len(self.hand_region.cards),
            "played_cards": len(self.played_region.cards),
            "joker_cards": len(self.joker_region.cards),
            "shop_cards": len(self.shop_region.cards),
            "playable_cards": len(self.get_playable_cards()),
            "regions": {
                region.name: {
                    "card_count": len(region.cards),
                    "card_types": {card_type.value: region.get_card_count(CardType(card_type.value)) 
                                 for card_type in CardType}
                }
                for region in self.get_all_regions()
            },
            "clickable_elements": len(self.clickable_cards) + len(self.buttons)
        }


class GameStateAnalyzer:
    """游戏状态分析器"""
    
    def __init__(self):
        # 区域划分参数（相对于屏幕的比例）
        self.region_configs = {
            "hand": {"y_range": (0.7, 1.0), "x_range": (0.0, 1.0)},      # 底部手牌区
            "played": {"y_range": (0.4, 0.7), "x_range": (0.3, 0.7)},    # 中央已打出区
            "jokers": {"y_range": (0.1, 0.4), "x_range": (0.0, 1.0)},    # 上方小丑牌区
            "shop": {"y_range": (0.3, 0.8), "x_range": (0.0, 0.3)},      # 左侧商店区
            "deck": {"y_range": (0.7, 1.0), "x_range": (0.8, 1.0)},      # 右下牌堆区
            "discard": {"y_range": (0.7, 1.0), "x_range": (0.0, 0.2)}    # 左下弃牌区
        }
        
        # 卡牌尺寸阈值（用于区分不同类型的卡牌显示）
        self.card_size_thresholds = {
            "small": 2000,    # 小卡牌（如商店中的卡牌）
            "medium": 8000,   # 中等卡牌（如手牌）
            "large": 15000    # 大卡牌（如被选中或放大的卡牌）
        }
    
    def analyze(self, detections: List[Detection], screen_width: int, screen_height: int) -> GameState:
        """
        分析检测结果，构建游戏状态
        
        Args:
            detections: YOLO检测结果
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            
        Returns:
            游戏状态对象
        """
        game_state = GameState()
        
        # 转换检测结果为卡牌对象
        cards = self._detections_to_cards(detections)
        
        # 按区域分配卡牌
        self._assign_cards_to_regions(cards, game_state, screen_width, screen_height)
        
        # 识别可交互元素
        self._identify_interactive_elements(game_state)
        
        # 分析游戏信息
        self._analyze_game_info(game_state)
        
        return game_state
    
    def _detections_to_cards(self, detections: List[Detection]) -> List[Card]:
        """将检测结果转换为卡牌对象"""
        cards = []
        
        for i, detection in enumerate(detections):
            # 只处理卡牌类型的检测
            try:
                card_type = CardType(detection.class_name)
            except ValueError:
                continue  # 跳过非卡牌类型
            
            card = Card(
                card_type=card_type,
                position=detection.center,
                bbox=detection.bbox,
                confidence=detection.confidence,
                detection_id=i
            )
            cards.append(card)
        
        return cards
    
    def _assign_cards_to_regions(self, cards: List[Card], game_state: GameState, 
                               screen_width: int, screen_height: int) -> None:
        """将卡牌分配到对应的游戏区域"""
        
        for card in cards:
            # 计算卡牌相对位置
            rel_x = card.position[0] / screen_width
            rel_y = card.position[1] / screen_height
            
            # 根据位置和类型分配到区域
            assigned = False
            
            # 特殊类型直接分配
            if card.card_type == CardType.JOKER_CARD:
                game_state.joker_region.cards.append(card)
                assigned = True
            elif card.card_type == CardType.CARD_PACK:
                game_state.shop_region.cards.append(card)
                assigned = True
            elif card.card_type in [CardType.TAROT_CARD, CardType.PLANET_CARD, CardType.SPECTRAL_CARD]:
                # 根据位置判断是在商店还是手牌
                if self._is_in_region(rel_x, rel_y, self.region_configs["shop"]):
                    game_state.shop_region.cards.append(card)
                else:
                    game_state.hand_region.cards.append(card)
                assigned = True
            
            # 如果未分配，根据位置分配
            if not assigned:
                for region_name, region_config in self.region_configs.items():
                    if self._is_in_region(rel_x, rel_y, region_config):
                        region = getattr(game_state, f"{region_name}_region")
                        region.cards.append(card)
                        assigned = True
                        break
            
            # 如果仍未分配，根据卡牌大小和位置做最后判断
            if not assigned:
                card_area = card.area
                
                if rel_y > 0.7:  # 底部区域
                    if card_area > self.card_size_thresholds["medium"]:
                        game_state.hand_region.cards.append(card)
                    else:
                        game_state.deck_region.cards.append(card)
                elif rel_y < 0.4:  # 顶部区域
                    game_state.joker_region.cards.append(card)
                else:  # 中间区域
                    game_state.played_region.cards.append(card)
    
    def _is_in_region(self, rel_x: float, rel_y: float, region_config: Dict) -> bool:
        """检查相对坐标是否在指定区域内"""
        x_range = region_config["x_range"]
        y_range = region_config["y_range"]
        
        return (x_range[0] <= rel_x <= x_range[1] and 
                y_range[0] <= rel_y <= y_range[1])
    
    def _identify_interactive_elements(self, game_state: GameState) -> None:
        """识别可交互的元素"""
        # 手牌中的正面扑克牌通常是可点击的
        for card in game_state.hand_region.cards:
            if card.card_type == CardType.POKER_CARD_FRONT:
                game_state.clickable_cards.append(card)
        
        # 商店中的卡牌也是可点击的
        for card in game_state.shop_region.cards:
            game_state.clickable_cards.append(card)
        
        # 小丑牌区域的卡牌可能可点击（用于重新排列）
        for card in game_state.joker_region.cards:
            if card.card_type == CardType.JOKER_CARD:
                game_state.clickable_cards.append(card)
    
    def _analyze_game_info(self, game_state: GameState) -> None:
        """分析游戏信息（分数、金钱等）"""
        # 这里可以根据检测到的文本或特定UI元素来分析游戏信息
        # 目前先设置基本信息
        game_state.round_info = {
            "hand_size": len(game_state.hand_region.cards),
            "jokers_count": len(game_state.joker_region.cards)
        }
    
    def get_card_play_suggestions(self, game_state: GameState) -> List[Dict]:
        """获取卡牌打出建议"""
        suggestions = []
        
        playable_cards = game_state.get_playable_cards()
        
        if not playable_cards:
            return suggestions
        
        # 基于卡牌位置和数量给出建议
        if len(playable_cards) >= 5:
            suggestions.append({
                "action": "play_hand",
                "cards": playable_cards[:5],
                "reason": "可以打出一手完整的牌"
            })
        elif len(playable_cards) >= 1:
            # 选择最大的卡牌（通常在中间位置）
            center_x = sum(card.position[0] for card in playable_cards) / len(playable_cards)
            closest_card = min(playable_cards, 
                             key=lambda c: abs(c.position[0] - center_x))
            
            suggestions.append({
                "action": "play_single",
                "cards": [closest_card],
                "reason": "打出单张卡牌"
            })
        
        return suggestions
    
    def get_shop_suggestions(self, game_state: GameState) -> List[Dict]:
        """获取商店购买建议"""
        suggestions = []
        
        shop_cards = game_state.shop_region.cards
        
        for card in shop_cards:
            if card.card_type == CardType.JOKER_CARD:
                suggestions.append({
                    "action": "buy_joker",
                    "card": card,
                    "reason": "购买小丑牌可以提供持续效果"
                })
            elif card.card_type in [CardType.TAROT_CARD, CardType.PLANET_CARD, CardType.SPECTRAL_CARD]:
                suggestions.append({
                    "action": "buy_consumable",
                    "card": card,
                    "reason": f"购买{card.card_type.value}获得即时效果"
                })
        
        return suggestions


def test_game_state_analyzer():
    """测试游戏状态分析器"""
    from yolo_detector import YOLODetector, Detection
    import cv2
    import os
    
    # 创建模拟检测结果
    mock_detections = [
        Detection(6, "poker_card_front", 0.9, (100, 400, 150, 480)),  # 手牌
        Detection(6, "poker_card_front", 0.8, (160, 400, 210, 480)),  # 手牌
        Detection(6, "poker_card_front", 0.85, (220, 400, 270, 480)), # 手牌
        Detection(2, "joker_card", 0.95, (50, 50, 120, 150)),         # 小丑牌
        Detection(2, "joker_card", 0.9, (130, 50, 200, 150)),         # 小丑牌
        Detection(9, "tarot_card", 0.7, (20, 200, 80, 300)),          # 商店塔罗牌
        Detection(1, "card_pack", 0.8, (20, 320, 80, 380)),           # 卡包
    ]
    
    # 创建分析器
    analyzer = GameStateAnalyzer()
    
    # 分析游戏状态
    game_state = analyzer.analyze(mock_detections, 800, 600)
    
    # 打印分析结果
    print("=== 游戏状态分析结果 ===")
    print(f"手牌数量: {len(game_state.hand_region.cards)}")
    print(f"小丑牌数量: {len(game_state.joker_region.cards)}")
    print(f"商店卡牌数量: {len(game_state.shop_region.cards)}")
    print(f"可交互元素数量: {len(game_state.clickable_cards)}")
    
    # 获取建议
    play_suggestions = analyzer.get_card_play_suggestions(game_state)
    shop_suggestions = analyzer.get_shop_suggestions(game_state)
    
    print("\n=== 打牌建议 ===")
    for suggestion in play_suggestions:
        print(f"动作: {suggestion['action']}, 原因: {suggestion['reason']}")
    
    print("\n=== 购买建议 ===")
    for suggestion in shop_suggestions:
        print(f"动作: {suggestion['action']}, 原因: {suggestion['reason']}")
    
    # 转换为字典格式
    state_dict = game_state.to_dict()
    print(f"\n=== 状态字典 ===")
    print(state_dict)


if __name__ == "__main__":
    test_game_state_analyzer()
