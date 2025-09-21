"""
AI决策模块
使用大语言模型进行小丑牌游戏策略推理和决策

Author: RainbowBird
"""

import json
import time
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import openai
from anthropic import Anthropic
from game_state import GameState, Card, CardType


class AIProvider(Enum):
    """AI提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # 本地模型（如Ollama）


@dataclass
class Decision:
    """AI决策结果"""
    action_type: str  # 动作类型：play_cards, buy_item, skip, end_turn等
    target_cards: List[Card]  # 目标卡牌
    target_position: Optional[tuple] = None  # 目标位置（如果需要）
    confidence: float = 0.0  # 决策置信度
    reasoning: str = ""  # 决策推理过程
    priority: int = 1  # 优先级（1-10，10最高）
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "action_type": self.action_type,
            "target_cards": [{"type": card.card_type.value, "position": card.position} 
                           for card in self.target_cards],
            "target_position": self.target_position,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "priority": self.priority
        }


class BalatroAI:
    """小丑牌游戏AI决策器"""
    
    def __init__(self, provider: AIProvider = AIProvider.OPENAI, 
                 api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name or self._get_default_model()
        
        # 初始化AI客户端
        self._init_client()
        
        # 游戏知识库
        self.game_knowledge = self._load_game_knowledge()
        
        # 决策历史
        self.decision_history: List[Dict] = []
        
    def _get_default_model(self) -> str:
        """获取默认模型名称"""
        if self.provider == AIProvider.OPENAI:
            return "gpt-4o-mini"
        elif self.provider == AIProvider.ANTHROPIC:
            return "claude-3-5-sonnet-20241022"
        else:
            return "llama3.1:8b"
    
    def _init_client(self) -> None:
        """初始化AI客户端"""
        if self.provider == AIProvider.OPENAI:
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == AIProvider.ANTHROPIC:
            self.client = Anthropic(api_key=self.api_key)
        else:
            # 本地模型（Ollama等）
            self.client = None
    
    def _load_game_knowledge(self) -> Dict:
        """加载游戏知识库"""
        return {
            "poker_hands": {
                "high_card": {"score": 5, "mult": 1},
                "pair": {"score": 10, "mult": 2},
                "two_pair": {"score": 20, "mult": 2},
                "three_of_a_kind": {"score": 30, "mult": 3},
                "straight": {"score": 30, "mult": 4},
                "flush": {"score": 35, "mult": 4},
                "full_house": {"score": 40, "mult": 4},
                "four_of_a_kind": {"score": 60, "mult": 7},
                "straight_flush": {"score": 100, "mult": 8},
                "royal_flush": {"score": 100, "mult": 8}
            },
            "card_values": {
                "A": 11, "K": 10, "Q": 10, "J": 10,
                "10": 10, "9": 9, "8": 8, "7": 7,
                "6": 6, "5": 5, "4": 4, "3": 3, "2": 2
            },
            "joker_effects": {
                "common": "提供基础分数或倍数加成",
                "uncommon": "提供条件性加成或特殊效果",
                "rare": "提供强力效果或改变游戏机制",
                "legendary": "提供游戏改变性的强力效果"
            },
            "strategy_tips": [
                "优先考虑能形成高分牌型的组合",
                "保留有潜力的卡牌，弃掉低价值卡牌",
                "合理使用小丑牌的特殊效果",
                "注意手牌数量限制，及时打出卡牌",
                "在商店中优先购买符合当前策略的物品"
            ]
        }
    
    def make_decision(self, game_state: GameState, context: Optional[Dict] = None) -> Decision:
        """
        基于当前游戏状态做出决策
        
        Args:
            game_state: 当前游戏状态
            context: 额外上下文信息（如回合信息、分数等）
            
        Returns:
            AI决策结果
        """
        # 构建提示词
        prompt = self._build_decision_prompt(game_state, context)
        
        # 调用AI模型
        try:
            response = self._call_ai_model(prompt)
            decision = self._parse_ai_response(response, game_state)
        except Exception as e:
            print(f"AI决策失败: {e}")
            # 回退到基础策略
            decision = self._fallback_decision(game_state)
        
        # 记录决策历史
        self.decision_history.append({
            "timestamp": time.time(),
            "game_state": game_state.to_dict(),
            "decision": decision.to_dict(),
            "context": context
        })
        
        return decision
    
    def _build_decision_prompt(self, game_state: GameState, context: Optional[Dict]) -> str:
        """构建AI决策提示词"""
        
        # 基础游戏状态描述
        state_desc = f"""
当前游戏状态：
- 手牌数量: {len(game_state.hand_region.cards)}
- 可打出的扑克牌: {len(game_state.get_playable_cards())}
- 小丑牌数量: {len(game_state.joker_region.cards)}
- 商店可购买物品: {len(game_state.shop_region.cards)}
- 已打出的牌: {len(game_state.played_region.cards)}

手牌详情:
"""
        
        # 添加手牌信息
        for i, card in enumerate(game_state.hand_region.cards):
            state_desc += f"  {i+1}. {card.card_type.value} 位置({card.position[0]}, {card.position[1]})\n"
        
        # 添加小丑牌信息
        if game_state.joker_region.cards:
            state_desc += "\n小丑牌:\n"
            for i, card in enumerate(game_state.joker_region.cards):
                state_desc += f"  {i+1}. {card.card_type.value} 位置({card.position[0]}, {card.position[1]})\n"
        
        # 添加商店信息
        if game_state.shop_region.cards:
            state_desc += "\n商店物品:\n"
            for i, card in enumerate(game_state.shop_region.cards):
                state_desc += f"  {i+1}. {card.card_type.value} 位置({card.position[0]}, {card.position[1]})\n"
        
        # 添加上下文信息
        context_desc = ""
        if context:
            context_desc = f"\n额外信息: {json.dumps(context, ensure_ascii=False, indent=2)}"
        
        # 构建完整提示词
        prompt = f"""
你是一个专业的小丑牌(Balatro)游戏AI助手。请基于当前游戏状态做出最优决策。

{state_desc}{context_desc}

游戏规则提醒：
1. 小丑牌是一个扑克牌为基础的roguelike游戏
2. 目标是通过打出扑克牌组合获得分数
3. 小丑牌提供特殊效果和加成
4. 需要在有限的手牌和回合内达到目标分数

请分析当前情况并提供决策建议。你的回答必须是JSON格式，包含以下字段：
{{
    "action_type": "动作类型(play_cards/buy_item/skip/end_turn)",
    "target_cards": ["目标卡牌的索引列表"],
    "reasoning": "详细的决策推理过程",
    "confidence": "决策置信度(0-1)",
    "priority": "优先级(1-10)"
}}

可能的动作类型：
- play_cards: 打出选定的卡牌
- buy_item: 购买商店中的物品
- skip: 跳过当前动作
- end_turn: 结束回合
- reroll_shop: 重新刷新商店
- sell_joker: 出售小丑牌

请给出你的决策：
"""
        
        return prompt
    
    def _call_ai_model(self, prompt: str) -> str:
        """调用AI模型获取响应"""
        if self.provider == AIProvider.OPENAI:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的小丑牌游戏AI助手，擅长分析游戏状态并做出最优决策。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
            
        elif self.provider == AIProvider.ANTHROPIC:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0.7,
                system="你是一个专业的小丑牌游戏AI助手，擅长分析游戏状态并做出最优决策。",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
            
        else:
            # 本地模型调用（需要根据具体实现调整）
            raise NotImplementedError("本地模型调用尚未实现")
    
    def _parse_ai_response(self, response: str, game_state: GameState) -> Decision:
        """解析AI响应为决策对象"""
        try:
            # 尝试解析JSON响应
            response_data = json.loads(response.strip())
            
            action_type = response_data.get("action_type", "skip")
            reasoning = response_data.get("reasoning", "")
            confidence = float(response_data.get("confidence", 0.5))
            priority = int(response_data.get("priority", 1))
            
            # 解析目标卡牌
            target_cards = []
            target_card_indices = response_data.get("target_cards", [])
            
            if action_type == "play_cards":
                playable_cards = game_state.get_playable_cards()
                for idx in target_card_indices:
                    if isinstance(idx, int) and 0 <= idx < len(playable_cards):
                        target_cards.append(playable_cards[idx])
                    elif isinstance(idx, str) and idx.isdigit():
                        idx = int(idx) - 1  # 转换为0基索引
                        if 0 <= idx < len(playable_cards):
                            target_cards.append(playable_cards[idx])
                            
            elif action_type == "buy_item":
                shop_cards = game_state.shop_region.cards
                for idx in target_card_indices:
                    if isinstance(idx, int) and 0 <= idx < len(shop_cards):
                        target_cards.append(shop_cards[idx])
                    elif isinstance(idx, str) and idx.isdigit():
                        idx = int(idx) - 1  # 转换为0基索引
                        if 0 <= idx < len(shop_cards):
                            target_cards.append(shop_cards[idx])
            
            return Decision(
                action_type=action_type,
                target_cards=target_cards,
                confidence=confidence,
                reasoning=reasoning,
                priority=priority
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"解析AI响应失败: {e}")
            print(f"原始响应: {response}")
            return self._fallback_decision(game_state)
    
    def _fallback_decision(self, game_state: GameState) -> Decision:
        """回退决策策略（基于规则的简单策略）"""
        playable_cards = game_state.get_playable_cards()
        
        # 如果有可打出的卡牌，优先打出
        if playable_cards:
            # 简单策略：打出所有可打出的卡牌
            return Decision(
                action_type="play_cards",
                target_cards=playable_cards,
                confidence=0.6,
                reasoning="回退策略：打出所有可用的扑克牌",
                priority=5
            )
        
        # 如果商店有物品，考虑购买
        shop_cards = game_state.shop_region.cards
        if shop_cards:
            # 优先购买小丑牌
            joker_cards = [card for card in shop_cards if card.card_type == CardType.JOKER_CARD]
            if joker_cards:
                return Decision(
                    action_type="buy_item",
                    target_cards=[joker_cards[0]],
                    confidence=0.5,
                    reasoning="回退策略：购买第一个可用的小丑牌",
                    priority=3
                )
        
        # 默认跳过
        return Decision(
            action_type="skip",
            target_cards=[],
            confidence=0.3,
            reasoning="回退策略：没有明显的最优选择，跳过当前动作",
            priority=1
        )
    
    def get_decision_explanation(self, decision: Decision) -> str:
        """获取决策的详细解释"""
        explanation = f"""
决策分析：
动作类型: {decision.action_type}
置信度: {decision.confidence:.2f}
优先级: {decision.priority}/10

推理过程:
{decision.reasoning}

目标卡牌: {len(decision.target_cards)} 张
"""
        
        if decision.target_cards:
            explanation += "具体卡牌:\n"
            for i, card in enumerate(decision.target_cards):
                explanation += f"  {i+1}. {card.card_type.value} 位置({card.position[0]}, {card.position[1]})\n"
        
        return explanation
    
    def update_game_knowledge(self, feedback: Dict) -> None:
        """根据游戏反馈更新知识库"""
        # 这里可以实现强化学习或知识更新逻辑
        pass
    
    def get_statistics(self) -> Dict:
        """获取AI决策统计信息"""
        if not self.decision_history:
            return {"total_decisions": 0}
        
        action_counts = {}
        total_confidence = 0
        
        for record in self.decision_history:
            action_type = record["decision"]["action_type"]
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
            total_confidence += record["decision"]["confidence"]
        
        return {
            "total_decisions": len(self.decision_history),
            "action_distribution": action_counts,
            "average_confidence": total_confidence / len(self.decision_history),
            "recent_decisions": self.decision_history[-5:]  # 最近5个决策
        }


def test_ai_decision():
    """测试AI决策模块"""
    from game_state import GameState, Card, CardType
    
    # 创建模拟游戏状态
    game_state = GameState()
    
    # 添加一些模拟卡牌
    game_state.hand_region.cards = [
        Card(CardType.POKER_CARD_FRONT, (100, 400), (80, 380, 120, 420), 0.9, 0),
        Card(CardType.POKER_CARD_FRONT, (150, 400), (130, 380, 170, 420), 0.8, 1),
        Card(CardType.POKER_CARD_FRONT, (200, 400), (180, 380, 220, 420), 0.85, 2),
    ]
    
    game_state.joker_region.cards = [
        Card(CardType.JOKER_CARD, (50, 100), (30, 80, 70, 120), 0.95, 3),
    ]
    
    game_state.shop_region.cards = [
        Card(CardType.JOKER_CARD, (30, 200), (10, 180, 50, 220), 0.7, 4),
        Card(CardType.TAROT_CARD, (30, 250), (10, 230, 50, 270), 0.8, 5),
    ]
    
    # 创建AI决策器（使用模拟模式，不需要真实API）
    ai = BalatroAI(provider=AIProvider.OPENAI, api_key="test_key")
    
    # 模拟决策（由于没有真实API，会使用回退策略）
    print("=== AI决策测试 ===")
    try:
        decision = ai.make_decision(game_state, {"round": 1, "score": 0})
        print(ai.get_decision_explanation(decision))
    except Exception as e:
        print(f"决策测试失败，使用回退策略: {e}")
        decision = ai._fallback_decision(game_state)
        print(ai.get_decision_explanation(decision))
    
    # 显示统计信息
    stats = ai.get_statistics()
    print(f"\n=== 决策统计 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_ai_decision()
