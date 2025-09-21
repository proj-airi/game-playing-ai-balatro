"""
自动操作模块
实现鼠标点击、键盘输入等自动化操作，执行AI决策

Author: RainbowBird
"""

import time
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import pyautogui
from pynput import mouse, keyboard
from pynput.mouse import Button, Listener as MouseListener
from pynput.keyboard import Key, Listener as KeyboardListener
import threading

from ai_decision import Decision
from game_state import Card


class ActionType(Enum):
    """操作类型枚举"""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    KEY_PRESS = "key_press"
    KEY_COMBINATION = "key_combination"
    WAIT = "wait"
    SCROLL = "scroll"


@dataclass
class Action:
    """操作动作数据类"""
    action_type: ActionType
    position: Optional[Tuple[int, int]] = None
    target_position: Optional[Tuple[int, int]] = None  # 拖拽目标位置
    key: Optional[str] = None
    keys: Optional[List[str]] = None  # 组合键
    duration: float = 0.1  # 操作持续时间
    delay_before: float = 0.0  # 操作前延迟
    delay_after: float = 0.1  # 操作后延迟
    
    def __str__(self):
        if self.action_type == ActionType.CLICK:
            return f"点击 {self.position}"
        elif self.action_type == ActionType.DRAG:
            return f"拖拽 {self.position} -> {self.target_position}"
        elif self.action_type == ActionType.KEY_PRESS:
            return f"按键 {self.key}"
        elif self.action_type == ActionType.WAIT:
            return f"等待 {self.duration}s"
        else:
            return f"{self.action_type.value}"


class SafetyManager:
    """安全管理器，防止误操作"""
    
    def __init__(self):
        self.emergency_stop = False
        self.safe_zones: List[Tuple[int, int, int, int]] = []  # (x1, y1, x2, y2)
        self.forbidden_zones: List[Tuple[int, int, int, int]] = []
        self.max_actions_per_second = 10
        self.action_count = 0
        self.last_reset_time = time.time()
        
        # 设置紧急停止热键
        self._setup_emergency_stop()
    
    def _setup_emergency_stop(self):
        """设置紧急停止热键 (Ctrl+Shift+Q)"""
        def on_key_combination():
            self.emergency_stop = True
            print("🚨 紧急停止已激活！")
        
        def on_press(key):
            try:
                if (hasattr(key, 'char') and key.char == 'q' and 
                    keyboard.Controller().pressed(Key.ctrl) and 
                    keyboard.Controller().pressed(Key.shift)):
                    on_key_combination()
            except AttributeError:
                pass
        
        # 在后台线程中监听键盘
        listener = KeyboardListener(on_press=on_press)
        listener.daemon = True
        listener.start()
    
    def add_safe_zone(self, x1: int, y1: int, x2: int, y2: int):
        """添加安全操作区域"""
        self.safe_zones.append((x1, y1, x2, y2))
    
    def add_forbidden_zone(self, x1: int, y1: int, x2: int, y2: int):
        """添加禁止操作区域"""
        self.forbidden_zones.append((x1, y1, x2, y2))
    
    def is_position_safe(self, x: int, y: int) -> bool:
        """检查位置是否安全"""
        # 检查是否在禁止区域
        for fx1, fy1, fx2, fy2 in self.forbidden_zones:
            if fx1 <= x <= fx2 and fy1 <= y <= fy2:
                return False
        
        # 如果设置了安全区域，检查是否在安全区域内
        if self.safe_zones:
            for sx1, sy1, sx2, sy2 in self.safe_zones:
                if sx1 <= x <= sx2 and sy1 <= y <= sy2:
                    return True
            return False
        
        return True
    
    def check_rate_limit(self) -> bool:
        """检查操作频率限制"""
        current_time = time.time()
        
        # 重置计数器（每秒）
        if current_time - self.last_reset_time >= 1.0:
            self.action_count = 0
            self.last_reset_time = current_time
        
        # 检查是否超过限制
        if self.action_count >= self.max_actions_per_second:
            return False
        
        self.action_count += 1
        return True
    
    def should_stop(self) -> bool:
        """检查是否应该停止操作"""
        return self.emergency_stop


class AutoController:
    """自动控制器"""
    
    def __init__(self, game_region: Optional[Tuple[int, int, int, int]] = None):
        self.game_region = game_region  # (x, y, width, height)
        self.safety_manager = SafetyManager()
        self.is_running = False
        self.action_queue: List[Action] = []
        
        # 配置pyautogui
        pyautogui.FAILSAFE = True  # 鼠标移到左上角停止
        pyautogui.PAUSE = 0.1  # 每个操作间的默认暂停
        
        # 如果设置了游戏区域，将其设为安全区域
        if game_region:
            x, y, w, h = game_region
            self.safety_manager.add_safe_zone(x, y, x + w, y + h)
    
    def set_game_region(self, x: int, y: int, width: int, height: int):
        """设置游戏区域"""
        self.game_region = (x, y, width, height)
        self.safety_manager.safe_zones.clear()
        self.safety_manager.add_safe_zone(x, y, x + width, y + height)
        print(f"设置游戏区域: ({x}, {y}, {width}, {height})")
    
    def execute_decision(self, decision: Decision) -> bool:
        """
        执行AI决策
        
        Args:
            decision: AI决策对象
            
        Returns:
            是否执行成功
        """
        if self.safety_manager.should_stop():
            print("操作已被紧急停止")
            return False
        
        try:
            actions = self._decision_to_actions(decision)
            return self._execute_actions(actions)
        except Exception as e:
            print(f"执行决策失败: {e}")
            return False
    
    def _decision_to_actions(self, decision: Decision) -> List[Action]:
        """将AI决策转换为具体操作序列"""
        actions = []
        
        if decision.action_type == "play_cards":
            # 打出卡牌：点击每张卡牌
            for card in decision.target_cards:
                actions.append(Action(
                    action_type=ActionType.CLICK,
                    position=card.position,
                    delay_before=0.1,
                    delay_after=0.2
                ))
            
            # 如果选择了多张卡牌，可能需要确认打出
            if len(decision.target_cards) > 1:
                # 添加确认操作（通常是点击"Play Hand"按钮）
                # 这里需要根据实际游戏界面调整
                actions.append(Action(
                    action_type=ActionType.WAIT,
                    duration=0.5
                ))
        
        elif decision.action_type == "buy_item":
            # 购买物品：点击商店中的物品
            for card in decision.target_cards:
                actions.append(Action(
                    action_type=ActionType.CLICK,
                    position=card.position,
                    delay_before=0.2,
                    delay_after=0.3
                ))
        
        elif decision.action_type == "skip":
            # 跳过：可能需要点击跳过按钮或等待
            actions.append(Action(
                action_type=ActionType.WAIT,
                duration=1.0
            ))
        
        elif decision.action_type == "end_turn":
            # 结束回合：通常需要按空格键或点击结束按钮
            actions.append(Action(
                action_type=ActionType.KEY_PRESS,
                key="space",
                delay_after=0.5
            ))
        
        elif decision.action_type == "reroll_shop":
            # 重新刷新商店：通常按R键
            actions.append(Action(
                action_type=ActionType.KEY_PRESS,
                key="r",
                delay_after=0.5
            ))
        
        return actions
    
    def _execute_actions(self, actions: List[Action]) -> bool:
        """执行操作序列"""
        print(f"开始执行 {len(actions)} 个操作...")
        
        for i, action in enumerate(actions):
            if self.safety_manager.should_stop():
                print("操作被紧急停止")
                return False
            
            if not self.safety_manager.check_rate_limit():
                print("操作频率过高，等待...")
                time.sleep(1.0)
                continue
            
            print(f"执行操作 {i+1}/{len(actions)}: {action}")
            
            # 操作前延迟
            if action.delay_before > 0:
                time.sleep(action.delay_before)
            
            # 执行具体操作
            success = self._execute_single_action(action)
            if not success:
                print(f"操作 {i+1} 执行失败")
                return False
            
            # 操作后延迟
            if action.delay_after > 0:
                time.sleep(action.delay_after)
        
        print("所有操作执行完成")
        return True
    
    def _execute_single_action(self, action: Action) -> bool:
        """执行单个操作"""
        try:
            if action.action_type == ActionType.CLICK:
                return self._click(action.position)
            
            elif action.action_type == ActionType.DOUBLE_CLICK:
                return self._double_click(action.position)
            
            elif action.action_type == ActionType.RIGHT_CLICK:
                return self._right_click(action.position)
            
            elif action.action_type == ActionType.DRAG:
                return self._drag(action.position, action.target_position)
            
            elif action.action_type == ActionType.KEY_PRESS:
                return self._key_press(action.key)
            
            elif action.action_type == ActionType.KEY_COMBINATION:
                return self._key_combination(action.keys)
            
            elif action.action_type == ActionType.WAIT:
                time.sleep(action.duration)
                return True
            
            elif action.action_type == ActionType.SCROLL:
                return self._scroll(action.position, action.duration)
            
            else:
                print(f"未知操作类型: {action.action_type}")
                return False
                
        except Exception as e:
            print(f"执行操作失败: {e}")
            return False
    
    def _click(self, position: Tuple[int, int]) -> bool:
        """执行点击操作"""
        x, y = position
        
        if not self.safety_manager.is_position_safe(x, y):
            print(f"位置 ({x}, {y}) 不安全，跳过点击")
            return False
        
        # 添加随机偏移，模拟人类操作
        offset_x = random.randint(-3, 3)
        offset_y = random.randint(-3, 3)
        
        pyautogui.click(x + offset_x, y + offset_y)
        return True
    
    def _double_click(self, position: Tuple[int, int]) -> bool:
        """执行双击操作"""
        x, y = position
        
        if not self.safety_manager.is_position_safe(x, y):
            print(f"位置 ({x}, {y}) 不安全，跳过双击")
            return False
        
        pyautogui.doubleClick(x, y)
        return True
    
    def _right_click(self, position: Tuple[int, int]) -> bool:
        """执行右键点击操作"""
        x, y = position
        
        if not self.safety_manager.is_position_safe(x, y):
            print(f"位置 ({x}, {y}) 不安全，跳过右键点击")
            return False
        
        pyautogui.rightClick(x, y)
        return True
    
    def _drag(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> bool:
        """执行拖拽操作"""
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        if not (self.safety_manager.is_position_safe(x1, y1) and 
                self.safety_manager.is_position_safe(x2, y2)):
            print(f"拖拽路径不安全，跳过操作")
            return False
        
        pyautogui.drag(x2 - x1, y2 - y1, duration=0.5, button='left')
        return True
    
    def _key_press(self, key: str) -> bool:
        """执行按键操作"""
        pyautogui.press(key)
        return True
    
    def _key_combination(self, keys: List[str]) -> bool:
        """执行组合键操作"""
        pyautogui.hotkey(*keys)
        return True
    
    def _scroll(self, position: Tuple[int, int], amount: float) -> bool:
        """执行滚轮操作"""
        x, y = position
        
        if not self.safety_manager.is_position_safe(x, y):
            print(f"位置 ({x}, {y}) 不安全，跳过滚轮操作")
            return False
        
        pyautogui.scroll(int(amount), x=x, y=y)
        return True
    
    def add_custom_action(self, action: Action):
        """添加自定义操作到队列"""
        self.action_queue.append(action)
    
    def execute_queue(self) -> bool:
        """执行操作队列"""
        if not self.action_queue:
            return True
        
        actions = self.action_queue.copy()
        self.action_queue.clear()
        
        return self._execute_actions(actions)
    
    def emergency_stop(self):
        """紧急停止所有操作"""
        self.safety_manager.emergency_stop = True
        self.action_queue.clear()
        print("🚨 自动控制已紧急停止")
    
    def reset_emergency_stop(self):
        """重置紧急停止状态"""
        self.safety_manager.emergency_stop = False
        print("✅ 紧急停止状态已重置")
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """获取当前鼠标位置"""
        return pyautogui.position()
    
    def wait_for_click(self, timeout: float = 10.0) -> Optional[Tuple[int, int]]:
        """等待用户点击，返回点击位置"""
        print(f"等待用户点击（{timeout}秒超时）...")
        
        clicked_position = None
        click_event = threading.Event()
        
        def on_click(x, y, button, pressed):
            nonlocal clicked_position
            if pressed and button == Button.left:
                clicked_position = (x, y)
                click_event.set()
                return False  # 停止监听
        
        # 启动鼠标监听
        listener = MouseListener(on_click=on_click)
        listener.start()
        
        # 等待点击或超时
        if click_event.wait(timeout):
            print(f"检测到点击: {clicked_position}")
            return clicked_position
        else:
            print("等待点击超时")
            listener.stop()
            return None


def test_auto_controller():
    """测试自动控制器"""
    from ai_decision import Decision
    from game_state import Card, CardType
    
    # 创建自动控制器
    controller = AutoController()
    
    # 设置游戏区域（示例）
    controller.set_game_region(100, 100, 800, 600)
    
    # 创建模拟决策
    mock_cards = [
        Card(CardType.POKER_CARD_FRONT, (200, 400), (180, 380, 220, 420), 0.9, 0),
        Card(CardType.POKER_CARD_FRONT, (250, 400), (230, 380, 270, 420), 0.8, 1),
    ]
    
    decision = Decision(
        action_type="play_cards",
        target_cards=mock_cards,
        confidence=0.8,
        reasoning="测试决策：打出两张卡牌",
        priority=5
    )
    
    print("=== 自动控制器测试 ===")
    print("注意：这将执行真实的鼠标操作！")
    print("按 Ctrl+Shift+Q 可以紧急停止")
    
    # 等待用户确认
    input("按回车键开始测试（确保鼠标不在重要位置）...")
    
    # 执行决策
    success = controller.execute_decision(decision)
    print(f"决策执行结果: {'成功' if success else '失败'}")
    
    # 测试自定义操作
    print("\n测试自定义操作...")
    controller.add_custom_action(Action(
        action_type=ActionType.WAIT,
        duration=1.0
    ))
    
    controller.add_custom_action(Action(
        action_type=ActionType.KEY_PRESS,
        key="space"
    ))
    
    success = controller.execute_queue()
    print(f"队列执行结果: {'成功' if success else '失败'}")


if __name__ == "__main__":
    test_auto_controller()
