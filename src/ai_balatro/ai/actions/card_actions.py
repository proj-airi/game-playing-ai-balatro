"""Card action engine for executing card-based actions in Balatro."""

import time
from typing import List
import numpy as np
import subprocess
import sys

from ...core.detection import Detection
from ...core.yolo_detector import YOLODetector
from ...core.screen_capture import ScreenCapture
from ...utils.logger import get_logger
from .schemas import CardAction

logger = get_logger(__name__)


class CardPositionDetector:
    """Detects and sorts cards by position from left to right."""
    
    def __init__(self):
        """Initialize card position detector."""
        # 优先级：可玩的牌类型
        self.playable_card_classes = [
            'poker_card_front',  # 扑克牌正面（最高优先级）
            'joker_card',        # 小丑牌
            'tarot_card',        # 塔罗牌
            'planet_card',       # 星球牌
            'spectral_card',     # 幽灵牌
        ]
    
    def get_hand_cards(self, detections: List[Detection]) -> List[Detection]:
        """
        从检测结果中提取手牌，按从左到右排序。
        
        Args:
            detections: YOLO检测结果
            
        Returns:
            排序后的手牌Detection列表
        """
        # 过滤出可玩的牌
        hand_cards = []
        
        for detection in detections:
            if self._is_playable_card(detection):
                hand_cards.append(detection)
        
        if not hand_cards:
            logger.warning("未检测到可玩的手牌")
            return []
        
        # 按x坐标排序（从左到右）
        hand_cards.sort(key=lambda card: card.bbox[0])  # x1坐标
        
        logger.info(f"检测到{len(hand_cards)}张手牌:")
        for i, card in enumerate(hand_cards):
            logger.info(f"  位置{i}: {card.class_name} at {card.center} (置信度: {card.confidence:.3f})")
        
        return hand_cards
    
    def _is_playable_card(self, detection: Detection) -> bool:
        """检查是否为可玩的牌。"""
        class_name = detection.class_name.lower()
        
        # 检查是否为可玩牌类型
        for card_class in self.playable_card_classes:
            if card_class in class_name:
                # 排除描述和背面
                if 'description' not in class_name and 'back' not in class_name:
                    return True
        
        return False


class CardActionEngine:
    """Engine for executing card actions based on position arrays."""
    
    def __init__(self, 
                 yolo_detector: YOLODetector,
                 screen_capture: ScreenCapture):
        """
        Initialize card action engine.
        
        Args:
            yolo_detector: YOLO检测器
            screen_capture: 屏幕捕获器
        """
        self.yolo_detector = yolo_detector
        self.screen_capture = screen_capture
        self.position_detector = CardPositionDetector()
        
        # 鼠标控制
        from pynput import mouse
        self.mouse = mouse.Controller()
        self.mouse_button = mouse.Button  # 保存Button引用
        
        # 点击间隔设置
        self.click_interval = 0.3  # 点击间隔（秒）
        self.action_delay = 0.5    # 动作间隔（秒）
        
        # 鼠标移动动画设置（优化速度）
        self.mouse_move_duration = 0.3  # 鼠标移动持续时间（秒，从0.5调整为0.3）
        self.mouse_move_steps = 15      # 移动步数（从20调整为15）
        self.click_hold_duration = 0.08 # 按住点击的时间（秒，从0.1调整为0.08）
        
        # 窗口焦点设置
        self.ensure_window_focus = True  # 是否在点击前确保窗口焦点
        self.focus_method = "auto"       # 焦点方法：auto, click, applescript
        
        logger.info("CardActionEngine初始化完成")
        logger.info(f"鼠标移动设置: 持续时间={self.mouse_move_duration}s, 步数={self.mouse_move_steps}, 点击保持={self.click_hold_duration}s")
        logger.info(f"窗口焦点设置: 启用={self.ensure_window_focus}, 方法={self.focus_method}")
    
    def set_mouse_animation_params(self, 
                                   move_duration: float = 0.5, 
                                   move_steps: int = 20, 
                                   click_hold_duration: float = 0.1) -> None:
        """
        设置鼠标移动动画参数。
        
        Args:
            move_duration: 鼠标移动持续时间（秒）
            move_steps: 移动步数（更多步数 = 更平滑）
            click_hold_duration: 点击保持时间（秒）
        """
        self.mouse_move_duration = move_duration
        self.mouse_move_steps = move_steps
        self.click_hold_duration = click_hold_duration
        
        logger.info(f"更新鼠标移动设置: 持续时间={move_duration}s, 步数={move_steps}, 点击保持={click_hold_duration}s")
    
    def ensure_game_window_focus(self) -> bool:
        """
        确保游戏窗口获得焦点。
        
        Returns:
            是否成功获得焦点
        """
        if not self.ensure_window_focus:
            return True  # 如果禁用焦点检查，直接返回成功
            
        try:
            # 获取捕获区域信息
            capture_region = self.screen_capture.get_capture_region()
            if not capture_region:
                logger.warning("无法获取捕获区域，跳过焦点处理")
                return True
            
            # 检测当前系统
            if sys.platform == "darwin":  # macOS
                return self._ensure_focus_macos(capture_region)
            elif sys.platform == "win32":  # Windows
                return self._ensure_focus_windows(capture_region)
            else:  # Linux/其他
                return self._ensure_focus_linux(capture_region)
                
        except Exception as e:
            logger.warning(f"窗口焦点处理失败: {e}")
            return True  # 失败时不阻止后续操作
    
    def _ensure_focus_macos(self, capture_region: dict) -> bool:
        """macOS系统下确保窗口焦点。"""
        try:
            if self.focus_method == "applescript" or self.focus_method == "auto":
                # 使用AppleScript激活窗口（推荐方法）
                script = '''
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    if frontApp is not "Balatro" then
                        tell application "Balatro" to activate
                        delay 0.1
                    end if
                end tell
                '''
                
                result = subprocess.run(
                    ["osascript", "-e", script], 
                    capture_output=True, 
                    text=True,
                    timeout=2
                )
                
                if result.returncode == 0:
                    logger.info("使用AppleScript激活Balatro窗口")
                    time.sleep(0.2)  # 等待窗口激活
                    return True
                else:
                    logger.warning(f"AppleScript激活失败: {result.stderr}")
            
            if self.focus_method == "click" or (self.focus_method == "auto" and self.focus_method != "applescript"):
                # 备选方案：点击窗口标题栏激活
                return self._click_to_focus(capture_region)
                
        except subprocess.TimeoutExpired:
            logger.warning("AppleScript执行超时")
        except Exception as e:
            logger.warning(f"macOS焦点处理失败: {e}")
            
        return True  # 失败时不阻止操作
    
    def _ensure_focus_windows(self, capture_region: dict) -> bool:
        """Windows系统下确保窗口焦点。"""
        try:
            # Windows下使用点击激活
            return self._click_to_focus(capture_region)
        except Exception as e:
            logger.warning(f"Windows焦点处理失败: {e}")
            return True
    
    def _ensure_focus_linux(self, capture_region: dict) -> bool:
        """Linux系统下确保窗口焦点。"""
        try:
            # Linux下使用点击激活
            return self._click_to_focus(capture_region)
        except Exception as e:
            logger.warning(f"Linux焦点处理失败: {e}")
            return True
    
    def _click_to_focus(self, capture_region: dict) -> bool:
        """通过点击窗口来获得焦点。"""
        try:
            # 计算窗口标题栏的点击位置（窗口顶部中央）
            title_bar_x = capture_region['left'] + capture_region['width'] // 2
            title_bar_y = capture_region['top'] + 10  # 标题栏区域
            
            current_pos = self.mouse.position
            
            # 快速点击标题栏激活窗口
            self.mouse.position = (title_bar_x, title_bar_y)
            time.sleep(0.05)
            self.mouse.click(self.mouse_button.left)
            time.sleep(0.1)
            
            # 恢复鼠标位置
            self.mouse.position = current_pos
            
            logger.info(f"通过点击激活窗口: ({title_bar_x}, {title_bar_y})")
            return True
            
        except Exception as e:
            logger.warning(f"点击激活窗口失败: {e}")
            return False
    
    def execute_card_action(self, positions: List[int], description: str = "", show_visualization: bool = False) -> bool:
        """
        执行牌动作。
        
        Args:
            positions: 位置数组，如 [1, 1, 1, 0] 或 [-1, -1, 0, 0]
            description: 动作描述
            show_visualization: 是否显示可视化窗口
            
        Returns:
            是否执行成功
        """
        logger.info(f"执行牌动作: {positions} - {description}")
        
        # 创建CardAction对象
        action = CardAction.from_array(positions, description)
        
        try:
            # 1. 捕获当前屏幕
            frame = self.screen_capture.capture_once()
            if frame is None:
                logger.error("屏幕捕获失败")
                return False
            
            # 2. 检测手牌
            detections = self.yolo_detector.detect(frame)
            hand_cards = self.position_detector.get_hand_cards(detections)
            
            if not hand_cards:
                logger.error("未检测到手牌")
                return False
            
            # 3. 验证位置数组长度
            if len(positions) > len(hand_cards):
                logger.warning(f"位置数组长度({len(positions)})超过手牌数量({len(hand_cards)})")
                # 截断位置数组
                positions = positions[:len(hand_cards)]
            
            # 4. 显示可视化（如果启用）
            if show_visualization:
                self._show_card_action_visualization(frame, hand_cards, action, positions)
            
            # 5. 确保游戏窗口焦点
            logger.info("确保游戏窗口获得焦点...")
            focus_success = self.ensure_game_window_focus()
            if focus_success:
                logger.info("✓ 游戏窗口焦点准备就绪")
            else:
                logger.warning("! 窗口焦点处理失败，继续执行操作")
            
            # 6. 执行点击操作
            success = self._execute_clicks(hand_cards, action)
            
            if success:
                # 7. 执行确认操作（出牌或弃牌）
                time.sleep(self.action_delay)
                success = self._execute_confirm_action(action)
            
            return success
            
        except Exception as e:
            logger.error(f"执行牌动作时发生错误: {e}")
            return False
    
    def _execute_clicks(self, hand_cards: List[Detection], action: CardAction) -> bool:
        """执行点击选择牌的操作。"""
        try:
            clicked_cards = []
            
            # 根据动作类型选择要点击的牌
            if action.is_play_action:
                # 出牌：点击position为1的牌
                target_indices = action.selected_indices
                logger.info(f"选择出牌: 位置 {target_indices}")
            elif action.is_discard_action:
                # 弃牌：点击position为-1的牌
                target_indices = action.discard_indices
                logger.info(f"选择弃牌: 位置 {target_indices}")
            else:
                logger.warning("未识别的动作类型")
                return False
            
            # 点击目标牌
            for index in target_indices:
                if index < len(hand_cards):
                    card = hand_cards[index]
                    success = self._click_card(card, index)
                    if success:
                        clicked_cards.append(card)
                        time.sleep(self.click_interval)
                    else:
                        logger.error(f"点击位置{index}的牌失败")
                        return False
                else:
                    logger.error(f"位置{index}超出手牌范围")
                    return False
            
            logger.info(f"成功选择了{len(clicked_cards)}张牌")
            return True
            
        except Exception as e:
            logger.error(f"执行点击操作时发生错误: {e}")
            return False
    
    def _click_card(self, card: Detection, index: int) -> bool:
        """点击指定的牌。"""
        try:
            # 获取牌的中心点
            center_x, center_y = card.center
            
            # 获取捕获区域信息用于坐标转换
            capture_region = self.screen_capture.get_capture_region()
            if not capture_region:
                logger.error("无法获取捕获区域信息")
                return False
            
            # 获取当前帧用于坐标缩放计算
            current_frame = self.screen_capture.capture_once()
            if current_frame is None:
                logger.error("无法获取当前帧")
                return False
            
            # 计算坐标转换
            actual_height, actual_width = current_frame.shape[:2]
            region_width = capture_region['width']
            region_height = capture_region['height']
            
            scale_x = region_width / actual_width
            scale_y = region_height / actual_height
            
            # 转换到屏幕坐标
            screen_x = capture_region['left'] + (center_x * scale_x)
            screen_y = capture_region['top'] + (center_y * scale_y)
            
            logger.info(f"点击位置{index}的牌: {card.class_name}")
            logger.info(f"  检测中心: ({center_x}, {center_y})")
            logger.info(f"  屏幕坐标: ({screen_x:.1f}, {screen_y:.1f})")
            
            # 平滑移动鼠标到目标位置
            if not self._smooth_move_mouse(int(screen_x), int(screen_y)):
                logger.error("平滑移动鼠标失败")
                return False
            
            # 短暂停顿，确保游戏识别鼠标位置
            time.sleep(0.2)
            
            # 执行更明确的点击操作（按下-保持-释放）
            logger.info("执行点击操作...")
            self.mouse.press(self.mouse_button.left)
            time.sleep(self.click_hold_duration)  # 保持按下状态
            self.mouse.release(self.mouse_button.left)
            
            # 点击后短暂等待
            time.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"点击牌时发生错误: {e}")
            return False
    
    def _execute_confirm_action(self, action: CardAction) -> bool:
        """执行确认动作（出牌或弃牌按钮）。"""
        try:
            if action.is_play_action:
                button_type = "play"
                logger.info("寻找并点击出牌按钮")
            elif action.is_discard_action:
                button_type = "discard" 
                logger.info("寻找并点击弃牌按钮")
            else:
                logger.info("无需确认动作")
                return True
            
            # 捕获当前屏幕寻找按钮
            frame = self.screen_capture.capture_once()
            if frame is None:
                logger.error("无法捕获屏幕寻找确认按钮")
                return False
            
            # TODO: 这里需要实现按钮检测和点击
            # 暂时使用简单的等待，实际应该检测按钮位置
            logger.warning(f"按钮检测功能待实现 - {button_type}")
            
            # 临时方案：等待用户手动操作
            logger.info(f"请手动点击{button_type}按钮完成操作")
            
            return True
            
        except Exception as e:
            logger.error(f"执行确认动作时发生错误: {e}")
            return False
    
    def hover_card(self, card_index: int, duration: float = 1.0) -> bool:
        """
        悬停在指定位置的牌上。
        
        Args:
            card_index: 牌的位置索引（从0开始）
            duration: 悬停持续时间
            
        Returns:
            是否执行成功
        """
        try:
            logger.info(f"悬停在位置{card_index}的牌上，持续{duration}秒")
            
            # 捕获屏幕并检测手牌
            frame = self.screen_capture.capture_once()
            if frame is None:
                return False
                
            detections = self.yolo_detector.detect(frame)
            hand_cards = self.position_detector.get_hand_cards(detections)
            
            if card_index >= len(hand_cards):
                logger.error(f"位置{card_index}超出手牌范围（共{len(hand_cards)}张牌）")
                return False
            
            card = hand_cards[card_index]
            
            # 计算屏幕坐标（复用_click_card的逻辑）
            center_x, center_y = card.center
            capture_region = self.screen_capture.get_capture_region()
            if not capture_region:
                return False
                
            current_frame = self.screen_capture.capture_once()
            if current_frame is None:
                return False
                
            actual_height, actual_width = current_frame.shape[:2]
            scale_x = capture_region['width'] / actual_width
            scale_y = capture_region['height'] / actual_height
            
            screen_x = capture_region['left'] + (center_x * scale_x)
            screen_y = capture_region['top'] + (center_y * scale_y)
            
            # 平滑移动鼠标到牌上
            if not self._smooth_move_mouse(int(screen_x), int(screen_y)):
                logger.error("平滑移动鼠标失败")
                return False
            
            logger.info(f"鼠标悬停在 ({screen_x:.1f}, {screen_y:.1f})")
            
            # 悬停指定时间
            time.sleep(duration)
            
            return True
            
        except Exception as e:
            logger.error(f"悬停操作失败: {e}")
            return False
    
    def _show_card_action_visualization(self, 
                                      image: np.ndarray, 
                                      hand_cards: List[Detection], 
                                      action: CardAction,
                                      positions: List[int]) -> None:
        """
        显示牌动作的可视化界面。
        
        Args:
            image: 原始图像
            hand_cards: 检测到的手牌列表
            action: 牌动作对象
            positions: 位置数组
        """
        try:
            import cv2
            
            # 创建可视化图像副本
            vis_image = image.copy()
            
            # 定义颜色
            colors = {
                'play': (0, 255, 0),      # 绿色 - 出牌
                'discard': (0, 0, 255),   # 红色 - 弃牌
                'neutral': (128, 128, 128) # 灰色 - 无操作
            }
            
            # 为每张牌绘制边框和标签
            for i, card in enumerate(hand_cards):
                if i < len(positions):
                    pos_value = positions[i]
                    if pos_value == 1:
                        color = colors['play']
                        action_text = "出牌"
                    elif pos_value == -1:
                        color = colors['discard']
                        action_text = "弃牌"
                    else:
                        color = colors['neutral']
                        action_text = "无操作"
                else:
                    color = colors['neutral']
                    action_text = "无操作"
                
                # 绘制边框
                x1, y1, x2, y2 = card.bbox
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
                
                # 绘制位置标签
                label = f"位置{i}: {action_text}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                # 计算文本大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # 绘制文本背景
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - text_height - baseline - 10),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # 绘制文本
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - baseline - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )
                
                # 在牌的中心绘制位置数字
                center_x, center_y = card.center
                cv2.circle(vis_image, (center_x, center_y), 15, (255, 255, 255), -1)
                cv2.putText(
                    vis_image,
                    str(i),
                    (center_x - 8, center_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2
                )
            
            # 添加标题信息
            title_text = f"操作预览: {action.action_type.value} - {positions}"
            cv2.putText(
                vis_image,
                title_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            # 添加说明文字
            instruction = "按任意键继续执行操作, ESC取消"
            cv2.putText(
                vis_image,
                instruction,
                (10, vis_image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1
            )
            
            # 显示窗口
            window_name = "Balatro Card Action Preview"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            cv2.imshow(window_name, vis_image)
            
            # 等待用户输入
            logger.info("显示操作预览窗口，按任意键继续...")
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(window_name)
            
            if key == 27:  # ESC键
                logger.info("用户取消操作")
                raise KeyboardInterrupt("用户取消操作")
                
        except Exception as e:
            logger.error(f"显示可视化时发生错误: {e}")
            # 继续执行，不因为可视化错误而中断操作
    
    def _smooth_move_mouse(self, target_x: int, target_y: int) -> bool:
        """
        平滑移动鼠标到目标位置。
        
        Args:
            target_x: 目标X坐标
            target_y: 目标Y坐标
            
        Returns:
            是否移动成功
        """
        try:
            import math
            
            # 获取当前鼠标位置
            start_x, start_y = self.mouse.position
            
            logger.info(f"平滑移动鼠标: ({start_x}, {start_y}) -> ({target_x}, {target_y})")
            
            # 计算移动距离
            distance_x = target_x - start_x
            distance_y = target_y - start_y
            total_distance = math.sqrt(distance_x**2 + distance_y**2)
            
            if total_distance < 5:  # 如果距离很近，直接移动
                self.mouse.position = (target_x, target_y)
                time.sleep(0.1)
                return True
            
            # 计算每步的移动量
            step_delay = self.mouse_move_duration / self.mouse_move_steps
            
            for i in range(self.mouse_move_steps + 1):
                # 使用缓动函数让移动更自然（慢-快-慢）
                progress = i / self.mouse_move_steps
                # 使用 ease-in-out 缓动函数
                eased_progress = self._ease_in_out(progress)
                
                current_x = int(start_x + distance_x * eased_progress)
                current_y = int(start_y + distance_y * eased_progress)
                
                self.mouse.position = (current_x, current_y)
                
                # 最后一步确保精确到达目标
                if i == self.mouse_move_steps:
                    self.mouse.position = (target_x, target_y)
                
                time.sleep(step_delay)
            
            # 验证最终位置
            final_x, final_y = self.mouse.position
            logger.info(f"鼠标移动完成: 最终位置 ({final_x}, {final_y})")
            
            return True
            
        except Exception as e:
            logger.error(f"平滑移动鼠标失败: {e}")
            # fallback: 直接移动
            try:
                self.mouse.position = (target_x, target_y)
                time.sleep(0.2)
                return True
            except Exception:
                return False
    
    def _ease_in_out(self, t: float) -> float:
        """
        缓动函数：慢-快-慢的移动效果。
        
        Args:
            t: 进度值 (0.0 到 1.0)
            
        Returns:
            缓动后的进度值
        """
        if t < 0.5:
            return 2 * t * t
        else:
            return -1 + (4 - 2 * t) * t
