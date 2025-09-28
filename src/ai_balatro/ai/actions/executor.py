"""Action executor for coordinating all game actions."""

from typing import List, Dict, Any, Optional
import json

from ...ai.llm.base import ProcessingResult, BaseProcessor, ProcessorType
from ...core.yolo_detector import YOLODetector
from ...core.multi_yolo_detector import MultiYOLODetector
from ...core.screen_capture import ScreenCapture
from ...utils.logger import get_logger
from .card_actions import CardActionEngine
from .schemas import GAME_ACTIONS, BUTTON_CONFIG

logger = get_logger(__name__)


class ActionExecutor(BaseProcessor):
    """统一的游戏动作执行器，处理所有类型的游戏操作。"""

    def __init__(
        self,
        yolo_detector: Optional[YOLODetector] = None,
        screen_capture: Optional[ScreenCapture] = None,
        multi_detector: Optional[MultiYOLODetector] = None,
    ):
        """
        初始化动作执行器。

        Args:
            yolo_detector: 传统单一YOLO检测器（向后兼容）
            screen_capture: 屏幕捕获器
            multi_detector: 多模型YOLO检测器（推荐使用）
        """
        super().__init__('ActionExecutor', ProcessorType.ACTION)

        # 验证必要参数
        if screen_capture is None:
            raise ValueError('screen_capture is required')

        self.screen_capture = screen_capture

        # 支持向后兼容和新的多模型系统
        if multi_detector is not None:
            self.multi_detector = multi_detector
            self.yolo_detector = None
            logger.info('ActionExecutor使用多模型YOLO检测器')
        elif yolo_detector is not None:
            self.yolo_detector = yolo_detector
            self.multi_detector = None
            logger.info('ActionExecutor使用传统单一YOLO检测器')
        else:
            # 如果都没有提供，则创建默认的多模型检测器
            self.multi_detector = MultiYOLODetector()
            self.yolo_detector = None
            logger.info('ActionExecutor创建默认多模型YOLO检测器')

        # 初始化子执行器
        self.card_engine = CardActionEngine(
            yolo_detector=self.yolo_detector,
            screen_capture=self.screen_capture,
            multi_detector=self.multi_detector,
        )

        logger.info('ActionExecutor初始化完成')

    def initialize(self) -> bool:
        """初始化执行器。"""
        try:
            self.is_initialized = True
            logger.info('ActionExecutor初始化成功')
            return True
        except Exception as e:
            logger.error(f'ActionExecutor初始化失败: {e}')
            return False

    def process(
        self, input_data: Any, context: Optional[Dict] = None
    ) -> ProcessingResult:
        """
        处理动作请求。

        Args:
            input_data: 动作数据，可以是函数调用结果或直接的动作参数
            context: 执行上下文

        Returns:
            处理结果
        """
        try:
            # 解析输入数据
            if isinstance(input_data, dict):
                action_data = input_data
            elif isinstance(input_data, str):
                # 尝试解析JSON字符串
                try:
                    action_data = json.loads(input_data)
                except json.JSONDecodeError:
                    return ProcessingResult(
                        success=False,
                        data=None,
                        errors=[f'无法解析动作数据: {input_data}'],
                    )
            else:
                return ProcessingResult(
                    success=False,
                    data=None,
                    errors=[f'不支持的动作数据类型: {type(input_data)}'],
                )

            # 根据动作类型执行相应操作
            if 'function_call' in action_data:
                return self._execute_function_call(action_data['function_call'])
            elif 'positions' in action_data:
                return self._execute_card_positions(action_data)
            else:
                return ProcessingResult(
                    success=False, data=None, errors=['无法识别的动作格式']
                )

        except Exception as e:
            logger.error(f'处理动作时发生错误: {e}')
            return ProcessingResult(success=False, data=None, errors=[str(e)])

    def _execute_function_call(self, function_call: Dict[str, Any]) -> ProcessingResult:
        """执行LLM函数调用。"""
        function_name = function_call.get('name')
        arguments = function_call.get('arguments', {})

        logger.info(f'执行函数调用: {function_name}')
        logger.info(f'参数: {arguments}')

        if function_name == 'select_cards_by_position':
            return self._execute_card_positions(arguments)
        elif function_name == 'hover_card':
            return self._execute_hover_card(arguments)
        elif function_name == 'click_button':
            return self._execute_click_button(arguments)
        else:
            return ProcessingResult(
                success=False, data=None, errors=[f'未知的函数: {function_name}']
            )

    def _execute_card_positions(
        self, args: Dict[str, Any], show_visualization: bool = False
    ) -> ProcessingResult:
        """执行基于位置数组的牌操作。"""
        positions = args.get('positions', [])
        description = args.get('description', '')

        if not positions:
            return ProcessingResult(
                success=False, data=None, errors=['位置数组不能为空']
            )

        # 验证位置数组格式
        if not all(val in [-1, 0, 1] for val in positions):
            return ProcessingResult(
                success=False, data=None, errors=['位置数组只能包含-1, 0, 1']
            )

        success = self.card_engine.execute_card_action(
            positions, description, show_visualization
        )

        return ProcessingResult(
            success=success,
            data={
                'action': 'card_positions',
                'positions': positions,
                'description': description,
                'executed': success,
            },
        )

    def _execute_hover_card(self, args: Dict[str, Any]) -> ProcessingResult:
        """执行悬停牌操作。"""
        card_index = args.get('card_index')
        duration = args.get('duration', 1.0)

        if card_index is None:
            return ProcessingResult(
                success=False, data=None, errors=['缺少card_index参数']
            )

        success = self.card_engine.hover_card(card_index, duration)

        return ProcessingResult(
            success=success,
            data={
                'action': 'hover_card',
                'card_index': card_index,
                'duration': duration,
                'executed': success,
            },
        )

    def _execute_click_button(self, args: Dict[str, Any]) -> ProcessingResult:
        """执行点击按钮操作。"""
        button_type = args.get('button_type')

        if not button_type:
            return ProcessingResult(
                success=False, data=None, errors=['缺少button_type参数']
            )

        if button_type not in BUTTON_CONFIG:
            return ProcessingResult(
                success=False, data=None, errors=[f'不支持的按钮类型: {button_type}']
            )

        # Use button detector to find and click the button
        try:
            # Capture current frame
            frame = self.screen_capture.capture_once()
            if frame is None:
                return ProcessingResult(
                    success=False, data=None, errors=['无法捕获屏幕画面']
                )

            # Find the button using the button detector
            button_config = BUTTON_CONFIG[button_type]
            keywords = button_config['keywords']

            # Detect UI elements
            if self.multi_detector:
                ui_detections = self.multi_detector.detect_ui(frame)
            else:
                # Fallback for single detector (should not happen normally)
                logger.warning('使用单一检测器作为后备方案')
                ui_detections = []

            # Look for button by matching keywords in class names
            target_button = None
            for detection in ui_detections:
                class_name_lower = detection.class_name.lower()
                if any(keyword.lower() in class_name_lower for keyword in keywords):
                    target_button = detection
                    break

            if target_button is None:
                return ProcessingResult(
                    success=False,
                    data={
                        'action': 'click_button',
                        'button_type': button_type,
                        'executed': False,
                        'available_buttons': [
                            d.class_name
                            for d in ui_detections
                            if 'button' in d.class_name.lower()
                        ],
                    },
                    errors=[f'未找到对应按钮: {button_type}，可能按钮不可见或检测失败'],
                )

            # Ensure the Balatro window is focused before attempting the click
            focus_ok = self.card_engine.mouse_controller.ensure_game_window_focus()
            if not focus_ok:
                logger.warning('窗口焦点处理失败，继续尝试点击按钮')

            # Convert detection coordinates to screen coordinates
            capture_region = self.screen_capture.get_capture_region()
            frame_height, frame_width = frame.shape[:2]
            if capture_region:
                scale_x = capture_region['width'] / frame_width
                scale_y = capture_region['height'] / frame_height
                screen_x = int(
                    capture_region['left'] + target_button.center[0] * scale_x
                )
                screen_y = int(
                    capture_region['top'] + target_button.center[1] * scale_y
                )
            else:
                logger.debug('未设置截图区域，使用检测坐标作为屏幕坐标')
                screen_x = int(target_button.center[0])
                screen_y = int(target_button.center[1])

            screen_position = (screen_x, screen_y)
            logger.info(
                f'按钮 {button_type} 检测中心 {target_button.center} -> 屏幕坐标 {screen_position}'
            )

            # Click the button using the card engine's mouse controller
            move_success = self.card_engine.mouse_controller.smooth_move_to(
                screen_x, screen_y
            )

            if move_success:
                # Add a small delay before clicking
                import time

                time.sleep(0.1)
                # Click at the button location
                click_success = self.card_engine.mouse_controller.click_at(
                    screen_x, screen_y, move_first=False
                )

                if click_success:
                    logger.info(f'成功点击按钮: {button_type} at {screen_position}')
                else:
                    logger.warning(
                        f'按钮点击可能失败: {button_type} at {screen_position}'
                    )

                return ProcessingResult(
                    success=True,
                    data={
                        'action': 'click_button',
                        'button_type': button_type,
                        'executed': True,
                        'button_position': screen_position,
                        'button_class': target_button.class_name,
                        'confidence': target_button.confidence,
                    },
                )
            else:
                return ProcessingResult(
                    success=False,
                    data=None,
                    errors=[f'鼠标移动到按钮位置失败: {screen_position}'],
                )

        except Exception as e:
            logger.error(f'按钮点击执行失败: {e}')
            return ProcessingResult(
                success=False, data=None, errors=[f'按钮点击执行失败: {e}']
            )

    def execute_from_array(
        self,
        positions: List[int],
        description: str = '',
        show_visualization: bool = False,
    ) -> bool:
        """
        便捷方法：直接从位置数组执行动作。

        Args:
            positions: 位置数组，如 [1, 1, 1, 0] 或 [-1, -1, 0, 0]
            description: 操作描述
            show_visualization: 是否显示可视化窗口

        Returns:
            是否执行成功

        Example:
            # 选择前三张牌出牌（带可视化）
            executor.execute_from_array([1, 1, 1, 0], "出前三张牌", show_visualization=True)

            # 弃掉前两张牌
            executor.execute_from_array([-1, -1, 0, 0], "弃掉前两张牌")
        """
        result = self._execute_card_positions(
            {'positions': positions, 'description': description}, show_visualization
        )
        return result.success

    def get_available_actions(self) -> List[Dict[str, Any]]:
        """获取可用的动作列表。"""
        return GAME_ACTIONS.copy()

    def get_action_status(self) -> Dict[str, Any]:
        """获取执行器状态信息。"""
        return {
            'initialized': self.is_initialized,
            'available_functions': [action['name'] for action in GAME_ACTIONS],
            'card_engine_ready': hasattr(self, 'card_engine'),
            'screen_capture_ready': self.screen_capture is not None,
            'yolo_detector_ready': self.yolo_detector is not None,
        }

    def set_mouse_animation_params(
        self,
        move_duration: float = 0.3,
        move_steps: int = 15,
        click_hold_duration: float = 0.08,
    ) -> None:
        """
        设置鼠标移动动画参数。

        Args:
            move_duration: 鼠标移动持续时间（秒），建议0.2-1.0（默认0.3，更快）
            move_steps: 移动步数（更多步数 = 更平滑），建议10-50（默认15）
            click_hold_duration: 点击保持时间（秒），建议0.05-0.2（默认0.08）
        """
        if hasattr(self, 'card_engine'):
            self.card_engine.set_mouse_animation_params(
                move_duration, move_steps, click_hold_duration
            )

    def set_window_focus_settings(
        self, enable: bool = True, method: str = 'auto'
    ) -> None:
        """
        设置窗口焦点处理参数。

        Args:
            enable: 是否启用窗口焦点确保功能
            method: 焦点处理方法 ("auto", "click", "applescript")
                - auto: 自动选择最佳方法（推荐）
                - click: 通过点击窗口标题栏激活
                - applescript: 仅macOS，使用AppleScript激活应用
        """
        if hasattr(self, 'card_engine'):
            self.card_engine.mouse_controller.ensure_window_focus = enable
            self.card_engine.mouse_controller.focus_method = method
            logger.info(f'窗口焦点设置更新: 启用={enable}, 方法={method}')
