#!/usr/bin/env python3
"""
Action Module Demo - 演示如何使用牌动作系统

使用方法:
    python examples/action_demo.py

功能:
    - 演示位置数组的牌动作
    - 展示出牌和弃牌操作
    - 悬停查看牌的详情
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_balatro.core.multi_yolo_detector import MultiYOLODetector
from ai_balatro.core.screen_capture import ScreenCapture
from ai_balatro.ai.actions import ActionExecutor
from ai_balatro.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """主函数：演示action模块功能。"""

    print('🃏 Balatro Action Module Demo - 双YOLO模型系统')
    print('=' * 60)

    try:
        # 1. 初始化组件
        print('\n1. 初始化双模型系统...')

        # 初始化多模型YOLO检测器
        multi_detector = MultiYOLODetector()
        available_models = multi_detector.get_available_models()
        print(f'   ✓ 双YOLO模型已加载: {available_models}')

        # 显示模型信息
        model_info = multi_detector.get_model_info()
        for model_name, info in model_info.items():
            if info['available']:
                print(f'     - {model_name.upper()}: {info["classes_count"]}个类别')

        # 初始化屏幕捕获
        capture = ScreenCapture()
        print('   ✓ 屏幕捕获器已初始化')

        # 初始化动作执行器（使用双模型系统）
        executor = ActionExecutor(screen_capture=capture, multi_detector=multi_detector)
        executor.initialize()
        print('   ✓ 双模型动作执行器已准备就绪')

        # 2. 检测游戏窗口
        print('\n2. 检测游戏窗口...')
        if not capture.select_region_interactive():
            print('   ❌ 未检测到Balatro游戏窗口')
            print('   💡 请确保Balatro游戏正在运行')
            return
        print('   ✓ 游戏窗口检测成功')

        # 3. 显示可用动作
        print('\n3. 可用动作:')
        actions = executor.get_available_actions()
        for i, action in enumerate(actions, 1):
            print(f'   {i}. {action["name"]}: {action["description"]}')

        # 4. 鼠标移动设置
        print('\n4. 鼠标移动设置:')
        print('   当前设置: 平滑移动动画已启用')
        mouse_config = input('是否调整鼠标移动速度? (y/N): ').strip().lower()

        if mouse_config in ['y', 'yes']:
            print('   鼠标移动速度设置:')
            print('     1. 快速移动 (0.3s, 15步)')
            print('     2. 标准移动 (0.5s, 20步) - 默认')
            print('     3. 慢速移动 (0.8s, 30步)')
            print('     4. 自定义设置')

            speed_choice = input('   请选择 (1-4, 默认2): ').strip()

            if speed_choice == '1':
                executor.set_mouse_animation_params(0.3, 15, 0.08)
                print('   ✓ 设置为快速移动模式')
            elif speed_choice == '3':
                executor.set_mouse_animation_params(0.8, 30, 0.12)
                print('   ✓ 设置为慢速移动模式')
            elif speed_choice == '4':
                try:
                    duration = float(input('   移动持续时间(秒, 0.2-2.0): ') or '0.5')
                    steps = int(input('   移动步数(10-50): ') or '20')
                    hold = float(input('   点击保持时间(秒, 0.05-0.3): ') or '0.1')
                    executor.set_mouse_animation_params(duration, steps, hold)
                    print(f'   ✓ 自定义设置完成: {duration}s, {steps}步')
                except ValueError:
                    print('   ❌ 输入无效，使用默认设置')
            else:
                print('   ✓ 使用标准移动模式')

        # 5. 演示模式选择
        print('\n5. 选择演示模式 (双模型系统):')
        print('   1. 智能出牌 [1, 1, 1, 0] - 卡牌检测 + 按钮识别')
        print('   2. 智能弃牌 [-1, -1, 0, 0] - 卡牌检测 + 按钮识别')
        print('   3. 悬停示例 - 查看第一张牌的详情')
        print('   4. 自定义位置数组')
        print('   5. 🔍 双模型可视化 - 实时查看检测过程')
        print('   6. 🎯 按钮检测演示 - 展示UI模型能力')
        print('   7. 鼠标移动测试 - 测试平滑移动效果')
        print('   8. 退出')

        while True:
            try:
                choice = input('\n请选择操作 (1-8): ').strip()

                if choice == '1':
                    # 智能出牌示例
                    print('\n🃏 执行智能出牌操作: [1, 1, 1, 0]')
                    print('   使用entities模型检测卡牌，UI模型检测出牌按钮')
                    success = executor.execute_from_array(
                        [1, 1, 1, 0], '双模型智能出牌 - 选择前三张牌'
                    )
                    print(f'   结果: {"✓ 成功" if success else "❌ 失败"}')

                elif choice == '2':
                    # 智能弃牌示例
                    print('\n🗑️  执行智能弃牌操作: [-1, -1, 0, 0]')
                    print('   使用entities模型检测卡牌，UI模型检测弃牌按钮')
                    success = executor.execute_from_array(
                        [-1, -1, 0, 0], '双模型智能弃牌 - 弃掉前两张牌'
                    )
                    print(f'   结果: {"✓ 成功" if success else "❌ 失败"}')

                elif choice == '3':
                    # 悬停示例
                    print('\n👆 悬停在第一张牌上...')
                    result = executor.process(
                        {
                            'function_call': {
                                'name': 'hover_card',
                                'arguments': {'card_index': 0, 'duration': 2.0},
                            }
                        }
                    )
                    print(f'   结果: {"✓ 成功" if result.success else "❌ 失败"}')

                elif choice == '4':
                    # 自定义位置数组
                    print('\n⚙️  自定义位置数组')
                    print('   提示: 使用 1 表示选择出牌，-1 表示弃牌，0 表示不操作')
                    print('   示例: 1,1,0,0 或 -1,-1,-1,0')

                    positions_input = input('   输入位置数组 (逗号分隔): ').strip()
                    description = input('   输入操作描述 (可选): ').strip()

                    try:
                        positions = [int(x.strip()) for x in positions_input.split(',')]

                        # 验证输入
                        if not all(val in [-1, 0, 1] for val in positions):
                            print('   ❌ 位置数组只能包含 -1, 0, 1')
                            continue

                        print(f'\n🎯 执行自定义操作: {positions}')
                        success = executor.execute_from_array(positions, description)
                        print(f'   结果: {"✓ 成功" if success else "❌ 失败"}')

                    except ValueError:
                        print('   ❌ 输入格式错误，请使用数字和逗号')

                elif choice == '5':
                    # 双模型可视化测试
                    print('\n🔍 双模型可视化测试')
                    print('   特色: 实时显示双模型检测过程和结果')
                    print('   选择测试操作:')
                    print('     a. 可视化智能出牌 [1, 1, 0, 0] - 显示卡牌+按钮检测')
                    print('     b. 可视化智能弃牌 [-1, -1, 0, 0] - 显示卡牌+按钮检测')
                    print('     c. 返回主菜单')

                    vis_choice = input('   请选择 (a/b/c): ').strip().lower()

                    if vis_choice == 'a':
                        print('\n🃏 双模型可视化出牌: [1, 1, 0, 0]')
                        print('   🔹 entities模型识别卡牌，UI模型识别出牌按钮')
                        success = executor.execute_from_array(
                            [1, 1, 0, 0], '双模型可视化出牌', show_visualization=True
                        )
                        print(f'   结果: {"✓ 成功" if success else "❌ 失败"}')
                    elif vis_choice == 'b':
                        print('\n🗑️  双模型可视化弃牌: [-1, -1, 0, 0]')
                        print('   🔹 entities模型识别卡牌，UI模型识别弃牌按钮')
                        success = executor.execute_from_array(
                            [-1, -1, 0, 0],
                            '双模型可视化弃牌',
                            show_visualization=True,
                        )
                        print(f'   结果: {"✓ 成功" if success else "❌ 失败"}')
                    elif vis_choice == 'c':
                        continue
                    else:
                        print('   ❌ 无效选择')

                elif choice == '6':
                    # 按钮检测演示
                    print('\n🎯 UI模型按钮检测演示')
                    print('   展示UI模型识别各种按钮的能力')

                    # 简单的按钮检测测试
                    print('   正在捕获当前屏幕并检测所有按钮...')
                    frame = capture.capture_once()
                    if frame is not None:
                        ui_detections = multi_detector.detect_ui(frame)
                        buttons = [
                            d for d in ui_detections if 'button' in d.class_name.lower()
                        ]

                        print(f'   ✓ UI模型检测结果: {len(ui_detections)} 个UI元素')
                        print(f'   ✓ 其中按钮: {len(buttons)} 个')

                        if buttons:
                            print('   检测到的按钮:')
                            for i, btn in enumerate(buttons[:5]):  # 显示前5个
                                print(
                                    f'     {i + 1}. {btn.class_name} (置信度: {btn.confidence:.3f})'
                                )
                            if len(buttons) > 5:
                                print(f'     ... 还有{len(buttons) - 5}个按钮')
                        else:
                            print('   当前界面未检测到按钮')

                        show_vis = (
                            input('   显示UI检测可视化窗口? (y/N): ').strip().lower()
                        )
                        if show_vis in ['y', 'yes']:
                            from ai_balatro.ai.actions.card_actions import (
                                DetectionVisualizer,
                            )

                            visualizer = DetectionVisualizer()
                            visualizer.show_detection_results(
                                frame, ui_detections, 'UI模型检测结果'
                            )
                    else:
                        print('   ❌ 屏幕捕获失败')

                elif choice == '7':
                    # 鼠标移动测试
                    print('\n🖱️  鼠标移动测试')
                    print('   这个测试会让你观察鼠标的平滑移动效果')
                    print('   鼠标会慢慢移动到第一张牌上，但不会点击')

                    test_confirm = input('   开始测试? (y/N): ').strip().lower()
                    if test_confirm in ['y', 'yes']:
                        result = executor.process(
                            {
                                'function_call': {
                                    'name': 'hover_card',
                                    'arguments': {'card_index': 0, 'duration': 1.0},
                                }
                            }
                        )
                        if result.success:
                            print('   ✅ 鼠标移动测试成功! 你应该看到了平滑的移动动画')
                        else:
                            print('   ❌ 鼠标移动测试失败')
                    else:
                        print('   测试取消')

                elif choice == '8':
                    print('\n👋 退出演示')
                    break

                else:
                    print('   ❌ 无效选择，请输入 1-8')

            except KeyboardInterrupt:
                print('\n\n👋 演示被中断')
                break
            except Exception as e:
                logger.error(f'执行操作时发生错误: {e}')
                print(f'   ❌ 发生错误: {e}')

    except Exception as e:
        logger.error(f'初始化失败: {e}')
        print(f'❌ 初始化失败: {e}')
        return


def show_usage():
    """显示使用说明。"""
    print('\n🚀 双YOLO模型系统 - 智能游戏控制')
    print('=' * 50)
    print('🎯 系统特色:')
    print('   • ENTITIES模型 - 精确识别卡牌、小丑牌等游戏实体')
    print('   • UI模型 - 智能检测按钮和界面元素')
    print('   • 自动化流程 - 卡牌选择 + 按钮识别 + 一键执行')
    print('   • 可视化调试 - 实时查看AI检测过程')
    print()
    print('📖 位置数组使用说明:')
    print('   • 数组中的每个数字代表对应位置牌的操作')
    print('   • 1: 选择该位置的牌用于出牌')
    print('   • -1: 选择该位置的牌用于弃牌')
    print('   • 0: 不对该位置的牌进行操作')
    print()
    print('📝 示例:')
    print('   [1, 1, 1, 0]    # 选择前三张牌出牌（自动点击出牌按钮）')
    print('   [-1, -1, 0, 0]  # 弃掉前两张牌（自动点击弃牌按钮）')
    print('   [1, 0, 1, 0]    # 选择第1和第3张牌出牌')
    print('   [-1, 0, 0, -1]  # 弃掉第1和第4张牌')
    print()
    print('⚠️  注意事项:')
    print('   • 同一个数组中不能同时包含正数和负数')
    print('   • 数组长度会根据当前手牌数量自动调整')
    print('   • 需要确保Balatro游戏窗口可见且未被遮挡')
    print('   • 系统会自动识别和点击对应的按钮（出牌/弃牌）')
    print()
    print('✨ 双模型优势:')
    print('   • 更精确的卡牌检测')
    print('   • 智能按钮识别（不再依赖固定坐标）')
    print('   • 实时可视化调试')
    print('   • 更强的适应性和鲁棒性')


if __name__ == '__main__':
    show_usage()
    main()
