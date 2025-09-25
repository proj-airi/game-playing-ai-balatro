#!/usr/bin/env python3
"""
测试双YOLO模型系统
- Entities模型：卡牌检测
- UI模型：按钮和界面检测

使用方法:
    python test_dual_yolo_models.py
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_balatro.core.multi_yolo_detector import MultiYOLODetector
from ai_balatro.core.screen_capture import ScreenCapture
from ai_balatro.ai.actions import ActionExecutor
from ai_balatro.ai.actions.card_actions import DetectionVisualizer
from ai_balatro.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """主测试函数。"""

    print('🎮 双YOLO模型系统测试')
    print('=' * 50)

    try:
        # 1. 初始化组件
        print('\n📦 1. 初始化双模型系统...')

        multi_detector = MultiYOLODetector()
        capture = ScreenCapture()
        visualizer = DetectionVisualizer()

        # 显示模型信息
        model_info = multi_detector.get_model_info()
        print('\n📊 模型信息:')
        for model_name, info in model_info.items():
            status = '✅ 可用' if info['available'] else '❌ 不可用'
            print(f'   {model_name.upper()}: {status}')
            if info['available']:
                print(f'      描述: {info["description"]}')
                print(f'      类别数: {info["classes_count"]}')
                print(
                    f'      类别示例: {", ".join(info["class_names"][:5])}{"..." if len(info["class_names"]) > 5 else ""}'
                )

        # 2. 创建动作执行器
        print('\n🤖 2. 初始化动作执行器...')
        executor = ActionExecutor(screen_capture=capture, multi_detector=multi_detector)
        executor.initialize()
        print('   ✓ 动作执行器初始化完成')

        # 3. 选择游戏窗口
        print('\n🎯 3. 选择游戏窗口...')
        if not capture.select_region_interactive():
            print('   ❌ 未选择游戏窗口')
            return
        print('   ✓ 游戏窗口已选择')

        # 4. 测试菜单
        while True:
            print('\n' + '=' * 60)
            print('🧪 双模型系统测试菜单:')
            print('   1. 📸 实体检测测试 (卡牌、小丑等)')
            print('   2. 🔲 UI检测测试 (按钮、界面元素)')
            print('   3. 🎯 双模型联合检测')
            print('   4. 🃏 完整出牌流程测试 (卡牌+按钮)')
            print('   5. 🗑️  完整弃牌流程测试 (卡牌+按钮)')
            print('   6. 📊 检测性能对比')
            print('   7. 🔧 模型信息显示')
            print('   8. 退出')

            choice = input('\n请选择测试项目 (1-8): ').strip()

            if choice == '1':
                test_entities_detection(multi_detector, capture, visualizer)
            elif choice == '2':
                test_ui_detection(multi_detector, capture, visualizer)
            elif choice == '3':
                test_combined_detection(multi_detector, capture, visualizer)
            elif choice == '4':
                test_play_cards_workflow(executor)
            elif choice == '5':
                test_discard_cards_workflow(executor)
            elif choice == '6':
                test_detection_performance(multi_detector, capture)
            elif choice == '7':
                show_model_details(multi_detector)
            elif choice == '8':
                print('\n👋 测试完成，退出')
                break
            else:
                print('   ❌ 无效选择，请输入 1-8')

    except Exception as e:
        logger.error(f'测试失败: {e}')
        print(f'❌ 测试失败: {e}')


def test_entities_detection(multi_detector, capture, visualizer):
    """测试实体检测（卡牌模型）。"""
    print('\n📸 实体检测测试...')

    try:
        frame = capture.capture_once()
        if frame is None:
            print('   ❌ 屏幕捕获失败')
            return

        print('   正在使用ENTITIES模型检测...')
        entities = multi_detector.detect_entities(frame)

        print(f'   ✅ 检测到 {len(entities)} 个实体:')
        for i, entity in enumerate(entities):
            print(
                f'      {i + 1}. {entity.class_name} (置信度: {entity.confidence:.3f})'
            )

        # 分类统计
        card_count = len([e for e in entities if 'card' in e.class_name.lower()])
        joker_count = len([e for e in entities if 'joker' in e.class_name.lower()])
        other_count = len(entities) - card_count - joker_count

        print('\n   📈 分类统计:')
        print(f'      卡牌: {card_count}')
        print(f'      小丑: {joker_count}')
        print(f'      其他: {other_count}')

        # 显示可视化
        show_vis = input('\n   显示可视化窗口? (y/N): ').strip().lower()
        if show_vis in ['y', 'yes']:
            visualizer.show_detection_results(
                frame, entities, 'ENTITIES Model Detection'
            )

    except Exception as e:
        print(f'   ❌ 实体检测测试失败: {e}')


def test_ui_detection(multi_detector, capture, visualizer):
    """测试UI检测（按钮模型）。"""
    print('\n🔲 UI检测测试...')

    try:
        frame = capture.capture_once()
        if frame is None:
            print('   ❌ 屏幕捕获失败')
            return

        print('   正在使用UI模型检测...')
        ui_elements = multi_detector.detect_ui(frame)

        print(f'   ✅ 检测到 {len(ui_elements)} 个UI元素:')
        for i, ui in enumerate(ui_elements):
            print(f'      {i + 1}. {ui.class_name} (置信度: {ui.confidence:.3f})')

        # 分类统计
        button_count = len([e for e in ui_elements if 'button' in e.class_name.lower()])
        ui_data_count = len(
            [
                e
                for e in ui_elements
                if 'ui_' in e.class_name.lower()
                and 'button' not in e.class_name.lower()
            ]
        )

        print('\n   📈 分类统计:')
        print(f'      按钮: {button_count}')
        print(f'      UI数据: {ui_data_count}')

        # 按钮详情
        if button_count > 0:
            print('\n   🔘 检测到的按钮:')
            buttons = [e for e in ui_elements if 'button' in e.class_name.lower()]
            for i, btn in enumerate(buttons):
                print(f'      {i + 1}. {btn.class_name}')

        # 显示可视化
        show_vis = input('\n   显示可视化窗口? (y/N): ').strip().lower()
        if show_vis in ['y', 'yes']:
            visualizer.show_detection_results(frame, ui_elements, 'UI Model Detection')

    except Exception as e:
        print(f'   ❌ UI检测测试失败: {e}')


def test_combined_detection(multi_detector, capture, visualizer):
    """测试双模型联合检测。"""
    print('\n🎯 双模型联合检测测试...')

    try:
        frame = capture.capture_once()
        if frame is None:
            print('   ❌ 屏幕捕获失败')
            return

        print('   正在进行双模型联合检测...')
        entities, ui_elements = multi_detector.detect_combined(frame)
        all_detections = entities + ui_elements

        print('   ✅ 联合检测结果:')
        print(f'      实体模型: {len(entities)} 个对象')
        print(f'      UI模型: {len(ui_elements)} 个对象')
        print(f'      总计: {len(all_detections)} 个对象')

        # 详细分类
        cards = [e for e in entities if 'card' in e.class_name.lower()]
        buttons = [e for e in ui_elements if 'button' in e.class_name.lower()]
        ui_data = [
            e
            for e in ui_elements
            if 'ui_' in e.class_name.lower() and 'button' not in e.class_name.lower()
        ]

        print('\n   📊 详细分类:')
        print(f'      🃏 卡牌: {len(cards)}')
        print(f'      🔘 按钮: {len(buttons)}')
        print(f'      📊 UI数据: {len(ui_data)}')
        print(
            f'      📦 其他: {len(all_detections) - len(cards) - len(buttons) - len(ui_data)}'
        )

        # 显示可视化
        show_vis = input('\n   显示联合检测可视化? (y/N): ').strip().lower()
        if show_vis in ['y', 'yes']:
            visualizer.show_detection_results(
                frame, all_detections, 'Combined Dual-Model Detection'
            )

    except Exception as e:
        print(f'   ❌ 联合检测测试失败: {e}')


def test_play_cards_workflow(executor):
    """测试完整出牌流程。"""
    print('\n🃏 完整出牌流程测试...')
    print('   这将测试：卡牌检测 → 选择卡牌 → 按钮检测 → 点击出牌按钮')

    confirm = input('   确定要执行完整出牌流程? (y/N): ').strip().lower()
    if confirm not in ['y', 'yes']:
        print('   测试取消')
        return

    try:
        print('\n   执行出牌流程: [1,1,0,0] (选择前两张牌)')
        success = executor.execute_from_array(
            [1, 1, 0, 0],
            '双模型系统出牌流程测试',
            show_visualization=True,  # 启用可视化查看检测过程
        )

        if success:
            print('   ✅ 完整出牌流程测试成功！')
            print('   🎉 卡牌检测和按钮检测都正常工作')
        else:
            print('   ❌ 出牌流程测试失败')

    except Exception as e:
        print(f'   ❌ 出牌流程测试失败: {e}')


def test_discard_cards_workflow(executor):
    """测试完整弃牌流程。"""
    print('\n🗑️  完整弃牌流程测试...')
    print('   这将测试：卡牌检测 → 选择卡牌 → 按钮检测 → 点击弃牌按钮')

    confirm = input('   确定要执行完整弃牌流程? (y/N): ').strip().lower()
    if confirm not in ['y', 'yes']:
        print('   测试取消')
        return

    try:
        print('\n   执行弃牌流程: [-1,-1,0,0] (弃掉前两张牌)')
        success = executor.execute_from_array(
            [-1, -1, 0, 0],
            '双模型系统弃牌流程测试',
            show_visualization=True,  # 启用可视化查看检测过程
        )

        if success:
            print('   ✅ 完整弃牌流程测试成功！')
            print('   🎉 卡牌检测和按钮检测都正常工作')
        else:
            print('   ❌ 弃牌流程测试失败')

    except Exception as e:
        print(f'   ❌ 弃牌流程测试失败: {e}')


def test_detection_performance(multi_detector, capture):
    """测试检测性能对比。"""
    print('\n📊 检测性能对比测试...')

    try:
        frame = capture.capture_once()
        if frame is None:
            print('   ❌ 屏幕捕获失败')
            return

        import time

        # 测试实体检测
        print('   测试实体检测性能...')
        start_time = time.time()
        entities = multi_detector.detect_entities(frame)
        entities_time = time.time() - start_time

        # 测试UI检测
        print('   测试UI检测性能...')
        start_time = time.time()
        ui_elements = multi_detector.detect_ui(frame)
        ui_time = time.time() - start_time

        # 测试联合检测
        print('   测试联合检测性能...')
        start_time = time.time()
        combined_entities, combined_ui = multi_detector.detect_combined(frame)
        combined_time = time.time() - start_time

        # 显示结果
        print('\n   ⏱️ 性能测试结果:')
        print(f'      实体检测: {entities_time:.3f}s ({len(entities)} 个对象)')
        print(f'      UI检测: {ui_time:.3f}s ({len(ui_elements)} 个对象)')
        print(
            f'      联合检测: {combined_time:.3f}s ({len(combined_entities + combined_ui)} 个对象)'
        )
        print(f'      理论串行: {(entities_time + ui_time):.3f}s')

        if combined_time < (entities_time + ui_time):
            print('   ✅ 联合检测效率更高！')
        else:
            print('   ℹ️ 联合检测时间接近串行执行')

    except Exception as e:
        print(f'   ❌ 性能测试失败: {e}')


def show_model_details(multi_detector):
    """显示模型详细信息。"""
    print('\n🔧 模型详细信息...')

    try:
        model_info = multi_detector.get_model_info()
        available_models = multi_detector.get_available_models()

        print(f'\n   📋 可用模型: {", ".join(available_models)}')

        for model_name, info in model_info.items():
            print(f'\n   📦 {model_name.upper()} 模型:')
            print(f'      状态: {"✅ 可用" if info["available"] else "❌ 不可用"}')
            print(f'      描述: {info["description"]}')
            print(f'      模型路径: {info["model_path"]}')

            if info['available']:
                print(f'      类别数量: {info["classes_count"]}')
                print('      类别列表:')
                class_names = info['class_names']
                for i, class_name in enumerate(class_names, 1):
                    print(f'         {i:2d}. {class_name}')

        # 显示特定类别
        button_classes = multi_detector.get_button_classes()
        card_classes = multi_detector.get_card_classes()

        if button_classes:
            print(f'\n   🔘 按钮类别 ({len(button_classes)}):')
            for btn_class in button_classes:
                print(f'      • {btn_class}')

        if card_classes:
            print(f'\n   🃏 卡牌类别 ({len(card_classes)}):')
            for card_class in card_classes:
                print(f'      • {card_class}')

    except Exception as e:
        print(f'   ❌ 获取模型信息失败: {e}')


if __name__ == '__main__':
    main()
