#!/usr/bin/env python3
"""
YOLO检测结果可视化测试

使用方法:
    python test_yolo_visualization.py

功能:
    - 显示YOLO检测结果的CV窗口
    - 查看所有检测到的对象（牌、按钮、UI等）
    - 帮助调试为什么检测不到按钮
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_balatro.core.yolo_detector import YOLODetector
from ai_balatro.core.screen_capture import ScreenCapture
from ai_balatro.ai.actions import ActionExecutor
from ai_balatro.ai.actions.card_actions import ButtonDetector, DetectionVisualizer
from ai_balatro.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """主函数。"""

    print('📸 YOLO检测结果可视化测试')
    print('=' * 50)

    try:
        # 1. 初始化组件
        print('\n1. 初始化组件...')

        detector = YOLODetector()
        print('   ✓ YOLO检测器已加载')

        capture = ScreenCapture()
        print('   ✓ 屏幕捕获器已初始化')

        executor = ActionExecutor(detector, capture)
        executor.initialize()
        print('   ✓ 动作执行器已准备就绪')

        button_detector = ButtonDetector()
        visualizer = DetectionVisualizer()
        print('   ✓ 检测工具已初始化')

        # 2. 选择捕获区域
        print('\n2. 选择游戏窗口...')
        if not capture.select_region_interactive():
            print('   ❌ 未选择游戏窗口')
            print('   💡 请确保Balatro游戏正在运行')
            return
        print('   ✓ 游戏窗口已选择')

        # 3. 测试循环
        while True:
            print('\n' + '=' * 50)
            print('🔬 YOLO检测可视化菜单:')
            print('   1. 📸 显示完整YOLO检测结果')
            print('   2. 🔍 显示按钮检测结果')
            print('   3. 🃏 显示牌类检测结果')
            print('   4. 🎯 测试出牌按钮检测 (带可视化)')
            print('   5. 🗑️  测试弃牌按钮检测 (带可视化)')
            print('   6. 📊 显示检测统计信息')
            print('   7. 🔄 连续检测模式')
            print('   8. 退出')

            choice = input('\n请选择 (1-8): ').strip()

            if choice == '1':
                show_full_detection_results(detector, capture, visualizer)
            elif choice == '2':
                show_button_detection_results(
                    detector, capture, button_detector, visualizer
                )
            elif choice == '3':
                show_card_detection_results(detector, capture, visualizer)
            elif choice == '4':
                test_play_button_detection(executor)
            elif choice == '5':
                test_discard_button_detection(executor)
            elif choice == '6':
                show_detection_statistics(detector, capture, button_detector)
            elif choice == '7':
                continuous_detection_mode(detector, capture, visualizer)
            elif choice == '8':
                print('\n👋 退出测试')
                break
            else:
                print('   ❌ 无效选择，请输入 1-8')

    except Exception as e:
        logger.error(f'测试失败: {e}')
        print(f'❌ 测试失败: {e}')


def show_full_detection_results(detector, capture, visualizer):
    """显示完整的YOLO检测结果。"""
    print('\n📸 显示完整YOLO检测结果...')

    try:
        frame = capture.capture_once()
        if frame is None:
            print('   ❌ 屏幕捕获失败')
            return

        print('   正在进行YOLO检测...')
        detections = detector.detect(frame)

        print(f'   检测到 {len(detections)} 个对象')
        for i, detection in enumerate(detections):
            print(
                f'   {i + 1}. {detection.class_name} (置信度: {detection.confidence:.3f})'
            )

        print('\n   显示可视化窗口...')
        visualizer.show_detection_results(frame, detections, '完整YOLO检测结果')

    except Exception as e:
        print(f'   ❌ 检测失败: {e}')


def show_button_detection_results(detector, capture, button_detector, visualizer):
    """显示按钮检测结果。"""
    print('\n🔍 显示按钮检测结果...')

    try:
        frame = capture.capture_once()
        if frame is None:
            print('   ❌ 屏幕捕获失败')
            return

        print('   正在进行YOLO检测...')
        detections = detector.detect(frame)

        print('   分析按钮检测结果...')
        all_buttons = button_detector.find_buttons(detections)

        if all_buttons:
            print(f'   ✅ 找到 {len(all_buttons)} 个按钮:')
            for i, button in enumerate(all_buttons):
                print(f'      {i + 1}. {button.button_type} ({button.class_name})')
                print(
                    f'          位置: {button.center}, 置信度: {button.confidence:.3f}'
                )

            # 显示按钮检测结果
            button_detections = [btn for btn in all_buttons]
            visualizer.show_detection_results(frame, button_detections, '按钮检测结果')
        else:
            print('   ❌ 未检测到任何按钮')
            print('   显示所有检测结果以供参考...')
            visualizer.show_detection_results(
                frame, detections, '无按钮检测 - 所有检测结果'
            )

    except Exception as e:
        print(f'   ❌ 按钮检测失败: {e}')


def show_card_detection_results(detector, capture, visualizer):
    """显示牌类检测结果。"""
    print('\n🃏 显示牌类检测结果...')

    try:
        frame = capture.capture_once()
        if frame is None:
            print('   ❌ 屏幕捕获失败')
            return

        print('   正在进行YOLO检测...')
        detections = detector.detect(frame)

        # 过滤出牌类检测
        card_detections = []
        for detection in detections:
            class_name = detection.class_name.lower()
            if any(
                kw in class_name
                for kw in ['card', 'poker', 'joker', 'tarot', 'planet', 'spectral']
            ):
                card_detections.append(detection)

        print(f'   找到 {len(card_detections)} 张牌:')
        for i, card in enumerate(card_detections):
            print(f'      {i + 1}. {card.class_name} (置信度: {card.confidence:.3f})')

        if card_detections:
            visualizer.show_detection_results(frame, card_detections, '牌类检测结果')
        else:
            print('   ❌ 未检测到任何牌')
            visualizer.show_detection_results(
                frame, detections, '无牌检测 - 所有检测结果'
            )

    except Exception as e:
        print(f'   ❌ 牌类检测失败: {e}')


def test_play_button_detection(executor):
    """测试出牌按钮检测（带可视化）。"""
    print('\n🎯 测试出牌按钮检测...')

    try:
        print('   这将执行出牌动作并显示按钮检测的可视化过程')
        confirm = input('   确定要测试? (y/N): ').strip().lower()

        if confirm in ['y', 'yes']:
            success = executor.execute_from_array(
                [1, 1, 0, 0],
                '可视化出牌按钮检测测试',
                show_visualization=True,  # 启用可视化
            )

            if success:
                print('   ✅ 出牌按钮检测测试完成')
            else:
                print('   ❌ 出牌按钮检测测试失败')
        else:
            print('   测试取消')

    except Exception as e:
        print(f'   ❌ 出牌按钮测试失败: {e}')


def test_discard_button_detection(executor):
    """测试弃牌按钮检测（带可视化）。"""
    print('\n🗑️  测试弃牌按钮检测...')

    try:
        print('   这将执行弃牌动作并显示按钮检测的可视化过程')
        confirm = input('   确定要测试? (y/N): ').strip().lower()

        if confirm in ['y', 'yes']:
            success = executor.execute_from_array(
                [-1, -1, 0, 0],
                '可视化弃牌按钮检测测试',
                show_visualization=True,  # 启用可视化
            )

            if success:
                print('   ✅ 弃牌按钮检测测试完成')
            else:
                print('   ❌ 弃牌按钮检测测试失败')
        else:
            print('   测试取消')

    except Exception as e:
        print(f'   ❌ 弃牌按钮测试失败: {e}')


def show_detection_statistics(detector, capture, button_detector):
    """显示检测统计信息。"""
    print('\n📊 显示检测统计信息...')

    try:
        frame = capture.capture_once()
        if frame is None:
            print('   ❌ 屏幕捕获失败')
            return

        print('   正在进行YOLO检测...')
        detections = detector.detect(frame)

        # 统计各类检测结果
        card_count = 0
        button_count = 0
        ui_count = 0
        other_count = 0

        print(f'\n   📈 检测统计 (总数: {len(detections)}):')
        print('   ' + '-' * 40)

        for detection in detections:
            class_name = detection.class_name.lower()
            if any(
                kw in class_name
                for kw in ['card', 'poker', 'joker', 'tarot', 'planet', 'spectral']
            ):
                card_count += 1
            elif 'button' in class_name:
                button_count += 1
            elif any(kw in class_name for kw in ['ui', 'menu', 'text', 'score']):
                ui_count += 1
            else:
                other_count += 1

        print(f'   🃏 牌类: {card_count}')
        print(f'   🔘 按钮: {button_count}')
        print(f'   🖥️  UI元素: {ui_count}')
        print(f'   ❓ 其他: {other_count}')

        # 显示按钮详情
        if button_count > 0:
            print('\n   🔍 按钮详细信息:')
            all_buttons = button_detector.find_buttons(detections)
            for i, button in enumerate(all_buttons):
                print(f'      {i + 1}. {button.button_type} ({button.class_name})')
                print(
                    f'         置信度: {button.confidence:.3f}, 位置: {button.center}'
                )

        # 显示检测置信度分布
        print('\n   📊 置信度分布:')
        high_conf = len([d for d in detections if d.confidence >= 0.8])
        med_conf = len([d for d in detections if 0.5 <= d.confidence < 0.8])
        low_conf = len([d for d in detections if d.confidence < 0.5])

        print(f'      高置信度 (≥0.8): {high_conf}')
        print(f'      中置信度 (0.5-0.8): {med_conf}')
        print(f'      低置信度 (<0.5): {low_conf}')

    except Exception as e:
        print(f'   ❌ 统计失败: {e}')


def continuous_detection_mode(detector, capture, visualizer):
    """连续检测模式。"""
    print('\n🔄 连续检测模式...')
    print('   这将持续显示YOLO检测结果')
    print('   按 ESC 退出连续模式')

    try:
        import cv2

        frame_count = 0
        while True:
            frame = capture.capture_once()
            if frame is None:
                print('   ❌ 屏幕捕获失败')
                break

            detections = detector.detect(frame)
            frame_count += 1

            # 创建可视化
            vis_image = frame.copy()

            # 绘制检测结果
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{detection.class_name} ({detection.confidence:.2f})'
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            # 添加帧信息
            cv2.putText(
                vis_image,
                f'Frame: {frame_count}, Objects: {len(detections)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                vis_image,
                'Press ESC to exit',
                (10, vis_image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # 显示窗口
            cv2.namedWindow('连续检测模式', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('连续检测模式', 1000, 600)
            cv2.imshow('连续检测模式', vis_image)

            # 检查用户输入
            key = cv2.waitKey(100) & 0xFF
            if key == 27:  # ESC键
                print('   退出连续检测模式')
                break

        cv2.destroyAllWindows()

    except Exception as e:
        print(f'   ❌ 连续检测失败: {e}')


if __name__ == '__main__':
    main()
