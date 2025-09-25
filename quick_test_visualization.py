#!/usr/bin/env python3
"""
快速测试YOLO检测结果可视化

使用方法:
    python quick_test_visualization.py
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """简单的可视化测试。"""

    try:
        from ai_balatro.core.multi_yolo_detector import MultiYOLODetector
        from ai_balatro.core.screen_capture import ScreenCapture
        from ai_balatro.ai.actions.card_actions import DetectionVisualizer

        print('🧪 快速可视化测试 - 双模型系统')
        print('=' * 40)

        # 初始化
        print('初始化组件...')
        multi_detector = MultiYOLODetector()
        capture = ScreenCapture()
        visualizer = DetectionVisualizer()

        # 选择区域
        print('请选择游戏窗口...')
        if not capture.select_region_interactive():
            print('❌ 未选择窗口')
            return

        # 捕获和检测
        print('捕获屏幕并进行YOLO检测...')
        frame = capture.capture_once()
        if frame is None:
            print('❌ 捕获失败')
            return

        # 使用双模型检测
        entity_detections, ui_detections = multi_detector.detect_combined(frame)
        all_detections = entity_detections + ui_detections
        print(f'✅ 实体检测: {len(entity_detections)} 个对象')
        print(f'✅ UI检测: {len(ui_detections)} 个对象')
        print(f'✅ 总计: {len(all_detections)} 个对象')

        # 显示所有检测类型
        print('\n--- 实体检测结果 ---')
        for i, detection in enumerate(entity_detections):
            print(
                f'  {i + 1}. {detection.class_name} (置信度: {detection.confidence:.3f})'
            )

        print('\n--- UI检测结果 ---')
        for i, detection in enumerate(ui_detections):
            print(
                f'  {i + 1}. {detection.class_name} (置信度: {detection.confidence:.3f})'
            )

        # 显示可视化窗口
        print('\n显示检测结果可视化窗口...')
        visualizer.show_detection_results(
            frame, all_detections, 'Dual YOLO Detection Results'
        )

        print('✅ 测试完成')

    except ImportError as e:
        print(f'❌ 导入失败: {e}')
    except Exception as e:
        print(f'❌ 测试失败: {e}')


if __name__ == '__main__':
    main()
