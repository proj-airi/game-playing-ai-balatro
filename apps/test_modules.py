#!/usr/bin/env python3
"""
模块测试脚本
测试各个模块是否正常工作

Author: RainbowBird
"""

import os
import sys
from pathlib import Path

# 当前目录就是apps，不需要额外添加路径

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        from screen_capture import ScreenCapture
        print("✅ 屏幕捕捉模块导入成功")
    except ImportError as e:
        print(f"❌ 屏幕捕捉模块导入失败: {e}")
        return False
    
    try:
        from yolo_detector import YOLODetector
        print("✅ YOLO检测模块导入成功")
    except ImportError as e:
        print(f"❌ YOLO检测模块导入失败: {e}")
        return False
    
    try:
        from game_state import GameStateAnalyzer
        print("✅ 游戏状态分析模块导入成功")
    except ImportError as e:
        print(f"❌ 游戏状态分析模块导入失败: {e}")
        return False
    
    try:
        from ai_decision import BalatroAI
        print("✅ AI决策模块导入成功")
    except ImportError as e:
        print(f"❌ AI决策模块导入失败: {e}")
        return False
    
    try:
        from auto_control import AutoController
        print("✅ 自动控制模块导入成功")
    except ImportError as e:
        print(f"❌ 自动控制模块导入失败: {e}")
        return False
    
    try:
        from balatro_ai_bot import BalatroAIBot
        print("✅ 主控制器模块导入成功")
    except ImportError as e:
        print(f"❌ 主控制器模块导入失败: {e}")
        return False
    
    return True

def test_dependencies():
    """测试依赖包"""
    print("\n📦 测试依赖包...")
    
    dependencies = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("PIL", "pillow"),
        ("pyautogui", "pyautogui"),
        ("pynput", "pynput"),
        ("mss", "mss"),
        ("ultralytics", "ultralytics"),
        ("openai", "openai"),
        ("anthropic", "anthropic")
    ]
    
    missing = []
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - 缺失")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️ 缺失依赖包: {', '.join(missing)}")
        print("请运行: pixi install 或 pip install " + " ".join(missing))
        return False
    
    return True

def test_model_files():
    """测试模型文件"""
    print("\n🎯 测试模型文件...")
    
    model_paths = [
        "/home/neko/Git/github.com/proj-airi/game-playing-ai-balatro/runs/v2-balatro-entities-2000-epoch/weights/best.pt",
        "/home/neko/Git/github.com/proj-airi/game-playing-ai-balatro/models/games-balatro-2024-yolo-entities-detection/model.pt",
        "/home/neko/Git/github.com/proj-airi/game-playing-ai-balatro/runs/v2-balatro-entities/weights/best.pt"
    ]
    
    found_model = False
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ 找到模型: {path}")
            found_model = True
        else:
            print(f"❌ 模型不存在: {path}")
    
    if not found_model:
        print("\n⚠️ 未找到可用的YOLO模型文件")
        print("请确保已训练好YOLO模型")
        return False
    
    return True

def test_screen_capture():
    """测试屏幕捕捉"""
    print("\n📸 测试屏幕捕捉...")
    
    try:
        from screen_capture import ScreenCapture
        
        capture = ScreenCapture()
        width, height = capture.get_screen_size()
        print(f"✅ 屏幕尺寸: {width}x{height}")
        
        # 测试单次截图
        frame = capture.capture_once()
        if frame is not None:
            print(f"✅ 截图成功: {frame.shape}")
        else:
            print("❌ 截图失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 屏幕捕捉测试失败: {e}")
        return False

def test_yolo_detector():
    """测试YOLO检测器"""
    print("\n🎯 测试YOLO检测器...")
    
    try:
        from yolo_detector import YOLODetector
        
        # 查找可用模型
        model_paths = [
            "/home/neko/Git/github.com/proj-airi/game-playing-ai-balatro/runs/v2-balatro-entities-2000-epoch/weights/best.pt",
            "/home/neko/Git/github.com/proj-airi/game-playing-ai-balatro/models/games-balatro-2024-yolo-entities-detection/model.pt"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("❌ 未找到YOLO模型文件")
            return False
        
        detector = YOLODetector(model_path, use_onnx=False)
        print(f"✅ YOLO检测器初始化成功")
        print(f"✅ 类别数量: {len(detector.class_names)}")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO检测器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 小丑牌AI机器人模块测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("依赖包", test_dependencies),
        ("模型文件", test_model_files),
        ("屏幕捕捉", test_screen_capture),
        ("YOLO检测器", test_yolo_detector)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！机器人已准备就绪")
        print("\n🚀 启动机器人:")
        print("  python run_bot.py")
        print("  或")
        print("  python apps/app.py")
    else:
        print("⚠️ 部分测试失败，请检查上述错误信息")
        print("\n🔧 常见解决方案:")
        print("  1. 安装依赖: pixi install")
        print("  2. 检查模型文件路径")
        print("  3. 确保系统支持屏幕捕捉")

if __name__ == "__main__":
    main()
