#!/usr/bin/env python3
"""
测试窗口检测功能

Author: RainbowBird
"""

from screen_capture import ScreenCapture

def main():
    """测试窗口检测"""
    print("🔍 测试小丑牌窗口检测功能...")
    
    # 创建屏幕捕捉器
    capture = ScreenCapture()
    
    # 首先列出所有窗口
    print("\n📋 列出所有窗口:")
    capture.list_all_windows()
    
    print("\n🎯 尝试检测小丑牌窗口...")
    
    # 主动调用窗口检测
    detection_success = capture._detect_balatro_window()
    print(f"检测结果: {'成功' if detection_success else '失败'}")
    
    # 获取窗口信息
    window_info = capture.get_window_info()
    if window_info:
        print("✅ 检测到小丑牌窗口:")
        print(f"   窗口名称: {window_info.get('name', 'N/A')}")
        print(f"   应用程序: {window_info.get('owner', 'N/A')}")
        bounds = window_info.get('bounds', {})
        print(f"   位置: ({bounds.get('X', 0)}, {bounds.get('Y', 0)})")
        print(f"   尺寸: {bounds.get('Width', 0)} x {bounds.get('Height', 0)}")
        
        # 测试捕捉区域
        region = capture.get_capture_region()
        if region:
            print(f"   捕捉区域: {region}")
        
        # 测试截图
        print("\n📸 测试屏幕捕捉...")
        frame = capture.capture_once()
        if frame is not None:
            print(f"✅ 捕捉成功，图像尺寸: {frame.shape}")
        else:
            print("❌ 捕捉失败")
    else:
        print("❌ 未检测到小丑牌窗口")
        print("💡 请确保:")
        print("   1. 小丑牌游戏已启动")
        print("   2. 游戏窗口可见且未被遮挡")
        print("   3. 窗口标题包含 'Balatro' 关键词")
        print("   4. 游戏不是在全屏模式下运行")

if __name__ == "__main__":
    main()
