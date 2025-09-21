#!/usr/bin/env python3
"""
屏幕检测演示启动脚本
简单的启动入口

Author: RainbowBird
"""

import os
import sys
from pathlib import Path

# 添加apps目录到Python路径
apps_path = Path(__file__).parent / "apps"
sys.path.insert(0, str(apps_path))

def main():
    """主函数"""
    try:
        from screen_detection_demo import main as demo_main
        demo_main()
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保已安装所有依赖包")
        print("运行: pixi install")
    except Exception as e:
        print(f"❌ 运行失败: {e}")

if __name__ == "__main__":
    main()
