#!/usr/bin/env python3
"""
小丑牌AI机器人启动脚本
简化的启动入口，提供基本的配置选项

Author: RainbowBird
"""

import argparse
import os

from ai_decision import AIProvider
from balatro_ai_bot import BalatroAIBot, BotConfig


def create_default_config() -> BotConfig:
    """创建默认配置"""
    config = BotConfig()

    # 设置模型路径
    model_paths = [
        "../runs/v2-balatro-entities-2000-epoch/weights/best.pt",
        "../models/games-balatro-2024-yolo-entities-detection/model.pt",
        "../runs/v2-balatro-entities/weights/best.pt",
        "runs/v2-balatro-entities-2000-epoch/weights/best.pt",
        "models/games-balatro-2024-yolo-entities-detection/model.pt"
    ]

    for path in model_paths:
        if os.path.exists(path):
            config.yolo_model_path = path
            break

    return config


def main():
    parser = argparse.ArgumentParser(description="小丑牌游戏AI机器人")

    # 模型配置
    parser.add_argument("--model", type=str, help="YOLO模型路径")
    parser.add_argument("--onnx", action="store_true", help="使用ONNX模型")

    # AI配置
    parser.add_argument(
        "--ai-provider",
        choices=["openai", "anthropic", "local"],
        default="openai",
        help="AI提供商",
    )
    parser.add_argument("--api-key", type=str, help="AI API密钥")
    parser.add_argument("--model-name", type=str, help="AI模型名称")

    # 检测配置
    parser.add_argument("--confidence", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU阈值")

    # 运行配置
    parser.add_argument("--fps", type=int, default=2, help="检测帧率")
    parser.add_argument(
        "--decision-interval", type=float, default=3.0, help="决策间隔（秒）"
    )
    parser.add_argument("--max-actions", type=int, default=20, help="每分钟最大操作数")

    # 安全配置
    parser.add_argument("--no-safety", action="store_true", help="禁用安全检查")
    parser.add_argument(
        "--auto-confirm", action="store_true", help="自动确认操作（危险）"
    )

    # 调试配置
    parser.add_argument("--no-display", action="store_true", help="不显示检测结果")
    parser.add_argument("--save-screenshots", action="store_true", help="保存截图")
    parser.add_argument("--quiet", action="store_true", help="静默模式")

    # 配置文件
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--save-config", type=str, help="保存配置到文件")

    args = parser.parse_args()

    # 创建配置
    if args.config and os.path.exists(args.config):
        print(f"📄 从文件加载配置: {args.config}")
        config = BalatroAIBot.load_config(args.config)
    else:
        config = create_default_config()

    # 应用命令行参数
    if args.model:
        config.yolo_model_path = args.model
    if args.onnx:
        config.use_onnx = True
    if args.ai_provider:
        config.ai_provider = AIProvider(args.ai_provider)
    if args.api_key:
        config.ai_api_key = args.api_key
    if args.model_name:
        config.ai_model_name = args.model_name

    config.confidence_threshold = args.confidence
    config.iou_threshold = args.iou
    config.fps = args.fps
    config.decision_interval = args.decision_interval
    config.max_actions_per_minute = args.max_actions

    if args.no_safety:
        config.enable_safety = False
    if args.auto_confirm:
        config.require_confirmation = False
    if args.no_display:
        config.show_detection = False
    if args.save_screenshots:
        config.save_screenshots = True
    if args.quiet:
        config.log_decisions = False

    # 保存配置（如果指定）
    if args.save_config:
        temp_bot = BalatroAIBot(config)
        temp_bot.save_config(args.save_config)
        print(f"✅ 配置已保存到: {args.save_config}")
        return

    # 检查模型文件
    if not os.path.exists(config.yolo_model_path):
        print(f"❌ YOLO模型文件不存在: {config.yolo_model_path}")
        print("\n可用的模型路径:")
        for path in [
            "../runs/v2-balatro-entities-2000-epoch/weights/best.pt",
            "../models/games-balatro-2024-yolo-entities-detection/model.pt",
        ]:
            status = "✅" if os.path.exists(path) else "❌"
            print(f"  {status} {path}")
        return

    # 显示配置信息
    print("🃏 小丑牌游戏AI机器人")
    print("=" * 60)
    print(f"📁 模型路径: {config.yolo_model_path}")
    print(f"🤖 AI提供商: {config.ai_provider.value}")
    print(f"🎯 置信度阈值: {config.confidence_threshold}")
    print(f"📺 检测帧率: {config.fps} FPS")
    print(f"🧠 决策间隔: {config.decision_interval}秒")
    print(f"🛡️ 安全模式: {'启用' if config.enable_safety else '禁用'}")
    print(f"✋ 需要确认: {'是' if config.require_confirmation else '否'}")
    print("=" * 60)

    # 安全警告
    if not config.enable_safety or not config.require_confirmation:
        print("⚠️  警告: 安全检查已禁用或自动确认已启用")
        print("⚠️  机器人将自动执行操作，请确保游戏窗口正确")
        print("⚠️  按 Ctrl+Shift+Q 可以紧急停止")

        if not args.auto_confirm:
            response = input("\n继续? (y/N): ").strip().lower()
            if response not in ["y", "yes", "是"]:
                print("操作取消")
                return

    # 创建并启动机器人
    bot = BalatroAIBot(config)

    try:
        if bot.start():
            print("\n🚀 机器人已启动")
            print("\n控制命令:")
            print("  p - 暂停/恢复")
            print("  s - 显示统计信息")
            print("  q - 退出")
            print("  Ctrl+C - 强制退出")
            print("  Ctrl+Shift+Q - 紧急停止")

            # 主循环
            while bot.running:
                try:
                    cmd = input().strip().lower()

                    if cmd == "q":
                        break
                    elif cmd == "p":
                        if bot.state.value == "running":
                            bot.pause()
                        elif bot.state.value == "paused":
                            bot.resume()
                    elif cmd == "s":
                        stats = bot.get_statistics()
                        print(f"\n📊 统计信息:")
                        print(f"  运行时间: {stats['runtime_seconds']:.1f}秒")
                        print(f"  处理帧数: {stats['total_frames']}")
                        print(f"  检测次数: {stats['total_detections']}")
                        print(f"  决策次数: {stats['total_decisions']}")
                        print(f"  执行操作: {stats['total_actions']}")
                        print(f"  平均FPS: {stats['fps']:.1f}")
                        print(f"  错误次数: {len(stats['errors'])}")
                        if stats["errors"]:
                            print(f"  最近错误: {stats['errors'][-1]}")

                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
        else:
            print("❌ 机器人启动失败")

    except KeyboardInterrupt:
        print("\n收到中断信号")

    finally:
        bot.stop()
        print("\n👋 机器人已停止")


if __name__ == "__main__":
    main()
