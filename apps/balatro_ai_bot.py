"""
小丑牌游戏AI机器人主控制器
协调屏幕捕捉、YOLO检测、游戏状态分析、AI决策和自动操作等所有模块

Author: RainbowBird
"""

import os
import time
import json
import threading
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import cv2
import numpy as np

from screen_capture import ScreenCapture
from yolo_detector import YOLODetector
from game_state import GameStateAnalyzer, GameState
from ai_decision import BalatroAI, AIProvider, Decision
from auto_control import AutoController


class BotState(Enum):
    """机器人状态枚举"""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class BotConfig:
    """机器人配置"""
    # 模型配置
    yolo_model_path: str = "/home/neko/Git/github.com/proj-airi/game-playing-ai-balatro/runs/v2-balatro-entities-2000-epoch/weights/best.pt"
    use_onnx: bool = False
    
    # AI配置
    ai_provider: AIProvider = AIProvider.OPENAI
    ai_api_key: Optional[str] = None
    ai_model_name: Optional[str] = None
    
    # 检测配置
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    
    # 运行配置
    fps: int = 2  # 检测帧率
    decision_interval: float = 3.0  # 决策间隔（秒）
    max_actions_per_minute: int = 20  # 每分钟最大操作数
    
    # 安全配置
    enable_safety: bool = True
    require_confirmation: bool = True  # 是否需要用户确认操作
    
    # 调试配置
    show_detection: bool = True
    save_screenshots: bool = False
    log_decisions: bool = True


class BalatroAIBot:
    """小丑牌游戏AI机器人"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.state = BotState.STOPPED
        
        # 核心模块
        self.screen_capture: Optional[ScreenCapture] = None
        self.yolo_detector: Optional[YOLODetector] = None
        self.game_analyzer: Optional[GameStateAnalyzer] = None
        self.ai_decision: Optional[BalatroAI] = None
        self.auto_controller: Optional[AutoController] = None
        
        # 运行时数据
        self.current_game_state: Optional[GameState] = None
        self.last_decision_time = 0
        self.action_count = 0
        self.action_reset_time = time.time()
        
        # 统计信息
        self.stats = {
            "total_frames": 0,
            "total_detections": 0,
            "total_decisions": 0,
            "total_actions": 0,
            "start_time": None,
            "errors": []
        }
        
        # 回调函数
        self.on_state_change: Optional[Callable] = None
        self.on_decision_made: Optional[Callable] = None
        self.on_action_executed: Optional[Callable] = None
        
        # 控制线程
        self.main_thread: Optional[threading.Thread] = None
        self.running = False
    
    def initialize(self) -> bool:
        """初始化所有模块"""
        try:
            self.state = BotState.INITIALIZING
            self._notify_state_change()
            
            print("🤖 初始化小丑牌AI机器人...")
            
            # 初始化屏幕捕捉
            print("📸 初始化屏幕捕捉模块...")
            self.screen_capture = ScreenCapture()
            
            # 初始化YOLO检测器
            print("🎯 初始化YOLO检测器...")
            if not os.path.exists(self.config.yolo_model_path):
                raise FileNotFoundError(f"YOLO模型文件不存在: {self.config.yolo_model_path}")
            
            self.yolo_detector = YOLODetector(
                self.config.yolo_model_path,
                use_onnx=self.config.use_onnx
            )
            
            # 初始化游戏状态分析器
            print("🎮 初始化游戏状态分析器...")
            self.game_analyzer = GameStateAnalyzer()
            
            # 初始化AI决策器
            print("🧠 初始化AI决策器...")
            self.ai_decision = BalatroAI(
                provider=self.config.ai_provider,
                api_key=self.config.ai_api_key,
                model_name=self.config.ai_model_name
            )
            
            # 初始化自动控制器
            print("🎮 初始化自动控制器...")
            self.auto_controller = AutoController()
            
            print("✅ 所有模块初始化完成")
            return True
            
        except Exception as e:
            self.state = BotState.ERROR
            self.stats["errors"].append(f"初始化失败: {str(e)}")
            print(f"❌ 初始化失败: {e}")
            return False
    
    def setup_game_region(self) -> bool:
        """设置游戏区域"""
        try:
            print("🎯 请选择游戏窗口区域...")
            
            if self.screen_capture.select_region_interactive():
                # 将游戏区域设置为安全区域
                if self.screen_capture.capture_region:
                    region = self.screen_capture.capture_region
                    self.auto_controller.set_game_region(
                        region["left"], region["top"],
                        region["width"], region["height"]
                    )
                print("✅ 游戏区域设置完成")
                return True
            else:
                print("❌ 游戏区域设置取消")
                return False
                
        except Exception as e:
            print(f"❌ 设置游戏区域失败: {e}")
            return False
    
    def start(self) -> bool:
        """启动机器人"""
        if self.state == BotState.RUNNING:
            print("⚠️ 机器人已在运行中")
            return True
        
        if not self.initialize():
            return False
        
        if not self.setup_game_region():
            return False
        
        try:
            self.running = True
            self.state = BotState.RUNNING
            self.stats["start_time"] = time.time()
            self._notify_state_change()
            
            # 启动主循环线程
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            print("🚀 小丑牌AI机器人已启动")
            print("按 Ctrl+Shift+Q 紧急停止")
            return True
            
        except Exception as e:
            self.state = BotState.ERROR
            self.stats["errors"].append(f"启动失败: {str(e)}")
            print(f"❌ 启动失败: {e}")
            return False
    
    def stop(self):
        """停止机器人"""
        print("🛑 正在停止机器人...")
        
        self.running = False
        self.state = BotState.STOPPED
        
        # 停止屏幕捕捉
        if self.screen_capture:
            self.screen_capture.stop_continuous_capture()
        
        # 紧急停止自动控制
        if self.auto_controller:
            self.auto_controller.emergency_stop()
        
        # 等待主线程结束
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=3.0)
        
        self._notify_state_change()
        print("✅ 机器人已停止")
    
    def pause(self):
        """暂停机器人"""
        if self.state == BotState.RUNNING:
            self.state = BotState.PAUSED
            self._notify_state_change()
            print("⏸️ 机器人已暂停")
    
    def resume(self):
        """恢复机器人"""
        if self.state == BotState.PAUSED:
            self.state = BotState.RUNNING
            self._notify_state_change()
            print("▶️ 机器人已恢复")
    
    def _main_loop(self):
        """主循环"""
        frame_time = 1.0 / self.config.fps
        
        while self.running:
            try:
                if self.state != BotState.RUNNING:
                    time.sleep(0.5)
                    continue
                
                loop_start = time.time()
                
                # 捕捉屏幕
                frame = self.screen_capture.capture_once()
                if frame is None:
                    continue
                
                self.stats["total_frames"] += 1
                
                # YOLO检测
                detections = self.yolo_detector.detect(
                    frame,
                    confidence_threshold=self.config.confidence_threshold,
                    iou_threshold=self.config.iou_threshold
                )
                
                self.stats["total_detections"] += len(detections)
                
                # 分析游戏状态
                screen_height, screen_width = frame.shape[:2]
                game_state = self.game_analyzer.analyze(detections, screen_width, screen_height)
                self.current_game_state = game_state
                
                # 显示检测结果（如果启用）
                if self.config.show_detection:
                    self._show_detection_result(frame, detections, game_state)
                
                # 保存截图（如果启用）
                if self.config.save_screenshots:
                    self._save_screenshot(frame, detections)
                
                # 检查是否需要做决策
                current_time = time.time()
                if current_time - self.last_decision_time >= self.config.decision_interval:
                    self._make_and_execute_decision(game_state)
                    self.last_decision_time = current_time
                
                # 控制帧率
                elapsed = time.time() - loop_start
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.stats["errors"].append(f"主循环错误: {str(e)}")
                print(f"❌ 主循环错误: {e}")
                time.sleep(1.0)  # 错误后等待一秒
    
    def _make_and_execute_decision(self, game_state: GameState):
        """做出决策并执行"""
        try:
            # 检查操作频率限制
            if not self._check_action_rate_limit():
                return
            
            # AI决策
            decision = self.ai_decision.make_decision(game_state)
            self.stats["total_decisions"] += 1
            
            if self.config.log_decisions:
                print(f"🧠 AI决策: {decision.action_type} (置信度: {decision.confidence:.2f})")
                print(f"   推理: {decision.reasoning}")
            
            # 通知决策回调
            if self.on_decision_made:
                self.on_decision_made(decision, game_state)
            
            # 检查是否需要用户确认
            if self.config.require_confirmation and decision.action_type != "skip":
                if not self._request_user_confirmation(decision):
                    print("⏭️ 用户取消操作")
                    return
            
            # 执行决策
            if decision.action_type != "skip":
                success = self.auto_controller.execute_decision(decision)
                if success:
                    self.stats["total_actions"] += 1
                    self.action_count += 1
                    print(f"✅ 操作执行成功")
                else:
                    print(f"❌ 操作执行失败")
                
                # 通知操作回调
                if self.on_action_executed:
                    self.on_action_executed(decision, success)
            
        except Exception as e:
            self.stats["errors"].append(f"决策执行错误: {str(e)}")
            print(f"❌ 决策执行错误: {e}")
    
    def _check_action_rate_limit(self) -> bool:
        """检查操作频率限制"""
        current_time = time.time()
        
        # 重置计数器（每分钟）
        if current_time - self.action_reset_time >= 60.0:
            self.action_count = 0
            self.action_reset_time = current_time
        
        # 检查是否超过限制
        if self.action_count >= self.config.max_actions_per_minute:
            print("⚠️ 操作频率达到限制，跳过此次决策")
            return False
        
        return True
    
    def _request_user_confirmation(self, decision: Decision) -> bool:
        """请求用户确认操作"""
        print(f"🤔 请确认操作: {decision.action_type}")
        print(f"   目标: {len(decision.target_cards)} 张卡牌")
        print(f"   推理: {decision.reasoning}")
        
        # 简单的控制台确认（实际应用中可以用GUI）
        try:
            response = input("确认执行? (y/n, 默认n): ").strip().lower()
            return response in ['y', 'yes', '是']
        except KeyboardInterrupt:
            return False
    
    def _show_detection_result(self, frame: np.ndarray, detections, game_state: GameState):
        """显示检测结果"""
        # 可视化检测结果
        vis_frame = self.yolo_detector.visualize_detections(frame, detections)
        
        # 添加游戏状态信息
        info_text = [
            f"手牌: {len(game_state.hand_region.cards)}",
            f"小丑牌: {len(game_state.joker_region.cards)}",
            f"商店: {len(game_state.shop_region.cards)}",
            f"检测: {len(detections)}",
            f"状态: {self.state.value}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(vis_frame, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("小丑牌AI机器人", vis_frame)
        
        # 按'q'退出显示
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop()
    
    def _save_screenshot(self, frame: np.ndarray, detections):
        """保存截图"""
        timestamp = int(time.time())
        filename = f"screenshot_{timestamp}.png"
        
        # 保存原始截图
        cv2.imwrite(filename, frame)
        
        # 保存带检测结果的截图
        vis_frame = self.yolo_detector.visualize_detections(frame, detections)
        vis_filename = f"detection_{timestamp}.png"
        cv2.imwrite(vis_filename, vis_frame)
    
    def _notify_state_change(self):
        """通知状态变化"""
        if self.on_state_change:
            self.on_state_change(self.state)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        current_time = time.time()
        runtime = current_time - self.stats["start_time"] if self.stats["start_time"] else 0
        
        return {
            **self.stats,
            "runtime_seconds": runtime,
            "current_state": self.state.value,
            "fps": self.stats["total_frames"] / runtime if runtime > 0 else 0,
            "decisions_per_minute": self.stats["total_decisions"] / (runtime / 60) if runtime > 0 else 0,
            "actions_per_minute": self.stats["total_actions"] / (runtime / 60) if runtime > 0 else 0,
            "current_game_state": self.current_game_state.to_dict() if self.current_game_state else None
        }
    
    def save_config(self, filename: str):
        """保存配置到文件"""
        config_dict = {
            "yolo_model_path": self.config.yolo_model_path,
            "use_onnx": self.config.use_onnx,
            "ai_provider": self.config.ai_provider.value,
            "ai_model_name": self.config.ai_model_name,
            "confidence_threshold": self.config.confidence_threshold,
            "iou_threshold": self.config.iou_threshold,
            "fps": self.config.fps,
            "decision_interval": self.config.decision_interval,
            "max_actions_per_minute": self.config.max_actions_per_minute,
            "enable_safety": self.config.enable_safety,
            "require_confirmation": self.config.require_confirmation,
            "show_detection": self.config.show_detection,
            "save_screenshots": self.config.save_screenshots,
            "log_decisions": self.config.log_decisions
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        print(f"配置已保存到: {filename}")
    
    @classmethod
    def load_config(cls, filename: str) -> 'BotConfig':
        """从文件加载配置"""
        with open(filename, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = BotConfig()
        config.yolo_model_path = config_dict.get("yolo_model_path", config.yolo_model_path)
        config.use_onnx = config_dict.get("use_onnx", config.use_onnx)
        config.ai_provider = AIProvider(config_dict.get("ai_provider", config.ai_provider.value))
        config.ai_model_name = config_dict.get("ai_model_name", config.ai_model_name)
        config.confidence_threshold = config_dict.get("confidence_threshold", config.confidence_threshold)
        config.iou_threshold = config_dict.get("iou_threshold", config.iou_threshold)
        config.fps = config_dict.get("fps", config.fps)
        config.decision_interval = config_dict.get("decision_interval", config.decision_interval)
        config.max_actions_per_minute = config_dict.get("max_actions_per_minute", config.max_actions_per_minute)
        config.enable_safety = config_dict.get("enable_safety", config.enable_safety)
        config.require_confirmation = config_dict.get("require_confirmation", config.require_confirmation)
        config.show_detection = config_dict.get("show_detection", config.show_detection)
        config.save_screenshots = config_dict.get("save_screenshots", config.save_screenshots)
        config.log_decisions = config_dict.get("log_decisions", config.log_decisions)
        
        return config


def main():
    """主函数"""
    print("🃏 小丑牌游戏AI机器人")
    print("=" * 50)
    
    # 创建配置
    config = BotConfig()
    
    # 检查模型文件
    if not os.path.exists(config.yolo_model_path):
        print(f"❌ YOLO模型文件不存在: {config.yolo_model_path}")
        print("请确保已训练好YOLO模型")
        return
    
    # 创建机器人
    bot = BalatroAIBot(config)
    
    # 设置回调函数
    def on_state_change(state):
        print(f"🔄 状态变化: {state.value}")
    
    def on_decision_made(decision, game_state):
        print(f"🎯 决策: {decision.action_type} (置信度: {decision.confidence:.2f})")
    
    def on_action_executed(decision, success):
        status = "成功" if success else "失败"
        print(f"⚡ 操作{status}: {decision.action_type}")
    
    bot.on_state_change = on_state_change
    bot.on_decision_made = on_decision_made
    bot.on_action_executed = on_action_executed
    
    try:
        # 启动机器人
        if bot.start():
            # 运行直到用户停止
            print("\n控制命令:")
            print("  p - 暂停/恢复")
            print("  s - 显示统计信息")
            print("  q - 退出")
            print("  Ctrl+C - 强制退出")
            
            while bot.running:
                try:
                    cmd = input().strip().lower()
                    
                    if cmd == 'q':
                        break
                    elif cmd == 'p':
                        if bot.state == BotState.RUNNING:
                            bot.pause()
                        elif bot.state == BotState.PAUSED:
                            bot.resume()
                    elif cmd == 's':
                        stats = bot.get_statistics()
                        print(json.dumps(stats, ensure_ascii=False, indent=2))
                    
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
        
    except KeyboardInterrupt:
        print("\n收到中断信号")
    
    finally:
        # 停止机器人
        bot.stop()
        
        # 显示最终统计
        stats = bot.get_statistics()
        print(f"\n📊 最终统计:")
        print(f"运行时间: {stats['runtime_seconds']:.1f}秒")
        print(f"处理帧数: {stats['total_frames']}")
        print(f"检测次数: {stats['total_detections']}")
        print(f"决策次数: {stats['total_decisions']}")
        print(f"执行操作: {stats['total_actions']}")
        print(f"错误次数: {len(stats['errors'])}")
        
        # 保存配置
        bot.save_config("bot_config.json")
        
        print("\n👋 再见！")


if __name__ == "__main__":
    main()
