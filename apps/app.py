#!/usr/bin/env python3
"""
屏幕检测演示程序
实时捕捉屏幕并使用YOLO模型进行检测标注

Author: RainbowBird
"""

import os
import time
import cv2
from typing import Optional, List

from screen_capture import ScreenCapture
from yolo_detector import YOLODetector, Detection

# 自动点击功能
from pynput.mouse import Button
from pynput import mouse


class ScreenDetectionDemo:
    """屏幕检测演示类"""
    
    def __init__(self, model_path: str, use_onnx: bool = False, auto_click: bool = False):
        """
        初始化演示程序
        
        Args:
            model_path: YOLO模型路径
            use_onnx: 是否使用ONNX模型
            auto_click: 是否启用自动点击功能
        """
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.auto_click = auto_click
        
        # 初始化模块
        print("🤖 初始化屏幕检测演示...")
        
        # 初始化屏幕捕捉
        print("📸 初始化屏幕捕捉...")
        self.screen_capture = ScreenCapture()
        
        # 初始化YOLO检测器
        print("🎯 初始化YOLO检测器...")
        self.yolo_detector = YOLODetector(model_path, use_onnx=use_onnx)
        
        # 检测参数
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        # 自动点击相关
        self.mouse_controller = mouse.Controller() if auto_click else None
        self.last_click_time = 0
        self.click_cooldown = 2.0  # 点击冷却时间（秒）
        self.last_clicked_card = None  # 记录上次点击的牌，避免重复点击
        
        # 统计信息
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        print("✅ 初始化完成")
        if auto_click:
            print("🖱️ 自动点击功能已启用")
    
    def set_detection_params(self, confidence: float = 0.5, iou: float = 0.45):
        """设置检测参数"""
        self.confidence_threshold = confidence
        self.iou_threshold = iou
        print(f"🎯 检测参数: 置信度={confidence}, IoU={iou}")
    
    def set_click_cooldown(self, cooldown: float = 1.0):
        """设置点击冷却时间"""
        self.click_cooldown = cooldown
        print(f"🖱️ 点击冷却时间: {cooldown}秒")
    
    def get_screen_size(self):
        """获取屏幕尺寸"""
        try:
            # 优先使用 pyobjc 获取屏幕尺寸（macOS）
            import Quartz
            main_display = Quartz.CGMainDisplayID()
            screen_width = Quartz.CGDisplayPixelsWide(main_display)
            screen_height = Quartz.CGDisplayPixelsHigh(main_display)
            return screen_width, screen_height
        except ImportError:
            try:
                # 备用方案：使用 mss 获取屏幕尺寸
                monitor = self.screen_capture.sct.monitors[0]  # 主显示器信息
                return monitor['width'], monitor['height']
            except Exception:
                # 最后的备用方案：返回常见的屏幕尺寸
                return 1920, 1080
        except Exception:
            # 如果 pyobjc 方法失败，尝试其他方法
            try:
                monitor = self.screen_capture.sct.monitors[0]
                return monitor['width'], monitor['height']
            except Exception:
                return 1920, 1080
    
    def find_first_card(self, detections: List[Detection]) -> Optional[Detection]:
        """
        找到第一张牌（最左边的牌）
        
        Args:
            detections: 检测结果列表
            
        Returns:
            Optional[Detection]: 第一张牌的检测结果，如果没有找到则返回None
        """
        # 过滤出卡牌类型的检测结果
        card_keywords = ['card', '牌', 'joker', 'playing']
        card_detections = []
        
        for det in detections:
            class_name_lower = det.class_name.lower()
            if any(keyword in class_name_lower for keyword in card_keywords):
                card_detections.append(det)
        
        if not card_detections:
            return None
        
        # 按x坐标排序，找到最左边的牌
        card_detections.sort(key=lambda d: d.bbox[0])  # 按x1坐标排序
        return card_detections[0]
    
    def auto_click_first_card(self, detections: List[Detection]) -> bool:
        """
        自动点击第一张牌
        
        Args:
            detections: 检测结果列表
            
        Returns:
            bool: 是否成功点击
        """
        if not self.auto_click or not self.mouse_controller:
            return False
        
        # 检查冷却时间
        current_time = time.time()
        if current_time - self.last_click_time < self.click_cooldown:
            return False
        
        # 找到第一张牌
        first_card = self.find_first_card(detections)
        if not first_card:
            return False
        
        # 检查是否是同一张牌（避免重复点击）
        card_signature = f"{first_card.class_name}_{first_card.bbox[0]}_{first_card.bbox[1]}"
        if self.last_clicked_card == card_signature:
            return False
        
        # 计算点击位置（牌的中心）
        center_x, center_y = first_card.center
        
        # 获取窗口信息，转换为屏幕坐标
        capture_region = self.screen_capture.get_capture_region()
        if capture_region:
            # 使用捕捉区域的坐标进行转换
            screen_x = capture_region['left'] + center_x
            screen_y = capture_region['top'] + center_y
        else:
            print("❌ 无法获取捕捉区域信息")
            return False
        
        try:
            # 执行点击
            print(f"🖱️ 准备点击第一张牌: {first_card.class_name}")
            print(f"   检测坐标: ({center_x}, {center_y})")
            print(f"   屏幕坐标: ({screen_x}, {screen_y})")
            print(f"   捕捉区域: {capture_region}")
            
            self.mouse_controller.position = (screen_x, screen_y)
            time.sleep(0.1)  # 短暂延迟确保鼠标移动到位
            self.mouse_controller.click(Button.left, 1)
            
            self.last_click_time = current_time
            self.last_clicked_card = card_signature  # 记录已点击的牌
            print("✅ 点击完成")
            return True
            
        except Exception as e:
            print(f"❌ 自动点击失败: {e}")
            return False
    
    def select_region(self) -> bool:
        """选择检测区域（现在自动检测窗口）"""
        print("🎯 自动检测小丑牌窗口...")
        return self.screen_capture.select_region_interactive()
    
    def run_single_detection(self, save_result: bool = True) -> bool:
        """运行单次检测"""
        print("📸 捕捉屏幕...")
        
        # 捕捉屏幕
        frame = self.screen_capture.capture_once()
        if frame is None:
            print("❌ 屏幕捕捉失败")
            return False
        
        print(f"✅ 捕捉成功，图像尺寸: {frame.shape}")
        
        # 运行检测
        print("🔍 运行YOLO检测...")
        detections = self.yolo_detector.detect(
            frame,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold
        )
        
        print(f"🎯 检测到 {len(detections)} 个对象:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det.class_name} (置信度: {det.confidence:.3f}) 位置: {det.bbox}")
        
        # 自动点击第一张牌
        if self.auto_click and detections:
            clicked = self.auto_click_first_card(detections)
            if clicked:
                print("✅ 已自动点击第一张牌")
        
        # 可视化结果
        vis_frame = self.yolo_detector.visualize_detections(frame, detections)
        
        # 添加统计信息
        info_text = [
            f"检测对象: {len(detections)}",
            f"置信度阈值: {self.confidence_threshold}",
            f"IoU阈值: {self.iou_threshold}",
            f"模型: {'ONNX' if self.use_onnx else 'PyTorch'}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(vis_frame, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示结果（设置窗口位置避免与游戏窗口重叠）
        window_name = "小丑牌检测结果"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # 获取游戏窗口信息来计算合适的显示位置
        window_info = self.screen_capture.get_window_info()
        if window_info:
            bounds = window_info['bounds']
            game_x = int(bounds['X'])
            game_y = int(bounds['Y'])
            game_width = int(bounds['Width'])
            game_height = int(bounds['Height'])
            
            # 获取屏幕尺寸
            screen_width, screen_height = self.get_screen_size()
            print(f"📺 屏幕尺寸: {screen_width}x{screen_height}")
            
            # 将检测窗口放在游戏窗口右侧，如果空间不够则放在下方
            if game_x + game_width + 400 < screen_width:
                # 右侧有足够空间
                display_x = game_x + game_width + 20
                display_y = game_y
            else:
                # 右侧空间不够，放在下方
                display_x = game_x
                display_y = game_y + game_height + 20
            
            cv2.moveWindow(window_name, display_x, display_y)
            cv2.resizeWindow(window_name, 600, 450)  # 设置合适的窗口大小
            print(f"🖼️ 检测窗口位置: ({display_x}, {display_y}), 游戏窗口: ({game_x}, {game_y}) {game_width}x{game_height}")
        
        cv2.imshow(window_name, vis_frame)
        
        # 保存结果
        if save_result:
            timestamp = int(time.time())
            original_filename = f"screen_capture_{timestamp}.png"
            detection_filename = f"detection_result_{timestamp}.png"
            
            cv2.imwrite(original_filename, frame)
            cv2.imwrite(detection_filename, vis_frame)
            
            print("💾 结果已保存:")
            print(f"  原始图像: {original_filename}")
            print(f"  检测结果: {detection_filename}")
        
        print("\n按任意键继续，按 'q' 退出...")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        return key != ord('q')
    
    def run_continuous_detection(self, fps: int = 2) -> None:
        """运行连续检测"""
        print(f"🚀 开始连续检测 (FPS: {fps})...")
        print("控制键:")
        print("  'q' - 退出")
        print("  's' - 保存当前帧")
        print("  '+' - 提高置信度阈值")
        print("  '-' - 降低置信度阈值")
        print("  空格 - 暂停/继续")
        if self.auto_click:
            print("  'c' - 手动触发点击第一张牌")
        
        frame_time = 1.0 / fps
        paused = False
        
        while True:
            if not paused:
                loop_start = time.time()
                
                # 捕捉屏幕
                frame = self.screen_capture.capture_once()
                if frame is None:
                    continue
                
                self.frame_count += 1
                
                # 运行检测
                detections = self.yolo_detector.detect(
                    frame,
                    confidence_threshold=self.confidence_threshold,
                    iou_threshold=self.iou_threshold
                )
                
                self.detection_count += len(detections)
                
                # 注意：连续模式下不自动点击，只在按 'c' 键时手动触发点击
                
                # 可视化结果
                vis_frame = self.yolo_detector.visualize_detections(frame, detections)
                
                # 添加实时信息
                runtime = time.time() - self.start_time
                avg_fps = self.frame_count / runtime if runtime > 0 else 0
                avg_detections = self.detection_count / self.frame_count if self.frame_count > 0 else 0
                
                auto_click_status = "开启" if self.auto_click else "关闭"
                info_text = [
                    f"检测对象: {len(detections)}",
                    f"置信度: {self.confidence_threshold:.2f}",
                    f"平均FPS: {avg_fps:.1f}",
                    f"平均检测数: {avg_detections:.1f}",
                    f"总帧数: {self.frame_count}",
                    f"自动点击: {auto_click_status}",
                    "空格:暂停 q:退出 s:保存 +/-:调整置信度" + (" c:点击" if self.auto_click else "")
                ]
                
                for i, text in enumerate(info_text):
                    color = (0, 255, 0) if i < 4 else (255, 255, 255)
                    cv2.putText(vis_frame, text, (10, 30 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                
                # 显示结果（设置窗口位置避免与游戏窗口重叠）
                window_name = "小丑牌实时检测"
                
                # 只在第一次创建窗口时设置位置
                if self.frame_count == 1:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    
                    # 获取游戏窗口信息来计算合适的显示位置
                    window_info = self.screen_capture.get_window_info()
                    if window_info:
                        bounds = window_info['bounds']
                        game_x = int(bounds['X'])
                        game_y = int(bounds['Y'])
                        game_width = int(bounds['Width'])
                        game_height = int(bounds['Height'])
                        
                        # 将检测窗口放在游戏窗口右侧，如果空间不够则放在下方
                        screen_width, screen_height = self.get_screen_size()
                        print(f"📺 屏幕尺寸: {screen_width}x{screen_height}")
                        
                        if game_x + game_width + 400 < screen_width:
                            # 右侧有足够空间
                            display_x = game_x + game_width + 20
                            display_y = game_y
                        else:
                            # 右侧空间不够，放在下方
                            display_x = game_x
                            display_y = game_y + game_height + 20
                        
                        cv2.moveWindow(window_name, display_x, display_y)
                        cv2.resizeWindow(window_name, 800, 600)  # 设置合适的窗口大小
                        print(f"🖼️ 实时检测窗口位置: ({display_x}, {display_y}), 游戏窗口: ({game_x}, {game_y}) {game_width}x{game_height}")
                
                cv2.imshow(window_name, vis_frame)
                
                # 控制帧率
                elapsed = time.time() - loop_start
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                timestamp = int(time.time())
                filename = f"realtime_detection_{timestamp}.png"
                cv2.imwrite(filename, vis_frame)
                print(f"💾 已保存: {filename}")
            elif key == ord('+') or key == ord('='):
                # 提高置信度
                self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                print(f"🎯 置信度阈值: {self.confidence_threshold:.2f}")
            elif key == ord('-'):
                # 降低置信度
                self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                print(f"🎯 置信度阈值: {self.confidence_threshold:.2f}")
            elif key == ord(' '):
                # 暂停/继续
                paused = not paused
                status = "暂停" if paused else "继续"
                print(f"⏸️ {status}")
            elif key == ord('c') and self.auto_click:
                # 手动触发点击
                if not paused:
                    frame = self.screen_capture.capture_once()
                    if frame is not None:
                        detections = self.yolo_detector.detect(
                            frame,
                            confidence_threshold=self.confidence_threshold,
                            iou_threshold=self.iou_threshold
                        )
                        if detections:
                            # 临时重置点击限制，允许手动点击
                            old_last_click_time = self.last_click_time
                            old_last_clicked_card = self.last_clicked_card
                            self.last_click_time = 0
                            self.last_clicked_card = None
                            
                            clicked = self.auto_click_first_card(detections)
                            if clicked:
                                print("🖱️ 手动点击成功")
                            else:
                                print("❌ 手动点击失败")
                                # 恢复之前的状态
                                self.last_click_time = old_last_click_time
                                self.last_clicked_card = old_last_clicked_card
                        else:
                            print("❌ 未检测到卡牌")
                else:
                    print("⚠️ 请先恢复检测（按空格键）")
        
        cv2.destroyAllWindows()
        
        # 显示统计信息
        runtime = time.time() - self.start_time
        print("\n📊 检测统计:")
        print(f"  运行时间: {runtime:.1f}秒")
        print(f"  总帧数: {self.frame_count}")
        print(f"  总检测数: {self.detection_count}")
        print(f"  平均FPS: {self.frame_count / runtime:.1f}")
        print(f"  平均检测数/帧: {self.detection_count / self.frame_count:.1f}")


def find_model_path() -> Optional[str]:
    """查找可用的模型文件"""
    model_paths = [
        "../runs/v2-balatro-entities-2000-epoch/weights/best.pt",
        "../models/games-balatro-2024-yolo-entities-detection/model.pt",
        "../runs/v2-balatro-entities/weights/best.pt",
        "runs/v2-balatro-entities-2000-epoch/weights/best.pt",
        "models/games-balatro-2024-yolo-entities-detection/model.pt"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            return path
    
    return None


def main():
    """主函数"""
    print("🃏 小丑牌屏幕检测演示")
    print("=" * 50)
    
    # 查找模型文件
    model_path = find_model_path()
    if not model_path:
        print("❌ 未找到YOLO模型文件")
        print("\n请确保模型文件存在于以下位置之一:")
        print("  - ../runs/v2-balatro-entities-2000-epoch/weights/best.pt")
        print("  - ../models/games-balatro-2024-yolo-entities-detection/model.pt")
        return
    
    print(f"✅ 找到模型: {model_path}")
    
    # 询问是否启用自动点击
    print("\n🖱️ 自动点击设置:")
    auto_click_input = input("是否启用自动点击第一张牌? (y/N): ").strip().lower()
    auto_click = auto_click_input in ['y', 'yes', '是']
    
    # 创建演示程序
    try:
        demo = ScreenDetectionDemo(model_path, use_onnx=False, auto_click=auto_click)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 自动检测小丑牌窗口
    if not demo.select_region():
        print("❌ 未检测到小丑牌窗口")
        print("💡 请确保:")
        print("   1. 小丑牌游戏已启动")
        print("   2. 游戏窗口可见且未被遮挡")
        print("   3. 窗口标题包含 'Balatro' 关键词")
        return
    
    # 设置自动点击参数
    if auto_click:
        print("\n🖱️ 自动点击参数设置:")
        cooldown_input = input("点击冷却时间 (秒, 默认1.0): ").strip()
        if cooldown_input:
            try:
                cooldown = float(cooldown_input)
                demo.set_click_cooldown(cooldown)
            except ValueError:
                print("⚠️ 无效输入，使用默认值")
    
    # 设置检测参数
    print("\n🎯 设置检测参数...")
    confidence = input("置信度阈值 (0.1-0.9, 默认0.5): ").strip()
    if confidence:
        try:
            demo.set_detection_params(confidence=float(confidence))
        except ValueError:
            print("⚠️ 无效输入，使用默认值")
    
    # 选择运行模式
    print("\n🚀 选择运行模式:")
    print("  1. 单次检测")
    print("  2. 连续检测")
    
    mode = input("请选择 (1/2, 默认2): ").strip()
    
    if mode == "1":
        # 单次检测模式
        print("\n📸 单次检测模式")
        while True:
            if not demo.run_single_detection():
                break
    else:
        # 连续检测模式
        print("\n🎥 连续检测模式")
        fps = input("检测帧率 (1-10, 默认2): ").strip()
        if fps:
            try:
                fps = int(fps)
                fps = max(1, min(10, fps))
            except ValueError:
                fps = 2
        else:
            fps = 2
        
        demo.run_continuous_detection(fps=fps)
    
    print("\n👋 检测演示结束")


if __name__ == "__main__":
    main()
