#!/usr/bin/env python3
"""
屏幕检测演示程序
实时捕捉屏幕并使用YOLO模型进行检测标注

Author: RainbowBird
"""

import os
import time
import cv2
import numpy as np
from typing import Optional

from screen_capture import ScreenCapture
from yolo_detector import YOLODetector


class ScreenDetectionDemo:
    """屏幕检测演示类"""
    
    def __init__(self, model_path: str, use_onnx: bool = False):
        """
        初始化演示程序
        
        Args:
            model_path: YOLO模型路径
            use_onnx: 是否使用ONNX模型
        """
        self.model_path = model_path
        self.use_onnx = use_onnx
        
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
        
        # 统计信息
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        print("✅ 初始化完成")
    
    def set_detection_params(self, confidence: float = 0.5, iou: float = 0.45):
        """设置检测参数"""
        self.confidence_threshold = confidence
        self.iou_threshold = iou
        print(f"🎯 检测参数: 置信度={confidence}, IoU={iou}")
    
    def select_region(self) -> bool:
        """选择检测区域"""
        print("🎯 请选择要检测的屏幕区域...")
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
        
        # 显示结果
        cv2.imshow("小丑牌检测结果", vis_frame)
        
        # 保存结果
        if save_result:
            timestamp = int(time.time())
            original_filename = f"screen_capture_{timestamp}.png"
            detection_filename = f"detection_result_{timestamp}.png"
            
            cv2.imwrite(original_filename, frame)
            cv2.imwrite(detection_filename, vis_frame)
            
            print(f"💾 结果已保存:")
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
                
                # 可视化结果
                vis_frame = self.yolo_detector.visualize_detections(frame, detections)
                
                # 添加实时信息
                runtime = time.time() - self.start_time
                avg_fps = self.frame_count / runtime if runtime > 0 else 0
                avg_detections = self.detection_count / self.frame_count if self.frame_count > 0 else 0
                
                info_text = [
                    f"检测对象: {len(detections)}",
                    f"置信度: {self.confidence_threshold:.2f}",
                    f"平均FPS: {avg_fps:.1f}",
                    f"平均检测数: {avg_detections:.1f}",
                    f"总帧数: {self.frame_count}",
                    "空格:暂停 q:退出 s:保存 +/-:调整置信度"
                ]
                
                for i, text in enumerate(info_text):
                    color = (0, 255, 0) if i < 4 else (255, 255, 255)
                    cv2.putText(vis_frame, text, (10, 30 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                
                # 显示结果
                cv2.imshow("小丑牌实时检测", vis_frame)
                
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
        
        cv2.destroyAllWindows()
        
        # 显示统计信息
        runtime = time.time() - self.start_time
        print(f"\n📊 检测统计:")
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
    
    # 创建演示程序
    try:
        demo = ScreenDetectionDemo(model_path, use_onnx=False)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 选择检测区域
    if not demo.select_region():
        print("❌ 未选择检测区域，退出")
        return
    
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
