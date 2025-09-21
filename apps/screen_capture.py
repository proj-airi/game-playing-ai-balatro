"""
屏幕捕捉模块
用于实时捕捉游戏界面，支持全屏和区域截图

Author: RainbowBird
"""

import time
import threading
from typing import Optional, Tuple, Callable
import numpy as np
import cv2
import mss
from PIL import Image


class ScreenCapture:
    """屏幕捕捉器，支持实时截图和区域选择"""
    
    def __init__(self):
        self.sct = mss.mss()
        self.capture_region: Optional[dict] = None
        self.is_capturing = False
        self.capture_thread: Optional[threading.Thread] = None
        self.frame_callback: Optional[Callable] = None
        self.fps = 10  # 默认10FPS
        
    def set_capture_region(self, x: int, y: int, width: int, height: int) -> None:
        """设置捕捉区域"""
        self.capture_region = {
            "top": y,
            "left": x,
            "width": width,
            "height": height
        }
        print(f"设置捕捉区域: x={x}, y={y}, width={width}, height={height}")
    
    def get_screen_size(self) -> Tuple[int, int]:
        """获取屏幕尺寸"""
        monitor = self.sct.monitors[1]  # 主显示器
        return monitor["width"], monitor["height"]
    
    def capture_once(self, region: Optional[dict] = None) -> np.ndarray:
        """单次截图"""
        if region is None:
            region = self.capture_region or self.sct.monitors[1]
        
        # 使用mss截图
        screenshot = self.sct.grab(region)
        
        # 转换为numpy数组 (BGRA -> BGR)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    def select_region_interactive(self) -> bool:
        """交互式选择捕捉区域"""
        print("请在屏幕上选择游戏窗口区域...")
        print("按下鼠标左键并拖拽选择区域，按ESC取消")
        
        # 全屏截图用于选择
        full_screen = self.capture_once(self.sct.monitors[1])
        
        # 创建选择窗口
        clone = full_screen.copy()
        cv2.namedWindow("选择游戏区域", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("选择游戏区域", 1200, 800)
        
        # 鼠标回调变量
        selecting = False
        start_point = None
        end_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal selecting, start_point, end_point, clone
            
            if event == cv2.EVENT_LBUTTONDOWN:
                selecting = True
                start_point = (x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE and selecting:
                clone = full_screen.copy()
                cv2.rectangle(clone, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("选择游戏区域", clone)
                
            elif event == cv2.EVENT_LBUTTONUP:
                selecting = False
                end_point = (x, y)
        
        cv2.setMouseCallback("选择游戏区域", mouse_callback)
        cv2.imshow("选择游戏区域", clone)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                cv2.destroyAllWindows()
                return False
            elif key == 13 and start_point and end_point:  # Enter键
                break
        
        cv2.destroyAllWindows()
        
        if start_point and end_point:
            # 计算选择区域
            x1, y1 = start_point
            x2, y2 = end_point
            
            # 确保坐标正确
            x = min(x1, x2)
            y = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # 缩放到实际屏幕坐标
            screen_width, screen_height = self.get_screen_size()
            display_height, display_width = full_screen.shape[:2]
            
            scale_x = screen_width / display_width
            scale_y = screen_height / display_height
            
            actual_x = int(x * scale_x)
            actual_y = int(y * scale_y)
            actual_width = int(width * scale_x)
            actual_height = int(height * scale_y)
            
            self.set_capture_region(actual_x, actual_y, actual_width, actual_height)
            return True
        
        return False
    
    def start_continuous_capture(self, callback: Callable[[np.ndarray], None], fps: int = 10) -> None:
        """开始连续捕捉"""
        if self.is_capturing:
            print("已经在捕捉中...")
            return
        
        self.frame_callback = callback
        self.fps = fps
        self.is_capturing = True
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print(f"开始连续捕捉，FPS: {fps}")
    
    def stop_continuous_capture(self) -> None:
        """停止连续捕捉"""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        print("停止连续捕捉")
    
    def _capture_loop(self) -> None:
        """捕捉循环"""
        frame_time = 1.0 / self.fps
        
        while self.is_capturing:
            start_time = time.time()
            
            try:
                frame = self.capture_once()
                if self.frame_callback:
                    self.frame_callback(frame)
            except Exception as e:
                print(f"捕捉帧时出错: {e}")
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def save_screenshot(self, filename: str, region: Optional[dict] = None) -> bool:
        """保存截图"""
        try:
            img = self.capture_once(region)
            cv2.imwrite(filename, img)
            print(f"截图已保存: {filename}")
            return True
        except Exception as e:
            print(f"保存截图失败: {e}")
            return False


def test_screen_capture():
    """测试屏幕捕捉功能"""
    capture = ScreenCapture()
    
    # 交互式选择区域
    if capture.select_region_interactive():
        print("区域选择成功")
        
        # 保存测试截图
        capture.save_screenshot("test_capture.png")
        
        # 显示实时捕捉
        def show_frame(frame):
            cv2.imshow("实时捕捉", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                capture.stop_continuous_capture()
        
        print("开始实时捕捉，按'q'退出")
        capture.start_continuous_capture(show_frame, fps=10)
        
        # 等待用户退出
        while capture.is_capturing:
            time.sleep(0.1)
        
        cv2.destroyAllWindows()
    else:
        print("区域选择取消")


if __name__ == "__main__":
    test_screen_capture()
