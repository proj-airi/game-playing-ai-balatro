"""
YOLO检测模块
使用训练好的YOLO模型检测小丑牌游戏中的各种卡牌和元素

Author: RainbowBird
"""

import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from ultralytics import YOLO
import onnxruntime as ort


class Detection:
    """检测结果数据类"""
    
    def __init__(self, class_id: int, class_name: str, confidence: float, 
                 bbox: Tuple[int, int, int, int]):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        
    @property
    def center(self) -> Tuple[int, int]:
        """获取边界框中心点"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        """获取边界框面积"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def __repr__(self):
        return f"Detection({self.class_name}, {self.confidence:.3f}, {self.bbox})"


class YOLODetector:
    """YOLO检测器，支持PyTorch和ONNX模型"""
    
    def __init__(self, model_path: str, use_onnx: bool = False):
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.model = None
        self.class_names = []
        
        # 加载类别名称
        self._load_class_names()
        
        # 加载模型
        self._load_model()
    
    def _load_class_names(self) -> None:
        """加载类别名称"""
        # 从项目中的classes.txt加载
        classes_file = "/home/neko/Git/github.com/proj-airi/game-playing-ai-balatro/data/labelled/v2-balatro-entities/classes.txt"
        
        if os.path.exists(classes_file):
            with open(classes_file, 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # 默认类别
            self.class_names = [
                'card_description', 'card_pack', 'joker_card', 'planet_card',
                'poker_card_back', 'poker_card_description', 'poker_card_front',
                'poker_card_stack', 'spectral_card', 'tarot_card'
            ]
        
        print(f"加载了 {len(self.class_names)} 个类别: {self.class_names}")
    
    def _load_model(self) -> None:
        """加载YOLO模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        if self.use_onnx:
            # 加载ONNX模型
            self.model = ort.InferenceSession(self.model_path)
            print(f"已加载ONNX模型: {self.model_path}")
        else:
            # 加载PyTorch模型
            self.model = YOLO(self.model_path)
            print(f"已加载PyTorch模型: {self.model_path}")
    
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5,
               iou_threshold: float = 0.45) -> List[Detection]:
        """
        检测图像中的对象
        
        Args:
            image: 输入图像 (BGR格式)
            confidence_threshold: 置信度阈值
            iou_threshold: IoU阈值用于NMS
            
        Returns:
            检测结果列表
        """
        if self.use_onnx:
            return self._detect_onnx(image, confidence_threshold, iou_threshold)
        else:
            return self._detect_pytorch(image, confidence_threshold, iou_threshold)
    
    def _detect_pytorch(self, image: np.ndarray, confidence_threshold: float,
                       iou_threshold: float) -> List[Detection]:
        """使用PyTorch模型进行检测"""
        # 运行推理
        results = self.model(image, conf=confidence_threshold, iou=iou_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    class_id = class_ids[i]
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    confidence = float(confidences[i])
                    bbox = tuple(map(int, boxes[i]))  # (x1, y1, x2, y2)
                    
                    detection = Detection(class_id, class_name, confidence, bbox)
                    detections.append(detection)
        
        return detections
    
    def _detect_onnx(self, image: np.ndarray, confidence_threshold: float,
                    iou_threshold: float) -> List[Detection]:
        """使用ONNX模型进行检测"""
        # 预处理图像
        input_tensor = self._preprocess_onnx(image)
        
        # 运行推理
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_tensor})
        
        # 后处理
        detections = self._postprocess_onnx(outputs[0], image.shape, 
                                          confidence_threshold, iou_threshold)
        
        return detections
    
    def _preprocess_onnx(self, image: np.ndarray) -> np.ndarray:
        """ONNX模型预处理"""
        # 调整图像大小到模型输入尺寸 (通常是640x640)
        input_size = 640
        h, w = image.shape[:2]
        
        # 保持宽高比的resize
        scale = min(input_size / w, input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # 创建正方形图像并居中放置
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        top = (input_size - new_h) // 2
        left = (input_size - new_w) // 2
        padded[top:top+new_h, left:left+new_w] = resized
        
        # 转换为模型输入格式
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # 添加batch维度
        
        return input_tensor
    
    def _postprocess_onnx(self, outputs: np.ndarray, original_shape: Tuple[int, int, int],
                         confidence_threshold: float, iou_threshold: float) -> List[Detection]:
        """ONNX模型后处理"""
        # outputs shape: (1, num_detections, 85) for YOLO
        # 85 = 4 (bbox) + 1 (objectness) + 80 (classes) for COCO, 
        # 对于我们的模型应该是 4 + 1 + num_classes
        
        detections = []
        predictions = outputs[0]  # 移除batch维度
        
        h, w = original_shape[:2]
        input_size = 640
        scale = min(input_size / w, input_size / h)
        
        for prediction in predictions:
            # 解析预测结果
            x_center, y_center, width, height = prediction[:4]
            objectness = prediction[4]
            class_scores = prediction[5:]
            
            # 过滤低置信度检测
            if objectness < confidence_threshold:
                continue
            
            # 找到最高分数的类别
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            final_confidence = objectness * class_score
            
            if final_confidence < confidence_threshold:
                continue
            
            # 转换坐标到原图尺寸
            x_center = (x_center - (input_size - w * scale) / 2) / scale
            y_center = (y_center - (input_size - h * scale) / 2) / scale
            width = width / scale
            height = height / scale
            
            # 转换为x1, y1, x2, y2格式
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            detection = Detection(class_id, class_name, final_confidence, (x1, y1, x2, y2))
            detections.append(detection)
        
        # 应用NMS
        detections = self._apply_nms(detections, iou_threshold)
        
        return detections
    
    def _apply_nms(self, detections: List[Detection], iou_threshold: float) -> List[Detection]:
        """应用非极大值抑制"""
        if not detections:
            return []
        
        # 按置信度排序
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            # 保留置信度最高的检测
            current = detections.pop(0)
            keep.append(current)
            
            # 移除与当前检测IoU过高的检测
            detections = [det for det in detections 
                         if self._calculate_iou(current.bbox, det.bbox) < iou_threshold]
        
        return keep
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_detections(self, image: np.ndarray, detections: List[Detection],
                           show_confidence: bool = True) -> np.ndarray:
        """可视化检测结果"""
        vis_image = image.copy()
        
        # 定义颜色 (BGR格式)
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红色
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 紫色
            (255, 165, 0),  # 橙色
            (0, 128, 255),  # 天蓝色
            (255, 192, 203) # 粉色
        ]
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = colors[detection.class_id % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = detection.class_name
            if show_confidence:
                label += f" {detection.confidence:.2f}"
            
            # 计算文本尺寸
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 绘制文本背景
            cv2.rectangle(vis_image, (x1, y1 - text_height - baseline - 5), 
                         (x1 + text_width, y1), color, -1)
            
            # 绘制文本
            cv2.putText(vis_image, label, (x1, y1 - baseline - 2), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return vis_image


def test_yolo_detector():
    """测试YOLO检测器"""
    # 使用最新训练的模型
    model_path = "/home/neko/Git/github.com/proj-airi/game-playing-ai-balatro/runs/v2-balatro-entities-2000-epoch/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    # 创建检测器
    detector = YOLODetector(model_path, use_onnx=False)
    
    # 测试图像
    test_image_path = "/home/neko/Git/github.com/proj-airi/game-playing-ai-balatro/test/testdata"
    
    if os.path.exists(test_image_path):
        # 查找测试图像
        for filename in os.listdir(test_image_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(test_image_path, filename)
                
                # 加载图像
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                print(f"检测图像: {filename}")
                
                # 运行检测
                detections = detector.detect(image, confidence_threshold=0.3)
                
                print(f"检测到 {len(detections)} 个对象:")
                for det in detections:
                    print(f"  {det}")
                
                # 可视化结果
                vis_image = detector.visualize_detections(image, detections)
                
                # 显示结果
                cv2.imshow(f"检测结果 - {filename}", vis_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                break
    else:
        print("测试数据目录不存在")


if __name__ == "__main__":
    test_yolo_detector()
