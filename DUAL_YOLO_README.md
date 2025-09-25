# 双YOLO模型系统集成完成 🎉

## 🚀 系统概述

现在已成功集成两个专用YOLO模型：

### 📦 模型分工
- **Entities模型** (`games-balatro-2024-yolo-entities-detection`)
  - 🃏 专门检测卡牌、小丑、塔罗等游戏实体
  - 📍 用于手牌识别和位置检测
  
- **UI模型** (`games-balatro-2024-yolo-ui-detection`)
  - 🔘 专门检测按钮和界面元素
  - 📊 包含20种按钮类型和13种UI数据元素

## 🎯 主要改进

### ✅ 核心功能
1. **MultiYOLODetector** - 统一管理双模型系统
2. **自动模型切换** - 卡牌检测用entities，按钮检测用UI
3. **完整流程自动化** - 选牌 → 检测按钮 → 自动点击确认
4. **可视化调试** - CV窗口显示检测结果，便于调试

### 🔧 技术特性
- **向后兼容** - 支持传统单模型和新双模型系统
- **错误恢复** - 模型加载失败时的优雅降级
- **性能优化** - 联合检测比串行执行更高效
- **跨平台** - 同时支持PyTorch和ONNX格式

## 📝 使用方法

### 🚀 快速开始
```bash
# 快速测试双模型检测
python quick_test_visualization.py

# 完整的双模型系统测试  
python test_dual_yolo_models.py
```

### 💻 代码示例
```python
from ai_balatro.core.multi_yolo_detector import MultiYOLODetector
from ai_balatro.core.screen_capture import ScreenCapture
from ai_balatro.ai.actions import ActionExecutor

# 创建双模型系统
multi_detector = MultiYOLODetector()
capture = ScreenCapture()
executor = ActionExecutor(screen_capture=capture, multi_detector=multi_detector)

# 完整出牌流程（卡牌检测 + 按钮检测 + 自动点击）
success = executor.execute_from_array(
    [1, 1, 1, 0],           # 选择前三张牌
    "自动出牌",             # 操作描述
    show_visualization=True  # 显示检测过程
)
```

### 🎮 支持的按钮类型
- `button_play` - 出牌按钮 ✅
- `button_discard` - 弃牌按钮 ✅  
- `button_skip` - 跳过按钮
- `button_purchase` - 购买按钮
- `button_sell` - 出售按钮
- 等20种按钮类型...

## 🧪 测试功能

### 📸 检测测试
1. **实体检测测试** - 验证卡牌识别效果
2. **UI检测测试** - 验证按钮识别效果  
3. **联合检测** - 验证双模型协同工作
4. **性能对比** - 测试检测速度和效率

### 🎯 完整流程测试
1. **出牌流程** - 选牌 → 检测出牌按钮 → 自动点击
2. **弃牌流程** - 选牌 → 检测弃牌按钮 → 自动点击
3. **可视化调试** - 实时显示检测过程

## 🔍 调试功能

### 🖼️ 可视化窗口
- **颜色编码**：绿色=卡牌，红色=按钮，黄色=UI，灰色=其他
- **详细标签**：显示类名和置信度
- **统计信息**：实时显示各类检测数量
- **交互控制**：按ESC关闭窗口

### 📊 检测信息
- **模型状态** - 显示可用模型和加载状态
- **类别列表** - 显示所有支持的检测类别
- **性能指标** - 显示检测时间和效率对比

## ⚡ 性能提升

### 🎯 检测精度
- **专业化模型** - 每个模型专注特定任务，检测更精确
- **减少误检** - 卡牌和按钮分开检测，避免混淆

### 🚀 执行效率  
- **并行检测** - 双模型可同时运行
- **智能缓存** - 避免重复检测同一帧
- **优化流程** - 只在需要时调用相应模型

## 🎉 使用效果

现在您可以：
1. **一键执行完整动作** - `executor.execute_from_array([1,1,0,0], "出牌")`
2. **实时查看检测过程** - 通过CV窗口观察YOLO检测结果  
3. **精确按钮识别** - 不再依赖固定坐标，智能识别按钮位置
4. **鲁棒性更强** - 适应不同分辨率和界面变化

双YOLO模型系统现已完全就绪！🚀

运行测试脚本查看效果：
```bash
python test_dual_yolo_models.py
```
