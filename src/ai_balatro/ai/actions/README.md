# Balatro Action Module 

Balatro游戏的动作执行模块，支持通过位置数组控制牌的选择、出牌和弃牌操作。

## 功能特性

- 🃏 **位置数组控制**: 使用简单的数组控制牌的操作
- 🎯 **智能牌检测**: 自动检测和排序手牌位置
- 🖱️ **精确点击**: 基于YOLO检测结果的精确点击控制
- 🎬 **平滑鼠标移动**: 自然的鼠标移动动画，避免直接瞬移
- 🔄 **多种动作**: 支持出牌、弃牌、悬停等多种操作
- 🖼️ **可视化预览**: 操作前显示预览窗口确认动作
- ⚙️ **可配置参数**: 支持自定义鼠标移动速度和点击参数
- 🧪 **测试覆盖**: 完整的单元测试和集成测试

## 核心组件

### 1. ActionExecutor (动作执行器)
统一的游戏动作执行器，处理所有类型的游戏操作。

```python
from ai_balatro.core.yolo_detector import YOLODetector
from ai_balatro.core.screen_capture import ScreenCapture
from ai_balatro.ai.actions import ActionExecutor

# 初始化
detector = YOLODetector()
capture = ScreenCapture()
executor = ActionExecutor(detector, capture)
executor.initialize()
```

### 2. CardActionEngine (牌动作引擎)
专门处理牌相关的动作逻辑。

### 3. CardPositionDetector (牌位置检测器)
检测和排序手牌位置，按从左到右的顺序。

## 使用方法

### 基本用法

```python
# 出牌操作 - 选择前三张牌出牌
success = executor.execute_from_array([1, 1, 1, 0], "出前三张牌")

# 弃牌操作 - 弃掉前两张牌  
success = executor.execute_from_array([-1, -1, 0, 0], "弃掉前两张牌")

# 选择性出牌 - 出第1张和第3张牌
success = executor.execute_from_array([1, 0, 1, 0], "选择第1和第3张牌")

# 🆕 带可视化预览的操作 - 在执行前显示预览窗口
success = executor.execute_from_array(
    [1, 1, 1, 0], 
    "出前三张牌", 
    show_visualization=True  # 启用可视化
)
```

### 位置数组说明

位置数组中的每个数字代表对应位置牌的操作：

- `1`: 选择该位置的牌用于**出牌**
- `-1`: 选择该位置的牌用于**弃牌**  
- `0`: 不对该位置的牌进行操作

### 常用示例

```python
# 示例1: 出牌操作
[1, 1, 1, 0, 0]    # 选择前3张牌出牌
[1, 0, 1, 0, 1]    # 选择第1、3、5张牌出牌
[0, 1, 1, 1, 0]    # 选择第2、3、4张牌出牌

# 示例2: 弃牌操作  
[-1, -1, 0, 0, 0]  # 弃掉前2张牌
[-1, 0, 0, -1, 0]  # 弃掉第1张和第4张牌
[0, -1, -1, -1, 0] # 弃掉第2、3、4张牌
```

### 高级用法

```python
# 1. 使用ProcessingResult接口
result = executor.process({
    "positions": [1, 1, 0, 0],
    "description": "选择前两张牌"
})

if result.success:
    print("操作成功!")
    print(f"执行数据: {result.data}")
else:
    print(f"操作失败: {result.errors}")

# 2. 悬停查看牌的详情
result = executor.process({
    "function_call": {
        "name": "hover_card", 
        "arguments": {"card_index": 0, "duration": 2.0}
    }
})

# 3. 🆕 自定义鼠标移动参数
executor.set_mouse_animation_params(
    move_duration=0.8,      # 移动持续时间（秒）
    move_steps=30,          # 移动步数（越多越平滑）
    click_hold_duration=0.12 # 点击保持时间（秒）
)

# 4. 获取可用动作
actions = executor.get_available_actions()
for action in actions:
    print(f"{action['name']}: {action['description']}")
```

### 🎬 鼠标移动配置

新版本实现了平滑的鼠标移动动画，解决了直接瞬移导致点击失效的问题：

#### 预设配置:
```python
# 快速移动（适合快节奏操作）
executor.set_mouse_animation_params(0.3, 15, 0.08)

# 标准移动（默认，平衡速度和稳定性）  
executor.set_mouse_animation_params(0.5, 20, 0.1)

# 慢速移动（适合需要高精确度的场景）
executor.set_mouse_animation_params(0.8, 30, 0.12)
```

#### 参数说明:
- **move_duration**: 鼠标移动总时间（秒），建议0.2-2.0
- **move_steps**: 移动插值步数，建议10-50（越多越平滑但越慢）
- **click_hold_duration**: 点击按下保持时间（秒），建议0.05-0.3

## LLM函数调用支持

支持OpenAI风格的函数调用格式：

```python
# 选择牌的位置
{
    "name": "select_cards_by_position",
    "arguments": {
        "positions": [1, 1, 1, 0],
        "description": "选择前三张牌出牌"
    }
}

# 悬停查看牌
{
    "name": "hover_card",
    "arguments": {
        "card_index": 0,
        "duration": 1.5
    }
}
```

## 注意事项

⚠️ **重要限制**:
- 同一个数组中不能同时包含正数(1)和负数(-1)
- 数组长度会根据当前手牌数量自动调整
- 需要确保Balatro游戏窗口可见且未被遮挡

⚠️ **系统要求**:
- macOS需要授予"辅助功能"权限用于鼠标控制
- 需要先运行Balatro游戏并保持窗口可见
- YOLO模型文件需要正确配置

### 🖼️ 可视化功能

新增的可视化功能让你在执行操作前能够预览即将进行的操作：

#### 可视化特性:
- **🎯 颜色标识**: 不同颜色标记不同操作类型
  - 绿色边框: 选择出牌的牌  
  - 红色边框: 选择弃牌的牌
  - 灰色边框: 不操作的牌
- **📍 位置编号**: 每张牌中心显示位置编号 (0, 1, 2, ...)
- **📋 操作信息**: 显示操作类型和位置数组
- **⌨️ 交互控制**: 按任意键确认执行，ESC取消

#### 使用方法:
```python
# 启用可视化预览
success = executor.execute_from_array(
    [1, 1, 0, 0], 
    "选择前两张牌出牌", 
    show_visualization=True
)

# 预览窗口会显示:
# - 所有检测到的手牌
# - 即将点击的牌 (绿色或红色边框)
# - 操作说明和控制提示
```

## 运行演示

```bash
# 运行交互式演示（包含可视化和鼠标移动选项）
python examples/action_demo.py

# 运行可视化功能专门演示
python examples/visualization_example.py

# 🆕 运行平滑鼠标移动测试
python examples/smooth_mouse_test.py

# 运行测试
pytest tests/test_action_module.py -v
```

### 演示脚本说明:
- **action_demo.py**: 主要演示脚本，包含所有功能测试
- **visualization_example.py**: 专门演示可视化预览功能
- **smooth_mouse_test.py**: 🆕 专门测试平滑鼠标移动功能

## 扩展开发

### 添加新的动作类型

1. 在`schemas.py`中添加新的ActionType枚举
2. 在`GAME_ACTIONS`中添加函数调用schema
3. 在`ActionExecutor`中实现对应的处理方法

### 添加按钮点击功能

目前按钮点击功能标记为TODO，可以通过以下方式扩展：

1. 在`card_actions.py`中实现按钮检测逻辑
2. 添加按钮位置的YOLO训练数据
3. 完善`_execute_confirm_action`方法

## 作者

- RainbowBird
- Claude Code (Assistant)
