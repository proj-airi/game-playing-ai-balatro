# Balatro Action Module 

Balatro游戏的动作执行模块，支持通过位置数组控制牌的选择、出牌和弃牌操作。

## 功能特性

- 🃏 **位置数组控制**: 使用简单的数组控制牌的操作
- 🎯 **智能牌检测**: 自动检测和排序手牌位置
- 🤖 **双YOLO模型**: 使用entities模型识别卡牌，UI模型识别按钮
- 🖱️ **精确点击**: 基于YOLO检测结果的精确点击控制
- 🎬 **平滑鼠标移动**: 自然的鼠标移动动画，避免直接瞬移
- 🔄 **多种动作**: 支持出牌、弃牌、悬停等多种操作
- 📝 **卡牌描述捕获**: 自动悬停捕获卡牌描述文本（OCR）
- 🖼️ **可视化预览**: 操作前显示预览窗口确认动作
- 🎯 **智能按钮检测**: UI模型自动检测并点击按钮（无需固定坐标）
- ⚙️ **可配置参数**: 支持自定义鼠标移动速度和点击参数
- 🪟 **窗口焦点管理**: 自动管理游戏窗口焦点
- 🧪 **测试覆盖**: 完整的单元测试和集成测试

## 核心组件

### 1. ActionExecutor (动作执行器)
统一的游戏动作执行器，处理所有类型的游戏操作。

```python
from ai_balatro.core.multi_yolo_detector import MultiYOLODetector
from ai_balatro.core.screen_capture import ScreenCapture
from ai_balatro.ai.actions import ActionExecutor

# 初始化（推荐使用双模型系统）
multi_detector = MultiYOLODetector()
capture = ScreenCapture()
executor = ActionExecutor(screen_capture=capture, multi_detector=multi_detector)
executor.initialize()
```

### 2. CardActionEngine (牌动作引擎)
专门处理牌相关的动作逻辑，包括卡牌点击、描述捕获等。

### 3. ButtonDetector (按钮检测器)
使用UI模型智能检测游戏按钮（出牌、弃牌、商店等）。

### 4. CardPositionDetector (牌位置检测器)
检测和排序手牌位置，按从左到右的顺序。

### 5. MouseController (鼠标控制器)
提供平滑鼠标移动、点击和窗口焦点管理功能。

### 6. DetectionVisualizer (可视化工具)
显示YOLO检测结果的可视化窗口，用于调试。

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

# 3. 获取卡牌描述（自动悬停捕获OCR文本）
card_descriptions = executor.card_engine.get_card_descriptions()
for i, card_desc in enumerate(card_descriptions):
    print(f"Card {i}: {card_desc['description_text']}")
    
# 格式化为LLM提示词
formatted_prompt = executor.card_engine.format_card_descriptions_for_llm(card_descriptions)

# 4. 自定义鼠标移动参数
executor.set_mouse_animation_params(
    move_duration=0.3,      # 移动持续时间（秒）
    move_steps=15,          # 移动步数（越多越平滑）
    click_hold_duration=0.08 # 点击保持时间（秒）
)

# 5. 配置窗口焦点管理
executor.set_window_focus_settings(
    enable=True,           # 启用窗口焦点管理
    method='auto'          # auto/click/applescript
)

# 6. 配置卡牌悬停行为
executor.card_engine.set_card_hover_settings(
    enable_hovering=True,        # 启用卡牌悬停
    hover_before_action=True,    # 动作前自动悬停捕获描述
    save_debug_images=False      # 保存调试图片
)

# 7. 获取可用动作
actions = executor.get_available_actions()
for action in actions:
    print(f"{action['name']}: {action['description']}")

# 8. 独立按钮点击
result = executor.process({
    "function_call": {
        "name": "click_button",
        "arguments": {"button_type": "shop"}
    }
})
```

### 🎬 鼠标移动配置

实现了平滑的鼠标移动动画（使用ease-in-out缓动函数），解决了直接瞬移导致点击失效的问题：

#### 预设配置:
```python
# 快速移动（适合快节奏操作，当前默认）
executor.set_mouse_animation_params(0.3, 15, 0.08)

# 标准移动（平衡速度和稳定性）  
executor.set_mouse_animation_params(0.5, 20, 0.1)

# 慢速移动（适合需要高精确度的场景）
executor.set_mouse_animation_params(0.8, 30, 0.12)
```

#### 参数说明:
- **move_duration**: 鼠标移动总时间（秒），建议0.2-1.0
- **move_steps**: 移动插值步数，建议10-50（越多越平滑但越慢）
- **click_hold_duration**: 点击按下保持时间（秒），建议0.05-0.2

### 🪟 窗口焦点管理

自动管理游戏窗口焦点，确保点击操作生效：

#### 焦点处理方法:
```python
# 自动选择最佳方法（推荐）
executor.set_window_focus_settings(enable=True, method='auto')

# macOS AppleScript方式（macOS推荐）
executor.set_window_focus_settings(enable=True, method='applescript')

# 点击窗口标题栏激活（跨平台）
executor.set_window_focus_settings(enable=True, method='click')

# 禁用自动焦点管理
executor.set_window_focus_settings(enable=False)
```

#### 平台支持:
- **macOS**: 优先使用AppleScript激活，回退到点击激活
- **Windows/Linux**: 使用点击窗口标题栏激活

## LLM函数调用支持

支持OpenAI风格的函数调用格式，可直接集成到LLM Agent中：

```python
# 1. 选择牌的位置（出牌/弃牌）
{
    "name": "select_cards_by_position",
    "arguments": {
        "positions": [1, 1, 1, 0],
        "description": "选择前三张牌出牌"
    }
}

# 2. 悬停查看牌的详细信息
{
    "name": "hover_card",
    "arguments": {
        "card_index": 0,
        "duration": 1.5
    }
}

# 3. 点击游戏按钮
{
    "name": "click_button",
    "arguments": {
        "button_type": "shop"  # play/discard/skip/shop/next等
    }
}
```

### 可用的按钮类型:
- `play`: 出牌按钮
- `discard`: 弃牌按钮
- `skip`: 跳过按钮
- `shop`: 商店按钮
- `next`: 继续/下一步按钮
- `sort_hand_rank`: 按牌面排序
- `sort_hand_suits`: 按花色排序

## 📝 卡牌描述捕获

系统集成了智能卡牌描述捕获功能，可以自动悬停并使用OCR识别卡牌文本：

### 自动捕获（推荐）

默认情况下，系统会在执行出牌/弃牌操作前自动捕获所有卡牌描述：

```python
# execute_card_action 会自动返回卡牌描述
result = executor.card_engine.execute_card_action(
    positions=[1, 1, 0, 0],
    description="出前两张牌"
)

if result['success']:
    for i, card_desc in enumerate(result['card_descriptions']):
        if card_desc['description_detected']:
            print(f"Card {i}: {card_desc['description_text']}")
```

### 手动捕获

也可以单独获取卡牌描述，不执行动作：

```python
# 获取所有手牌描述
card_descriptions = executor.card_engine.get_card_descriptions()

# 格式化为LLM提示词
formatted_text = executor.card_engine.format_card_descriptions_for_llm(card_descriptions)
print(formatted_text)
```

### 配置选项

```python
# 配置卡牌悬停行为
executor.card_engine.set_card_hover_settings(
    enable_hovering=True,         # 启用卡牌悬停功能
    hover_before_action=True,     # 在动作执行前自动捕获描述
    save_debug_images=False       # 保存调试图片（用于OCR调试）
)
```

### 返回数据格式

```python
card_description = {
    'card_index': 0,                          # 卡牌位置索引
    'description_detected': True,             # 是否检测到描述
    'description_text': 'King of Hearts...',  # OCR识别的文本
    'tooltip_bbox': [x1, y1, x2, y2],        # 提示框位置
    'confidence': 0.95                        # 检测置信度
}
```

## 注意事项

⚠️ **重要限制**:
- 同一个数组中不能同时包含正数(1)和负数(-1)
- 数组长度会根据当前手牌数量自动调整
- 需要确保Balatro游戏窗口可见且未被遮挡
- 卡牌描述捕获需要一定时间（每张卡约0.5-1秒）

⚠️ **系统要求**:
- macOS需要授予"辅助功能"权限用于鼠标控制
- 需要先运行Balatro游戏并保持窗口可见
- 双YOLO模型文件需要正确配置（entities + UI模型）
- OCR功能需要额外的OCR引擎（用于卡牌描述捕获）

⚠️ **最佳实践**:
- 使用双模型系统（`MultiYOLODetector`）而非单一检测器
- 启用窗口焦点管理以确保点击生效
- 对于需要LLM决策的场景，启用自动卡牌描述捕获
- 使用可视化功能进行调试和验证
- 根据实际情况调整鼠标移动速度参数

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
# 运行交互式演示（包含双模型系统、可视化和鼠标移动选项）
pixi run python examples/action_demo.py

# 运行测试
pixi run pytest tests/test_action_module.py -v
```

### 演示脚本功能:
`action_demo.py` 包含以下演示功能：
- ✅ 智能出牌/弃牌（双模型系统）
- ✅ 卡牌悬停和描述捕获
- ✅ 自定义位置数组操作
- ✅ 双模型可视化预览
- ✅ UI模型按钮检测演示
- ✅ 平滑鼠标移动测试
- ✅ 鼠标移动速度配置

## 扩展开发

### 添加新的动作类型

1. 在`schemas.py`中添加新的ActionType枚举
2. 在`GAME_ACTIONS`中添加函数调用schema
3. 在`ActionExecutor`中实现对应的处理方法

### 添加新的按钮类型

按钮检测功能已完全实现，使用UI模型进行智能检测。添加新按钮类型：

1. 在`schemas.py`的`BUTTON_CONFIG`中添加按钮配置
2. 在`button_detector.py`的`button_class_map`中添加类名映射
3. 确保UI模型已训练识别该按钮类型

### 卡牌描述捕获

系统已集成`CardTooltipService`，可以自动悬停捕获卡牌描述：

- 使用OCR识别卡牌文本
- 支持自动格式化为LLM提示词
- 可配置是否在动作前自动捕获

## 作者

- RainbowBird
- Claude Code (Assistant)
