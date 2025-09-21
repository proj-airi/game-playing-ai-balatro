# 小丑牌游戏AI机器人使用指南

## 概述

这是一个基于YOLO目标检测和大语言模型的小丑牌(Balatro)游戏自动化AI机器人。系统包含以下核心模块：

- **屏幕捕捉模块**: 实时捕捉游戏界面
- **YOLO检测模块**: 识别游戏中的各种卡牌和元素
- **游戏状态分析模块**: 解析检测结果为游戏状态
- **AI决策模块**: 使用大模型进行策略推理
- **自动操作模块**: 执行鼠标点击等操作

## 快速开始

### 1. 环境准备

确保已安装所有依赖：

```bash
# 使用pixi安装依赖
pixi install

# 或者使用pip
pip install ultralytics opencv-python pillow pyautogui pynput mss openai anthropic
```

### 2. 模型准备

确保YOLO模型文件存在：
- 默认路径: `runs/v2-balatro-entities-2000-epoch/weights/best.pt`
- 或者: `models/games-balatro-2024-yolo-entities-detection/model.pt`

### 3. 基本使用

```bash
# 基本启动（需要手动确认每个操作）
python run_bot.py

# 使用OpenAI API（需要设置API密钥）
python run_bot.py --ai-provider openai --api-key YOUR_API_KEY

# 使用Anthropic Claude
python run_bot.py --ai-provider anthropic --api-key YOUR_API_KEY

# 自动模式（危险，会自动执行操作）
python run_bot.py --auto-confirm --no-safety
```

## 详细配置

### 命令行参数

#### 模型配置
- `--model PATH`: 指定YOLO模型路径
- `--onnx`: 使用ONNX模型（更快的推理）

#### AI配置
- `--ai-provider {openai,anthropic,local}`: AI提供商
- `--api-key KEY`: AI API密钥
- `--model-name NAME`: AI模型名称

#### 检测配置
- `--confidence FLOAT`: 置信度阈值 (默认: 0.5)
- `--iou FLOAT`: IoU阈值 (默认: 0.45)

#### 运行配置
- `--fps INT`: 检测帧率 (默认: 2)
- `--decision-interval FLOAT`: 决策间隔秒数 (默认: 3.0)
- `--max-actions INT`: 每分钟最大操作数 (默认: 20)

#### 安全配置
- `--no-safety`: 禁用安全检查
- `--auto-confirm`: 自动确认操作（危险）

#### 调试配置
- `--no-display`: 不显示检测结果窗口
- `--save-screenshots`: 保存截图
- `--quiet`: 静默模式

#### 配置文件
- `--config FILE`: 从配置文件加载设置
- `--save-config FILE`: 保存当前配置到文件

### 配置文件示例

创建 `my_config.json`:

```json
{
  "yolo_model_path": "path/to/your/model.pt",
  "use_onnx": false,
  "ai_provider": "openai",
  "ai_model_name": "gpt-4o-mini",
  "confidence_threshold": 0.6,
  "iou_threshold": 0.4,
  "fps": 3,
  "decision_interval": 2.0,
  "max_actions_per_minute": 30,
  "enable_safety": true,
  "require_confirmation": false,
  "show_detection": true,
  "save_screenshots": false,
  "log_decisions": true
}
```

然后使用：
```bash
python run_bot.py --config my_config.json
```

## 使用流程

### 1. 启动机器人

```bash
python run_bot.py --api-key YOUR_OPENAI_API_KEY
```

### 2. 选择游戏区域

启动后，机器人会要求你选择游戏窗口区域：
1. 会显示当前屏幕截图
2. 用鼠标拖拽选择游戏窗口
3. 按回车确认，ESC取消

### 3. 运行控制

机器人启动后，可以使用以下命令：
- `p`: 暂停/恢复机器人
- `s`: 显示统计信息
- `q`: 退出机器人
- `Ctrl+C`: 强制退出
- `Ctrl+Shift+Q`: 紧急停止（全局热键）

### 4. 操作确认

如果启用了操作确认（默认），机器人会在执行每个操作前询问：
```
🤔 请确认操作: play_cards
   目标: 2 张卡牌
   推理: 打出一对，可以获得较好的分数
确认执行? (y/n, 默认n):
```

## 安全注意事项

### ⚠️ 重要警告

1. **游戏窗口**: 确保正确选择游戏窗口区域，避免误操作其他应用
2. **紧急停止**: 记住 `Ctrl+Shift+Q` 可以随时紧急停止
3. **操作确认**: 首次使用建议保持操作确认开启
4. **频率限制**: 默认限制每分钟20个操作，防止过于频繁

### 安全功能

- **安全区域**: 只在指定的游戏区域内操作
- **频率限制**: 防止操作过于频繁
- **紧急停止**: 全局热键立即停止所有操作
- **操作确认**: 每个操作前需要用户确认

## 故障排除

### 常见问题

1. **模型文件不存在**
   ```
   ❌ YOLO模型文件不存在: path/to/model.pt
   ```
   解决：检查模型路径，或重新训练模型

2. **API密钥错误**
   ```
   ❌ AI决策失败: Invalid API key
   ```
   解决：检查API密钥是否正确设置

3. **检测结果为空**
   - 检查游戏窗口是否正确选择
   - 调整置信度阈值 `--confidence 0.3`
   - 确保游戏界面清晰可见

4. **操作执行失败**
   - 检查游戏窗口是否在前台
   - 确保鼠标位置安全
   - 检查是否有其他程序干扰

### 调试模式

启用详细日志：
```bash
python run_bot.py --save-screenshots --log-decisions
```

这会：
- 保存每帧的截图和检测结果
- 记录所有AI决策过程
- 便于分析问题

## 性能优化

### 提高检测速度

1. **使用ONNX模型**:
   ```bash
   python run_bot.py --onnx
   ```

2. **降低检测帧率**:
   ```bash
   python run_bot.py --fps 1
   ```

3. **调整检测阈值**:
   ```bash
   python run_bot.py --confidence 0.6 --iou 0.5
   ```

### 提高决策质量

1. **增加决策间隔**:
   ```bash
   python run_bot.py --decision-interval 5.0
   ```

2. **使用更强的AI模型**:
   ```bash
   python run_bot.py --model-name gpt-4
   ```

## 扩展开发

### 添加新的卡牌类型

1. 在 `data/labelled/v2-balatro-entities/classes.txt` 中添加新类别
2. 重新训练YOLO模型
3. 在 `game_state.py` 中更新 `CardType` 枚举

### 自定义AI策略

1. 修改 `ai_decision.py` 中的 `_build_decision_prompt` 方法
2. 添加新的决策类型到 `_parse_ai_response` 方法
3. 在 `auto_control.py` 中实现对应的操作

### 添加新的操作类型

1. 在 `auto_control.py` 中添加新的 `ActionType`
2. 实现对应的执行方法
3. 在 `_decision_to_actions` 中添加转换逻辑

## 许可证

MIT License - 详见 LICENSE 文件

## 作者

RainbowBird - Project AIRI Team

---

**免责声明**: 此工具仅用于学习和研究目的。使用自动化工具可能违反游戏服务条款，请自行承担风险。
