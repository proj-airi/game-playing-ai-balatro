# Balatro Game Detection System

A real-time screen detection system for the Balatro card game using YOLO object detection. The system can automatically detect and interact with game elements through screen capture and computer vision.

## Features

- **Real-time Detection**: Uses YOLO models to detect cards and game elements in real-time
- **Auto-click Functionality**: Automatically clicks on detected cards with configurable cooldown
- **Window Detection**: Automatically finds and focuses on Balatro game windows
- **Flexible Configuration**: YAML-based configuration system
- **Multiple Detection Modes**: Single detection and continuous detection modes
- **Cross-platform Support**: Works on macOS, Windows, and Linux

## Project Structure

```
apps/
├── config/                 # Configuration management
│   ├── config.yaml        # Main configuration file
│   └── settings.py        # Settings loader
├── core/                  # Core detection modules
│   ├── detection.py       # Detection data structures
│   ├── screen_capture.py  # Screen capture functionality
│   └── yolo_detector.py   # YOLO detection engine
├── services/              # Service layer
│   ├── auto_click_service.py    # Auto-click functionality
│   └── detection_service.py     # Main detection service
├── ui/                    # User interface
│   └── demo_app.py        # Demo application
├── utils/                 # Utility modules
│   ├── logger.py          # Logging utilities
│   └── path_utils.py      # Path handling utilities
├── main.py               # Main entry point
└── requirements.txt      # Python dependencies
```

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **macOS Additional Setup** (for automatic window detection):
   ```bash
   pip install pyobjc-framework-Quartz
   ```

3. **Model Files**: Ensure YOLO model files are available at one of the configured paths in `config/config.yaml`.

## Usage

### Basic Usage

Run the main application:
```bash
python main.py
```

The application will:
1. Auto-detect the Balatro game window
2. Ask if you want to enable auto-click functionality
3. Allow you to configure detection parameters
4. Choose between single or continuous detection mode

### Configuration

Edit `config/config.yaml` to customize:
- Model file paths
- Detection thresholds
- Auto-click settings
- UI preferences
- Logging configuration

### Detection Modes

1. **Single Detection Mode**: Capture and analyze one frame at a time
2. **Continuous Detection Mode**: Real-time detection with live preview

### Controls (Continuous Mode)

- `q` - Exit
- `s` - Save current frame
- `+/-` - Adjust confidence threshold
- `Space` - Pause/Resume
- `c` - Manual click trigger (if auto-click enabled)

## Configuration Options

### Model Configuration
```yaml
model:
  search_paths:
    - "../models/games-balatro-2024-yolo-entities-detection/model.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45
```

### Auto-click Configuration
```yaml
auto_click:
  enabled: false
  cooldown_seconds: 1.0
  card_keywords:
    - "card"
    - "joker"
    - "playing"
```

### Screen Capture Configuration
```yaml
screen_capture:
  default_fps: 10
  window_keywords:
    - "Balatro"
```

## API Usage

### Using the Detection Service

```python
from services.detection_service import DetectionService

# Initialize service
service = DetectionService(
    model_path="path/to/model.pt",
    enable_auto_click=True
)

# Select detection region
service.select_detection_region()

# Run single detection
service.run_single_detection()

# Run continuous detection
service.run_continuous_detection(fps=2)
```

### Using Individual Components

```python
from core.screen_capture import ScreenCapture
from core.yolo_detector import YOLODetector

# Screen capture
capture = ScreenCapture()
frame = capture.capture_once()

# YOLO detection
detector = YOLODetector("model.pt")
detections = detector.detect(frame)
```

## Logging

Logs are written to both console and file (`balatro_detection.log` by default). Log level and format can be configured in `config/config.yaml`.

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure model files exist at configured paths
2. **Window not detected**: Make sure Balatro is running and window title contains "Balatro"
3. **Permission errors**: On macOS, grant accessibility permissions to Terminal/IDE
4. **Import errors**: Install all required dependencies from `requirements.txt`

### Debug Mode

Enable debug logging by setting `logging.level: "DEBUG"` in the configuration file.

## Development

### Adding New Detection Classes

1. Update the classes file or `model.default_classes` in config
2. Retrain the YOLO model with new classes
3. Update visualization colors if needed

### Extending Auto-click Logic

Modify `services/auto_click_service.py` to implement custom clicking strategies.

## License

This project is for educational and research purposes. Please ensure compliance with game terms of service when using automated interaction features.
