"""Configuration management for Balatro detection system."""

import os
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path


class Settings:
    """Configuration settings manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings from configuration file.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation key.
        
        Args:
            key: Configuration key in dot notation (e.g., 'model.confidence_threshold')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot notation key.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save current configuration to file."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
    
    # Model settings
    @property
    def model_search_paths(self) -> List[str]:
        """Get model search paths."""
        return self.get('model.search_paths', [])
    
    @property
    def model_classes_file(self) -> str:
        """Get model classes file path."""
        return self.get('model.classes_file', '')
    
    @property
    def model_default_classes(self) -> List[str]:
        """Get default model classes."""
        return self.get('model.default_classes', [])
    
    @property
    def model_onnx_path(self) -> str:
        """Get ONNX model path."""
        return self.get('model.onnx_path', '')
    
    # Detection settings
    @property
    def detection_confidence_threshold(self) -> float:
        """Get detection confidence threshold."""
        return self.get('detection.confidence_threshold', 0.5)
    
    @property
    def detection_iou_threshold(self) -> float:
        """Get detection IoU threshold."""
        return self.get('detection.iou_threshold', 0.45)
    
    @property
    def detection_input_size(self) -> int:
        """Get detection input size."""
        return self.get('detection.input_size', 640)
    
    # Screen capture settings
    @property
    def screen_capture_fps(self) -> int:
        """Get screen capture FPS."""
        return self.get('screen_capture.default_fps', 10)
    
    @property
    def screen_capture_window_keywords(self) -> List[str]:
        """Get window detection keywords."""
        return self.get('screen_capture.window_keywords', ['Balatro'])
    
    @property
    def screen_capture_excluded_apps(self) -> List[str]:
        """Get excluded applications."""
        return self.get('screen_capture.excluded_apps', [])
    
    # Auto click settings
    @property
    def auto_click_enabled(self) -> bool:
        """Get auto click enabled status."""
        return self.get('auto_click.enabled', False)
    
    @property
    def auto_click_cooldown(self) -> float:
        """Get auto click cooldown seconds."""
        return self.get('auto_click.cooldown_seconds', 1.0)
    
    @property
    def auto_click_card_keywords(self) -> List[str]:
        """Get card detection keywords for auto click."""
        return self.get('auto_click.card_keywords', ['card', 'joker', 'playing'])
    
    # UI settings
    @property
    def ui_colors(self) -> List[List[int]]:
        """Get UI colors for visualization."""
        return self.get('ui.colors', [[0, 255, 0]])
    
    @property
    def ui_window_names(self) -> Dict[str, str]:
        """Get UI window names."""
        return self.get('ui.window_names', {})
    
    @property
    def ui_window_sizes(self) -> Dict[str, List[int]]:
        """Get UI window sizes."""
        return self.get('ui.window_sizes', {})
    
    # Logging settings
    @property
    def logging_level(self) -> str:
        """Get logging level."""
        return self.get('logging.level', 'INFO')
    
    @property
    def logging_format(self) -> str:
        """Get logging format."""
        return self.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    @property
    def logging_file(self) -> str:
        """Get logging file path."""
        return self.get('logging.file', 'balatro_detection.log')
    
    @property
    def logging_max_bytes(self) -> int:
        """Get logging max bytes."""
        return self.get('logging.max_bytes', 10485760)
    
    @property
    def logging_backup_count(self) -> int:
        """Get logging backup count."""
        return self.get('logging.backup_count', 5)


# Global settings instance
settings = Settings()
