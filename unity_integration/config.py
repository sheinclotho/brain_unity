"""
Configuration Management for TwinBrain Framework
================================================

Centralized configuration management to avoid hardcoded values.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any
import logging


@dataclass
class ServerConfig:
    """WebSocket server configuration"""
    host: str = "127.0.0.1"  # Changed from 0.0.0.0 for security
    port: int = 8765
    timeout: int = 30
    max_connections: int = 10
    enable_ssl: bool = False
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None


@dataclass
class ModelConfig:
    """Model configuration"""
    model_path: Optional[str] = None
    device: str = "cpu"
    n_regions: int = 200
    atlas_name: str = "Schaefer200"
    demo_mode: bool = False


@dataclass
class OutputConfig:
    """Output and export configuration"""
    output_dir: str = "unity_project/brain_data/model_output"
    cache_dir: str = "unity_project/brain_data/cache"
    export_format: str = "json"
    compress: bool = False
    max_frames: int = 50


@dataclass
class StimulationConfig:
    """Stimulation simulation configuration"""
    default_amplitude: float = 0.5
    default_pattern: str = "sine"
    default_frequency: float = 10.0
    default_duration: int = 50
    max_amplitude: float = 10.0
    min_amplitude: float = 0.01


@dataclass
class VisualizationConfig:
    """Visualization settings"""
    activity_threshold: float = 0.3
    connection_threshold: float = 0.5
    max_connections: int = 10000
    show_connections: bool = True
    color_scheme: str = "hot"


@dataclass
class TwinBrainConfig:
    """Main TwinBrain configuration"""
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    stimulation: StimulationConfig = field(default_factory=StimulationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate()
    
    def _validate(self):
        """Validate configuration values"""
        # Validate server
        if not 1 <= self.server.port <= 65535:
            raise ValueError(f"Invalid port: {self.server.port}")
        
        # Validate model
        if self.model.n_regions <= 0:
            raise ValueError(f"Invalid n_regions: {self.model.n_regions}")
        
        # Validate stimulation
        if not 0.0 <= self.stimulation.default_amplitude <= self.stimulation.max_amplitude:
            raise ValueError(f"Invalid default_amplitude: {self.stimulation.default_amplitude}")
        
        # Validate visualization
        if not 0.0 <= self.visualization.activity_threshold <= 1.0:
            raise ValueError(f"Invalid activity_threshold: {self.visualization.activity_threshold}")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TwinBrainConfig':
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            TwinBrainConfig instance
        
        Example:
            >>> config = TwinBrainConfig.from_file("config.json")
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TwinBrainConfig':
        """
        Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
        
        Returns:
            TwinBrainConfig instance
        """
        # Extract nested configurations
        server_data = data.get('server', {})
        model_data = data.get('model', {})
        output_data = data.get('output', {})
        stimulation_data = data.get('stimulation', {})
        visualization_data = data.get('visualization', {})
        
        return cls(
            server=ServerConfig(**server_data),
            model=ModelConfig(**model_data),
            output=OutputConfig(**output_data),
            stimulation=StimulationConfig(**stimulation_data),
            visualization=VisualizationConfig(**visualization_data),
            log_level=data.get('log_level', 'INFO'),
            log_file=data.get('log_file')
        )
    
    @classmethod
    def from_env(cls) -> 'TwinBrainConfig':
        """
        Load configuration from environment variables.
        
        Environment variables should be prefixed with TWINBRAIN_
        
        Example:
            TWINBRAIN_SERVER_HOST=0.0.0.0
            TWINBRAIN_SERVER_PORT=8080
            TWINBRAIN_MODEL_PATH=/path/to/model.pt
        
        Returns:
            TwinBrainConfig instance
        """
        config = cls()
        
        # Server configuration from environment
        if host := os.getenv('TWINBRAIN_SERVER_HOST'):
            config.server.host = host
        if port := os.getenv('TWINBRAIN_SERVER_PORT'):
            config.server.port = int(port)
        
        # Model configuration from environment
        if model_path := os.getenv('TWINBRAIN_MODEL_PATH'):
            config.model.model_path = model_path
        if device := os.getenv('TWINBRAIN_MODEL_DEVICE'):
            config.model.device = device
        if n_regions := os.getenv('TWINBRAIN_MODEL_N_REGIONS'):
            config.model.n_regions = int(n_regions)
        
        # Output configuration from environment
        if output_dir := os.getenv('TWINBRAIN_OUTPUT_DIR'):
            config.output.output_dir = output_dir
        
        # Log level from environment
        if log_level := os.getenv('TWINBRAIN_LOG_LEVEL'):
            config.log_level = log_level
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'server': asdict(self.server),
            'model': asdict(self.model),
            'output': asdict(self.output),
            'stimulation': asdict(self.stimulation),
            'visualization': asdict(self.visualization),
            'log_level': self.log_level,
            'log_file': self.log_file
        }
    
    def save(self, config_path: str):
        """
        Save configuration to JSON file.
        
        Args:
            config_path: Path to save configuration
        
        Example:
            >>> config = TwinBrainConfig()
            >>> config.save("config.json")
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def setup_logging(self):
        """
        Setup logging based on configuration.
        
        Example:
            >>> config = TwinBrainConfig()
            >>> config.setup_logging()
        """
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        handlers = [logging.StreamHandler()]
        if self.log_file:
            handlers.append(logging.FileHandler(self.log_file))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )


def create_default_config(output_path: str = "config.json"):
    """
    Create a default configuration file.
    
    Args:
        output_path: Path to save default configuration
    
    Example:
        >>> create_default_config("my_config.json")
    """
    config = TwinBrainConfig()
    config.save(output_path)
    print(f"Default configuration saved to: {output_path}")


def merge_configs(base_config: TwinBrainConfig, override_config: Dict[str, Any]) -> TwinBrainConfig:
    """
    Merge two configurations with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration dictionary
    
    Returns:
        Merged configuration
    
    Example:
        >>> base = TwinBrainConfig()
        >>> override = {"server": {"port": 9000}}
        >>> merged = merge_configs(base, override)
    """
    base_dict = base_config.to_dict()
    
    # Deep merge
    def deep_merge(base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged_dict = deep_merge(base_dict, override_config)
    return TwinBrainConfig.from_dict(merged_dict)


# Example usage
if __name__ == "__main__":
    # Create default config
    config = TwinBrainConfig()
    print("Default configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save to file
    config.save("config_example.json")
    
    # Load from file
    loaded_config = TwinBrainConfig.from_file("config_example.json")
    print("\nLoaded configuration successfully")
    
    # Setup logging
    config.setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded and logging setup complete")
