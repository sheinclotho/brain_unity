# TwinBrain - Brain Visualization Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Unity 2019.1+](https://img.shields.io/badge/unity-2019.1+-green.svg)](https://unity.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

TwinBrain is a comprehensive framework for visualizing brain imaging data in Unity 3D. It provides a complete workflow from FreeSurfer brain imaging data to interactive 3D visualization, including WebSocket server, virtual stimulation simulation, and real-time data export.

## ✨ Features

- 🧠 **FreeSurfer Integration**: Load and process FreeSurfer surface and annotation files
- 🎮 **Unity 3D Visualization**: Real-time brain visualization in Unity
- ⚡ **Virtual Stimulation**: Simulate brain stimulation effects (tACS, TMS, etc.)
- 🔄 **Real-time Communication**: WebSocket server for Unity-Python communication
- 📊 **Multi-modal Support**: fMRI, EEG, and connectivity data
- 🚀 **One-Click Setup**: Automated installation and configuration
- 📁 **Auto File Monitoring**: Automatically detect and load new data

## 📋 Requirements

### Software
- **Python**: 3.8 or higher
- **Unity**: 2019.1 or higher (2020 LTS or 2021 LTS recommended)
- **RAM**: 8GB minimum, 16GB recommended for large datasets

### Optional
- **FreeSurfer**: 7.0+ for generating brain surface 3D models
- **GPU**: Recommended for model training and inference

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/sheinclotho/brain_unity.git
cd brain_unity

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Create Unity Project

Create a new 3D project in Unity Hub:
- Project Name: `TwinBrainDemo` (or any name you prefer)
- Template: 3D
- Unity Version: 2019.1+

### 3. One-Click Setup

```bash
# Full installation (with FreeSurfer data)
python unity_one_click_install.py \
    --unity-project /path/to/TwinBrainDemo \
    --freesurfer-dir /path/to/freesurfer

# Basic installation (using default sphere models)
python unity_one_click_install.py \
    --unity-project /path/to/TwinBrainDemo
```

### 4. Unity Setup

1. Open the Unity project in Unity Hub
2. Wait for Newtonsoft.Json package to install (1-2 minutes)
3. Go to menu: **TwinBrain → Auto Setup Scene**
4. Check all options and click "Start Auto Setup"
5. Wait for completion (~30 seconds to 2 minutes)

### 5. Start Backend Server

```bash
# With trained model
python unity_startup.py --model results/hetero_gnn_trained.pt

# Demo mode (no model required)
python unity_startup.py --demo
```

### 6. Run in Unity

1. Press **Play** in Unity Editor
2. You should see "Connected to server" in the Console
3. Use the stimulation UI panel (bottom-left) to apply virtual stimulation

## 📖 Documentation

- **[User Guide (Chinese)](UNIFIED_GUIDE.md)**: Comprehensive guide from installation to usage
- **[Unity User Guide (Chinese)](Unity使用指南.md)**: Detailed Unity operations
- **[API Documentation](API_DOCUMENTATION.md)**: Complete API reference
- **[Project Specification (Chinese)](项目规范说明书.md)**: Project standards and conventions
- **[Code Review Report](CODE_REVIEW_REPORT.md)**: Detailed code analysis and optimization suggestions

## 🏗️ Architecture

```
brain_unity/
├── unity_integration/          # Core Python modules
│   ├── brain_state_exporter.py    # JSON exporter
│   ├── realtime_server.py          # WebSocket server
│   ├── model_server.py             # Model inference service
│   ├── stimulation_simulator.py    # Stimulation simulator
│   ├── freesurfer_loader.py        # FreeSurfer data loader
│   ├── workflow_manager.py         # Workflow manager
│   └── obj_generator.py            # OBJ model generator
├── unity_examples/             # Unity C# scripts
│   ├── BrainVisualization.cs       # Main visualization script
│   ├── WebSocketClient_Improved.cs # WebSocket client
│   ├── StimulationInput.cs         # Stimulation input UI
│   └── ...
├── unity_startup.py            # Server startup script
├── unity_one_click_install.py  # One-click installation
└── setup_unity_project.py      # Unity project setup
```

## 🎯 Usage Examples

### Python: Start Server

```python
# Start server with specific model
python unity_startup.py --model path/to/model.pt --port 8765

# Start in demo mode
python unity_startup.py --demo
```

### Python: Virtual Stimulation

```python
from unity_integration import ModelServer

# Initialize server
server = ModelServer(model_path="model.pt")

# Simulate stimulation
results = server.simulate_stimulation(
    target_regions=[10, 20, 30],
    amplitude=0.5,
    pattern="sine",
    frequency=10.0,
    duration=50
)
```

### Unity C#: Connect to Server

```csharp
// In Unity, attach WebSocketClient component to GameObject
WebSocketClientImproved wsClient = GetComponent<WebSocketClientImproved>();

// Request brain state
wsClient.GetBrainState((response) => {
    Debug.Log("Received brain state");
});

// Simulate stimulation
int[] regions = {10, 20, 30};
wsClient.SimulateStimulation(regions, 0.5f, "sine", (response) => {
    Debug.Log("Stimulation completed");
});
```

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run tests
pytest tests/

# With coverage report
pytest --cov=unity_integration tests/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FreeSurfer for brain imaging data processing
- Unity Technologies for the 3D engine
- PyTorch team for the deep learning framework

## 📧 Contact

- **Project**: [https://github.com/sheinclotho/brain_unity](https://github.com/sheinclotho/brain_unity)
- **Issues**: [https://github.com/sheinclotho/brain_unity/issues](https://github.com/sheinclotho/brain_unity/issues)

## 🗺️ Roadmap

- [ ] Add unit tests (target: 60% coverage)
- [ ] Implement authentication for WebSocket
- [ ] Add configuration file management
- [ ] Performance optimization
- [ ] English documentation
- [ ] CI/CD pipeline
- [ ] Docker support
- [ ] Cloud deployment guide

## 📊 Status

**Version**: 4.1  
**Status**: Active Development  
**Python**: 3.8+  
**Unity**: 2019.1+

---

**Made with ❤️ by the TwinBrain Team**
