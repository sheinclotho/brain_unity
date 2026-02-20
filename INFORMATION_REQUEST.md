# 信息需求文档 / Information Request Document

**日期 / Date**: 2026-02-19  
**目的 / Purpose**: 深入优化TwinBrain框架 / Further optimize TwinBrain framework  

---

## 概述 / Overview

为了更深入地优化TwinBrain框架，需要以下信息。这些信息将帮助我：
1. 优化数据处理流程
2. 改进模型推理性能
3. 提供更准确的错误处理
4. 创建更好的测试用例

To further optimize the TwinBrain framework, the following information is needed. This will help:
1. Optimize data processing pipeline
2. Improve model inference performance
3. Provide more accurate error handling
4. Create better test cases

---

## 🔴 高优先级 / High Priority

### 1. 缓存文件格式 / Cache File Format

**所需信息 / Required Information**:
```
- [ ] 示例缓存文件（.pt格式）
      Example cache file (.pt format)
      
- [ ] 数据结构说明
      Data structure documentation
      
- [ ] 维度含义
      Dimension meanings
      例如 / Example:
      - shape: (200, 1000, 64) 表示什么?
        What does shape (200, 1000, 64) represent?
      - 200 = 脑区数量? Number of brain regions?
      - 1000 = 时间点? Time points?
      - 64 = 特征数量? Number of features?
      
- [ ] 数值范围
      Value ranges
      - 原始值范围是多少? What is the raw value range?
      - 是否已归一化? Is it normalized?
      - 如果归一化，使用什么方法？
        If normalized, what method was used?
```

**示例文件位置 / Example File Location**:
```
请上传到: Please upload to:
- brain_unity/examples/sample_cache/eeg_data.pt
- brain_unity/examples/sample_cache/hetero_graphs.pt
```

**文档模板 / Documentation Template**:
```python
# 缓存文件结构说明 / Cache File Structure Documentation

"""
EEG数据缓存 / EEG Data Cache: eeg_data.pt
============================================

Format: PyTorch Tensor
Shape: (n_subjects, n_regions, n_timepoints, n_features)
Example: (10, 200, 1000, 64)

Dimensions:
- Axis 0: Number of subjects (训练样本数)
- Axis 1: Number of brain regions (脑区数量, usually 200 for Schaefer200)
- Axis 2: Number of timepoints (时间点数量, e.g., 1000)
- Axis 3: Number of features per timepoint (特征数量, e.g., 64 for EEG bands)

Value Range:
- Raw: [-X, +Y] 原始EEG信号范围
- Normalized: [0, 1] or [-1, 1] 归一化后的范围
- Method: Z-score / Min-Max / Other

Features (for EEG):
- Band powers: Delta, Theta, Alpha, Beta, Gamma
- Connectivity measures
- Other metrics...

Loading:
>>> import torch
>>> data = torch.load('eeg_data.pt')
>>> print(f"Shape: {data.shape}")
>>> print(f"Data type: {data.dtype}")
>>> print(f"Value range: [{data.min():.4f}, {data.max():.4f}]")
"""
```

### 2. 模型格式和使用方法 / Model Format and Usage

**所需信息 / Required Information**:
```
- [ ] 训练好的模型文件示例
      Example trained model file
      位置 / Location: examples/sample_model/hetero_gnn_trained.pt
      
- [ ] 模型架构说明
      Model architecture description
      - 模型类型 / Model type: GNN? Transformer? Other?
      - 层结构 / Layer structure
      - 输入输出规范 / Input/output specification
      
- [ ] 模型checkpoint结构
      Model checkpoint structure
      ```python
      checkpoint = {
          'model_state_dict': {...},  # 模型权重 / Model weights
          'config': {...},             # 配置信息 / Configuration
          'epoch': int,                # 训练轮次 / Training epoch
          'optimizer_state_dict': {...}, # 优化器状态 / Optimizer state
          ... # 其他keys / Other keys
      }
      ```
      
- [ ] 推理代码示例
      Inference code example
      ```python
      # 如何加载模型 / How to load model
      model = load_model('path/to/model.pt')
      
      # 如何准备输入 / How to prepare input
      input_data = prepare_input(...)
      
      # 如何执行推理 / How to perform inference
      with torch.no_grad():
          output = model(input_data)
      
      # 如何解释输出 / How to interpret output
      predictions = post_process(output)
      ```
      
- [ ] 模型输入预处理步骤
      Model input preprocessing steps
      
- [ ] 模型输出后处理步骤
      Model output postprocessing steps
```

---

## 🟡 中优先级 / Medium Priority

### 3. 典型使用场景 / Typical Use Cases

**需要了解 / Need to Know**:
```
- [ ] 主要应用场景
      Main application scenarios
      例如 / Example:
      - 脑疾病研究 / Brain disease research
      - 脑机接口 / Brain-computer interface
      - 神经科学研究 / Neuroscience research
      - 教学演示 / Educational demonstration
      
- [ ] 典型工作流程
      Typical workflow
      Step 1: ...
      Step 2: ...
      Step 3: ...
      
- [ ] 数据规模
      Data scale
      - 平均文件大小 / Average file size
      - 时间点数量范围 / Timepoint count range
      - 同时处理的subjects数量 / Number of concurrent subjects
      
- [ ] 性能要求
      Performance requirements
      - 可接受的延迟 / Acceptable latency
      - 吞吐量要求 / Throughput requirements
      - 内存限制 / Memory constraints
```

### 4. FreeSurfer 数据 / FreeSurfer Data

**需要示例 / Need Examples**:
```
- [ ] 示例FreeSurfer文件
      Sample FreeSurfer files
      - lh.pial (左半球表面 / Left hemisphere surface)
      - rh.pial (右半球表面 / Right hemisphere surface)
      - lh.Schaefer2018_200Parcels_7Networks_order.annot
      - rh.Schaefer2018_200Parcels_7Networks_order.annot
      
- [ ] Atlas信息
      Atlas information
      - 使用哪个图谱? Which atlas is used?
      - Schaefer200? AAL? Destrieux? Other?
      - 脑区标签列表 / Region label list
      - 网络划分 / Network divisions
```

---

## 🟢 低优先级 / Low Priority

### 5. 部署环境 / Deployment Environment

**环境信息 / Environment Info**:
```
- [ ] 目标操作系统
      Target OS: Windows / Linux / macOS
      
- [ ] Python版本
      Python version: 3.8 / 3.9 / 3.10 / 3.11
      
- [ ] Unity版本
      Unity version: 2019.x / 2020.x / 2021.x
      
- [ ] 硬件配置
      Hardware configuration
      - CPU: ...
      - GPU: ... (如果使用 / if used)
      - RAM: ...
      - 存储: ... / Storage: ...
      
- [ ] 网络环境
      Network environment
      - 本地使用 / Local use
      - 局域网 / LAN
      - 远程访问 / Remote access
```

### 6. 已知问题 / Known Issues

**请列出 / Please List**:
```
- [ ] 目前遇到的bug或问题
      Current bugs or issues
      
- [ ] 性能瓶颈
      Performance bottlenecks
      
- [ ] 功能不足
      Missing features
      
- [ ] 用户体验问题
      User experience issues
```

---

## 如何提供信息 / How to Provide Information

### 选项 1: 上传示例文件 / Option 1: Upload Sample Files

创建以下目录结构并上传文件:
Create the following directory structure and upload files:

```
brain_unity/
└── examples/
    ├── sample_cache/
    │   ├── eeg_data.pt              # 示例EEG缓存
    │   ├── hetero_graphs.pt          # 示例图数据
    │   └── README.md                 # 数据说明
    ├── sample_model/
    │   ├── hetero_gnn_trained.pt     # 示例模型
    │   └── README.md                 # 模型说明
    └── sample_freesurfer/
        ├── lh.pial
        ├── rh.pial
        ├── lh.*.annot
        └── rh.*.annot
```

### 选项 2: 创建文档 / Option 2: Create Documentation

创建以下文档文件:
Create the following documentation files:

```
brain_unity/
├── docs/
│   ├── DATA_FORMAT.md          # 数据格式说明
│   ├── MODEL_USAGE.md          # 模型使用指南
│   ├── TYPICAL_WORKFLOW.md     # 典型工作流程
│   └── TROUBLESHOOTING.md      # 故障排查
```

### 选项 3: 添加注释到代码 / Option 3: Add Comments to Code

在相关代码文件中添加详细注释说明数据结构和使用方法。
Add detailed comments in relevant code files explaining data structures and usage.

---

## 预期输出 / Expected Output

提供这些信息后，我将能够:
After providing this information, I will be able to:

1. ✅ **优化数据加载和处理**
   - 更高效的缓存读取
   - 更好的内存管理
   - 批处理优化

2. ✅ **改进模型推理**
   - 正确的输入预处理
   - 准确的输出解释
   - 性能优化建议

3. ✅ **完善错误处理**
   - 针对性的验证
   - 更好的错误消息
   - 自动恢复机制

4. ✅ **创建完整测试**
   - 使用真实数据的集成测试
   - 端到端测试
   - 性能基准测试

5. ✅ **提供具体优化建议**
   - 针对实际使用场景的优化
   - 性能调优参数
   - 最佳实践指南

---

## 联系方式 / Contact

如有任何问题，请通过以下方式联系:
For any questions, please contact via:

- GitHub Issues: https://github.com/sheinclotho/brain_unity/issues
- Pull Request comments
- Repository discussions

---

**感谢您的配合！/ Thank you for your cooperation!**

提供这些信息将大大提高优化的质量和针对性。
Providing this information will greatly improve the quality and relevance of optimizations.
