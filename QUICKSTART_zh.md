# Simple Decision AI - 快速启动指南

欢迎使用 Simple Decision AI！这个指南将帮助您快速了解和使用这个二元决策AI系统。

## 🚀 快速开始

### 1. 查看项目演示

```bash
# 运行演示脚本，了解系统功能
python scripts/demo.py
```

### 2. 查看项目结构

```bash
# 查看完整的项目结构
cat project_structure.tree

# 或者查看简化的目录结构
tree -L 3
```

### 3. 运行基础示例

```bash
# 查看代码使用示例
python examples/basic_usage.py
```

## 📋 系统功能概述

### 核心能力
- ✅ **二元决策**: 对输入文本进行"是/否"判断
- ✅ **置信度评估**: 提供0.0-1.0的置信度分数
- ✅ **推理解释**: 生成决策的推理过程说明
- ✅ **批量处理**: 支持批量文本处理
- ✅ **历史记录**: 保存决策历史和统计信息

### 判断类型
1. **事实性判断** - "The sky is blue" → Yes
2. **逻辑性判断** - "2 + 2 = 4" → Yes  
3. **常识性判断** - "People need water" → Yes
4. **数学性判断** - "10 > 5" → Yes

## 🛠️ 系统架构

```
简单决策AI系统
├── 📝 文本处理器 (TextProcessor)
├── 🧠 模型管理器 (ModelManager)  
├── ⚡ 推理引擎 (InferenceEngine)
└── 🎯 决策制定器 (DecisionMaker)
```

### 核心组件说明

| 组件 | 功能 | 文件位置 |
|------|------|----------|
| TextProcessor | 文本预处理和标记化 | `src/core/text_processor.py` |
| ModelManager | AI模型加载和管理 | `src/core/model_manager.py` |
| InferenceEngine | 模型推理和预测 | `src/core/inference_engine.py` |
| DecisionMaker | 决策制定和结果格式化 | `src/core/decision_maker.py` |

## 📖 使用方法

### 命令行界面 (CLI)

```bash
# 单个决策
python -m src.interfaces.cli decide "The sky is blue"

# 批量处理
python -m src.interfaces.cli batch input_texts.json

# 交互模式
python -m src.interfaces.cli interactive

# 查看统计信息
python -m src.interfaces.cli stats

# 查看决策历史
python -m src.interfaces.cli history --limit 10

# 设置置信度阈值
python -m src.interfaces.cli set-threshold --threshold 0.8
```

### 代码调用示例

```python
from src.core.decision_maker import DecisionMaker

# 初始化决策器
decision_maker = DecisionMaker()
decision_maker.initialize()

# 单个决策
result = decision_maker.decide("The sky is blue")
print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.2%}")

# 批量决策
texts = ["2 + 2 = 4", "Cats can fly", "Water is wet"]
results = decision_maker.decide_batch(texts)
for result in results:
    print(f"'{result['input']}' → {result['decision']}")
```

## 📊 输出格式

### JSON格式
```json
{
  "decision": "Yes",
  "confidence": 0.95,
  "reasoning": "Decision based on high confidence and clear facts",
  "timestamp": "2024-01-01T12:00:00.000000",
  "input": "The sky is blue."
}
```

### 文本格式
```
Decision: Yes
Confidence: 95.0%
Reasoning: Decision based on high confidence and clear facts
```

## ⚙️ 配置说明

### 模型配置 (`config/model_config.yaml`)
```yaml
model:
  name: "bert-base-uncased"
  cache_dir: "./models/pretrained"
  max_sequence_length: 512
  num_labels: 2

inference:
  confidence_threshold: 0.5
  batch_size: 1
  device: "auto"
```

### 应用配置 (`config/app_config.yaml`)
```yaml
app:
  name: "Simple Decision AI"
  debug: false

logging:
  level: "INFO"
  file_handler:
    enabled: true
    filename: "./logs/app.log"
```

## 🔧 当前状态

### ✅ 已完成
- 项目架构和框架代码
- 核心模块实现
- CLI命令行接口
- 配置系统
- 日志系统
- 基础文档

### 🚧 待完成（需要进一步开发）
- 实际的AI模型训练/加载
- 训练数据准备
- 模型微调
- 性能优化
- Web API接口
- 单元测试

## 📝 重要说明

> ⚠️ **注意**: 当前版本为系统框架，包含完整的代码结构和接口，但需要训练好的模型才能进行实际的决策判断。

### 下一步开发建议

1. **准备环境**
   ```bash
   pip install torch transformers numpy pandas PyYAML click
   ```

2. **准备训练数据**
   - 收集是/否判断的数据集
   - 格式化为训练所需的JSON格式

3. **训练/加载模型**
   - 微调BERT模型或加载预训练的分类模型
   - 配置模型路径

4. **测试系统**
   - 运行实际的决策任务
   - 验证准确率和性能

## 🆘 获取帮助

- 📖 查看 `README.md` 了解完整功能
- 📋 查看 `PRD.md` 了解产品需求
- 🌳 查看 `project_structure.tree` 了解代码结构
- 💻 运行 `python scripts/demo.py` 查看演示
- 🔍 查看 `examples/` 目录中的代码示例

## 🎯 项目目标

创建一个能够进行简单思考和判断的AI系统，具备：
- 🤖 **智能判断**: 不仅仅是关键词匹配，而是真正的语义理解
- 🎯 **专注性**: 专门针对二元决策优化
- 🔧 **可扩展**: 模块化设计，易于扩展功能
- 📈 **可解释**: 提供决策推理过程

---

**开始您的AI决策之旅吧！** 🚀