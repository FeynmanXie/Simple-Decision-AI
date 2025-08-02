# Simple Decision AI

一个能够进行简单思考和判断的AI系统，专门用于对各种陈述或问题进行"是"或"否"的二元判断。

## 项目概述

Simple Decision AI 是一个基于深度学习的二元决策系统，能够：

- 接收英文文本输入
- 进行简单的逻辑推理和判断
- 输出"Yes"或"No"的决策结果
- 提供决策的置信度分数
- 生成简单的推理解释

## 项目结构

```
simple-decision-ai/
├── README.md                          # 项目说明文档
├── PRD.md                             # 产品需求文档
├── project_structure.tree            # 项目结构说明
├── config/                            # 配置文件目录
│   ├── model_config.yaml             # 模型配置
│   ├── training_config.yaml          # 训练配置
│   └── app_config.yaml               # 应用配置
├── src/                              # 源代码目录
│   ├── core/                         # 核心模块
│   │   ├── model_manager.py          # 模型管理器
│   │   ├── inference_engine.py       # 推理引擎
│   │   ├── text_processor.py         # 文本处理器
│   │   └── decision_maker.py         # 决策制定器
│   ├── data/                         # 数据处理模块
│   ├── training/                     # 训练模块
│   ├── utils/                        # 工具模块
│   │   ├── config_loader.py          # 配置加载器
│   │   ├── logger.py                 # 日志工具
│   │   ├── validators.py             # 验证器
│   │   └── helpers.py                # 辅助函数
│   └── interfaces/                   # 接口模块
│       ├── cli.py                    # 命令行接口
│       └── api.py                    # API接口
├── data/                             # 数据目录
├── models/                           # 模型目录
├── logs/                             # 日志目录
├── tests/                            # 测试目录
├── scripts/                          # 脚本目录
├── docs/                             # 文档目录
└── examples/                         # 示例目录
    └── basic_usage.py               # 基础使用示例
```

## 核心功能

### 1. 文本理解与处理
- 英文文本预处理和标准化
- 智能分词和标记化
- 输入验证和清理

### 2. 二元决策推理
- 基于BERT等预训练模型的语义理解
- 逻辑推理和常识判断
- 置信度评估

### 3. 结果解释
- 决策推理过程说明
- 置信度分析
- 关键因素识别

## 快速开始

### 1. 项目初始化

```bash
# 克隆项目
git clone <repository-url>
cd simple-decision-ai

# 查看项目结构
cat project_structure.tree
```

### 2. 基础使用示例

```python
# 查看基础使用示例
python examples/basic_usage.py
```

### 3. 命令行界面

```bash
# 单个决策
python -m src.interfaces.cli decide "The sky is blue"

# 批量处理
python -m src.interfaces.cli batch input_texts.json

# 交互模式
python -m src.interfaces.cli interactive

# 查看统计信息
python -m src.interfaces.cli stats
```

## 配置说明

### 模型配置 (config/model_config.yaml)
- 模型名称和路径
- 标记器设置
- 推理参数

### 训练配置 (config/training_config.yaml)
- 训练参数
- 数据配置
- 评估设置

### 应用配置 (config/app_config.yaml)
- API设置
- 日志配置
- 性能参数

## 输入输出格式

### 输入
- 英文文本字符串（最大长度512字符）
- 支持问句和陈述句

### 输出
```json
{
  "decision": "Yes/No",
  "confidence": 0.85,
  "reasoning": "决策推理说明",
  "timestamp": "2024-01-01T12:00:00",
  "input": "原始输入文本"
}
```

## 判断类型

系统能够处理以下类型的判断：

1. **事实性判断**
   - "The sky is blue" → Yes
   - "Water freezes at 0°C" → Yes

2. **逻辑性判断**
   - "If A > B and B > C, then A > C" → Yes
   - "2 + 2 = 5" → No

3. **常识性判断**
   - "People need water to survive" → Yes
   - "Cats can fly naturally" → No

4. **数学性判断**
   - "10 > 5" → Yes
   - "3 × 4 = 11" → No

## 开发计划

### 当前状态
- ✅ 项目架构设计完成
- ✅ 核心模块框架完成
- ✅ 配置系统完成
- ✅ CLI接口完成
- ✅ 基础示例完成

### 下一步计划
- [ ] 数据收集和预处理
- [ ] 模型训练和微调
- [ ] 性能优化和测试
- [ ] Web接口开发
- [ ] 文档完善

## 技术栈

- **Python 3.8+**
- **PyTorch** - 深度学习框架
- **Transformers** - 预训练模型库
- **BERT** - 基础语言理解模型
- **Click** - 命令行界面
- **PyYAML** - 配置管理

## 项目特点

1. **模块化设计** - 清晰的代码结构，易于维护和扩展
2. **配置驱动** - 灵活的配置系统，支持不同场景
3. **多种接口** - CLI、API等多种使用方式
4. **完整日志** - 详细的日志记录和错误处理
5. **可解释性** - 提供决策推理过程说明

## 注意事项

1. 当前版本为框架代码，需要训练模型才能实际使用
2. 仅支持英文输入
3. 专注于二元（是/否）决策
4. 需要GPU支持以获得最佳性能

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请创建Issue或联系开发团队。