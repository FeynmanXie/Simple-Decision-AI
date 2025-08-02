"""
Demo script for Simple Decision AI.

This script demonstrates the AI system's capabilities with example texts.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path for absolute imports from 'src'
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger


def print_banner():
    """Print a welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    Simple Decision AI                         ║
║                  二元决策AI系统演示                            ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def demo_text_examples():
    """Show example texts that the AI can process."""
    print("\n🤖 AI可以处理的判断类型示例：\n")
    
    examples = [
        ("事实性判断", [
            "The sky is blue.",
            "Water boils at 100 degrees Celsius.",
            "Paris is the capital of France.",
            "The Earth is flat."
        ]),
        ("逻辑性判断", [
            "If A > B and B > C, then A > C.",
            "2 + 2 = 4",
            "All cats are animals, Fluffy is a cat, so Fluffy is an animal.",
            "If it's raining, then the ground is wet."
        ]),
        ("常识性判断", [
            "People need water to survive.",
            "Cats can fly naturally.",
            "Fire is hot.",
            "Ice is warmer than boiling water."
        ]),
        ("数学性判断", [
            "10 > 5",
            "3 × 4 = 12",
            "The square root of 16 is 4.",
            "2 + 2 = 5"
        ])
    ]
    
    for category, texts in examples:
        print(f"📝 {category}:")
        for text in texts:
            print(f"   • {text}")
        print()


def demo_expected_outputs():
    """Show expected output format."""
    print("📊 预期输出格式：\n")
    
    example_output = {
        "decision": "Yes",
        "confidence": 0.95,
        "reasoning": "Decision 'Yes' based on: Very high confidence in the decision, clear distinction between options, positive language indicators present, input is a statement",
        "timestamp": "2024-01-01T12:00:00.000000",
        "input": "The sky is blue."
    }
    
    print("JSON格式输出：")
    import json
    print(json.dumps(example_output, indent=2, ensure_ascii=False))
    
    print("\n文本格式输出：")
    print(f"Decision: {example_output['decision']}")
    print(f"Confidence: {example_output['confidence']:.2%}")
    print(f"Reasoning: {example_output['reasoning']}")


def demo_cli_commands():
    """Show CLI command examples."""
    print("\n💻 命令行使用示例：\n")
    
    commands = [
        ("单个决策", 'python -m src.interfaces.cli decide "The sky is blue"'),
        ("批量处理", 'python -m src.interfaces.cli batch input_texts.json'),
        ("交互模式", 'python -m src.interfaces.cli interactive'),
        ("查看统计", 'python -m src.interfaces.cli stats'),
        ("查看历史", 'python -m src.interfaces.cli history --limit 5'),
        ("设置阈值", 'python -m src.interfaces.cli set-threshold --threshold 0.8')
    ]
    
    for description, command in commands:
        print(f"🔹 {description}:")
        print(f"   {command}")
        print()


def demo_project_structure():
    """Show project structure highlights."""
    print("📁 项目结构亮点：\n")
    
    structure = [
        ("核心模块", "src/core/ - 包含决策制定器、推理引擎、模型管理器"),
        ("配置系统", "config/ - YAML配置文件，支持灵活配置"),
        ("接口层", "src/interfaces/ - CLI命令行接口"),
        ("工具模块", "src/utils/ - 日志、验证、配置加载等工具"),
        ("示例代码", "examples/ - 基础使用示例"),
        ("文档", "README.md, PRD.md - 完整的项目文档")
    ]
    
    for component, description in structure:
        print(f"🔸 {component}: {description}")


def demo_next_steps():
    """Show next development steps."""
    print("\n🚀 下一步开发计划：\n")
    
    steps = [
        "1️⃣ 安装依赖包 (PyTorch, Transformers, 等)",
        "2️⃣ 准备训练数据集",
        "3️⃣ 训练或微调BERT模型",
        "4️⃣ 配置模型路径和参数",
        "5️⃣ 运行实际的决策任务",
        "6️⃣ 性能优化和测试",
        "7️⃣ 部署和生产环境配置"
    ]
    
    for step in steps:
        print(f"   {step}")


def main():
    """Main demo function."""
    # Set up logging
    logger = setup_logger("Demo", log_level="INFO")
    
    try:
        print_banner()
        
        print("欢迎使用 Simple Decision AI 演示！")
        print("这是一个能够进行二元判断的AI系统框架。")
        
        demo_text_examples()
        demo_expected_outputs()
        demo_cli_commands()
        demo_project_structure()
        demo_next_steps()
        
        print("\n" + "="*60)
        print("📖 更多信息请查看：")
        print("   • README.md - 项目概述和使用指南")
        print("   • PRD.md - 详细的产品需求文档")
        print("   • project_structure.tree - 完整的项目结构")
        print("   • examples/basic_usage.py - 代码使用示例")
        
        print("\n✨ 感谢使用 Simple Decision AI！")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)