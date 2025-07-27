#!/bin/bash

# CCGL 仓储管理系统快速启动脚本

echo "🚀 启动 CCGL 仓储管理系统..."

# 检查Python版本
python_version=$(python3 --version 2>&1)
echo "Python版本: $python_version"

# 检查是否安装了依赖
echo "📦 检查依赖包..."
if ! python3 -c "import pandas" 2>/dev/null; then
    echo "⚠️ 未检测到pandas，正在安装依赖..."
    pip3 install -r requirements.txt
fi

# 检查配置文件
if [ ! -f ".env" ]; then
    echo "📝 创建环境配置文件..."
    cp .env.example .env
    echo "请编辑 .env 文件配置您的数据库和API密钥"
fi

# 创建必要的目录
echo "📁 创建必要目录..."
mkdir -p logs data models cache

# 运行系统检查
echo "🔍 运行系统检查..."
python3 tests/test_basic_functionality.py

# 提供启动选项
echo ""
echo "🎯 选择启动模式:"
echo "1) 基础分析模式"
echo "2) MCP分布式模式" 
echo "3) AI增强模式"
echo "4) 快速启动向导"
echo ""

read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo "启动基础分析模式..."
        python3 main.py -c config.yml
        ;;
    2)
        echo "启动MCP分布式模式..."
        python3 main_mcp.py --start-mcp-servers
        ;;
    3)
        echo "启动AI增强模式..."
        python3 main_llm.py --interactive
        ;;
    4)
        echo "启动快速启动向导..."
        python3 quick_start.py
        ;;
    *)
        echo "无效选择，启动快速向导..."
        python3 quick_start.py
        ;;
esac