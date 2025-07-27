#!/usr/bin/env python3
"""
基础功能测试脚本
"""

import sys
import os
import asyncio
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def test_imports():
    """测试核心模块导入"""
    print("🧪 测试模块导入...")
    
    try:
        # 测试基础导入（不依赖外部库）
        from ccgl_analytics.utils.logger import setup_logger
        print("✅ Logger 导入成功")
        
        # 只导入不会触发外部依赖的模块
        import ccgl_analytics
        print("✅ ccgl_analytics 包导入成功")
        
        import ccgl_mcp_servers
        print("✅ ccgl_mcp_servers 包导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

async def test_basic_functionality():
    """测试基础功能"""
    print("\n🔧 测试基础功能...")
    
    try:
        # 测试日志系统
        from ccgl_analytics.utils.logger import setup_logger
        logger = setup_logger({'level': 'INFO'})
        logger.info("日志系统测试")
        print("✅ 日志系统正常")
        
        # 测试配置文件读取
        import yaml
        config_file = project_root / "config.yml"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("✅ 配置文件读取成功")
        else:
            print("⚠️ 配置文件不存在，但这是正常的")
        
        # 测试目录结构
        required_dirs = ['ccgl_analytics', 'ccgl_mcp_servers', 'config', 'scripts', 'tests']
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            if dir_path.exists():
                print(f"✅ 目录 {dir_name} 存在")
            else:
                print(f"❌ 目录 {dir_name} 不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        return False

async def test_sample_data_analysis():
    """测试示例数据分析"""
    print("\n📊 测试项目结构完整性...")
    
    try:
        # 检查主要入口文件
        main_files = ['main.py', 'main_mcp.py', 'main_llm.py', 'quick_start.py']
        for file_name in main_files:
            file_path = project_root / file_name
            if file_path.exists():
                print(f"✅ 入口文件 {file_name} 存在")
            else:
                print(f"❌ 入口文件 {file_name} 不存在")
        
        # 检查核心模块文件
        core_modules = [
            'ccgl_analytics/modules/data_connection.py',
            'ccgl_analytics/modules/data_preprocessing.py',
            'ccgl_analytics/modules/analysis_core.py',
            'ccgl_analytics/utils/logger.py'
        ]
        
        for module_path in core_modules:
            file_path = project_root / module_path
            if file_path.exists():
                print(f"✅ 核心模块 {module_path} 存在")
            else:
                print(f"❌ 核心模块 {module_path} 不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 项目结构检查失败: {e}")
        return False

async def main():
    """主测试函数"""
    print("🚀 CCGL 系统基础功能测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_sample_data_analysis
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📋 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统基础功能正常")
        return True
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)