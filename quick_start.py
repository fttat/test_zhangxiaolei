#!/usr/bin/env python3
"""
CCGL 仓储管理系统 - 快速启动脚本

提供快速启动向导和一键部署功能
"""

import argparse
import asyncio
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, Any

# 确保模块可以被导入
sys.path.insert(0, str(Path(__file__).parent))


class QuickStart:
    """快速启动类"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.config_file = self.base_dir / "config.yml"
        
    def check_requirements(self) -> Dict[str, bool]:
        """检查系统要求"""
        checks = {}
        
        # 检查Python版本
        python_version = sys.version_info
        checks['python'] = python_version >= (3, 8)
        
        # 检查必需的包
        required_packages = [
            'numpy', 'pandas', 'scikit-learn', 'fastapi', 'uvicorn'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                checks[f'package_{package}'] = True
            except ImportError:
                checks[f'package_{package}'] = False
        
        # 检查配置文件
        checks['config_file'] = self.config_file.exists()
        
        # 检查环境变量文件
        env_file = self.base_dir / ".env"
        checks['env_file'] = env_file.exists()
        
        return checks
    
    def display_requirements_status(self, checks: Dict[str, bool]):
        """显示系统要求检查结果"""
        print("=== 系统要求检查 ===")
        
        for check, status in checks.items():
            status_str = "✓" if status else "✗"
            print(f"{status_str} {check}: {'通过' if status else '失败'}")
        
        all_passed = all(checks.values())
        if all_passed:
            print("\n✅ 所有检查通过！")
        else:
            print("\n❌ 部分检查失败，请先安装依赖")
        
        return all_passed
    
    def install_dependencies(self):
        """安装依赖"""
        print("正在安装依赖包...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True)
            print("✅ 依赖安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖安装失败: {e}")
            return False
        return True
    
    def setup_environment(self):
        """设置环境"""
        env_example = self.base_dir / ".env.example"
        env_file = self.base_dir / ".env"
        
        if not env_file.exists() and env_example.exists():
            print("创建环境配置文件...")
            import shutil
            shutil.copy(env_example, env_file)
            print("✅ 环境配置文件已创建，请编辑 .env 文件设置您的配置")
            return True
        elif env_file.exists():
            print("✅ 环境配置文件已存在")
            return True
        else:
            print("❌ 找不到环境配置模板")
            return False
    
    def setup_database(self):
        """设置数据库"""
        print("初始化数据库...")
        try:
            script_path = self.base_dir / "scripts" / "setup_database.py"
            if script_path.exists():
                subprocess.run([sys.executable, str(script_path)], check=True)
                print("✅ 数据库初始化成功")
            else:
                print("⚠️ 数据库初始化脚本不存在，跳过数据库设置")
        except subprocess.CalledProcessError as e:
            print(f"❌ 数据库初始化失败: {e}")
            return False
        return True
    
    async def test_basic_functionality(self):
        """测试基础功能"""
        print("测试基础功能...")
        try:
            from ccgl_analytics.modules.data_connection import DataConnection
            from ccgl_analytics.utils.logger import setup_logger
            
            # 测试日志系统
            logger = setup_logger({})
            logger.info("日志系统测试")
            print("✅ 日志系统正常")
            
            # 测试数据连接（但不要求连接成功）
            print("✅ 核心模块加载成功")
            
        except Exception as e:
            print(f"❌ 基础功能测试失败: {e}")
            return False
        return True
    
    def launch_application(self, mode: str = "basic"):
        """启动应用程序"""
        print(f"启动 CCGL 系统 ({mode} 模式)...")
        
        try:
            if mode == "basic":
                subprocess.run([sys.executable, "main.py", "-c", "config.yml"], check=True)
            elif mode == "mcp":
                subprocess.run([sys.executable, "main_mcp.py", "--start-mcp-servers"], check=True)
            elif mode == "llm":
                subprocess.run([sys.executable, "main_llm.py", "--interactive"], check=True)
            elif mode == "web":
                subprocess.run([sys.executable, "main.py", "-w"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 应用启动失败: {e}")
        except KeyboardInterrupt:
            print("\n用户中断应用")
    
    def interactive_setup(self):
        """交互式设置"""
        print("=== CCGL 仓储管理系统快速启动向导 ===")
        print()
        
        # 1. 检查系统要求
        checks = self.check_requirements()
        if not self.display_requirements_status(checks):
            
            # 询问是否安装依赖
            if not checks.get('package_numpy', True):  # 检查是否有包缺失
                install = input("是否安装缺失的依赖包? (y/n): ").lower() == 'y'
                if install:
                    if not self.install_dependencies():
                        print("依赖安装失败，请手动安装")
                        return
                else:
                    print("跳过依赖安装")
                    return
        
        # 2. 设置环境
        if not self.setup_environment():
            print("环境设置失败")
            return
        
        # 3. 设置数据库
        db_setup = input("是否初始化数据库? (y/n): ").lower() == 'y'
        if db_setup:
            self.setup_database()
        
        # 4. 测试功能
        print("\n正在测试基础功能...")
        asyncio.run(self.test_basic_functionality())
        
        # 5. 选择启动模式
        print("\n选择启动模式:")
        print("1. 基础分析模式 (main.py)")
        print("2. MCP分布式模式 (main_mcp.py)")
        print("3. AI增强模式 (main_llm.py)")
        print("4. Web仪表板模式")
        print("5. 跳过启动")
        
        choice = input("请选择 (1-5): ").strip()
        
        if choice == "1":
            self.launch_application("basic")
        elif choice == "2":
            self.launch_application("mcp")
        elif choice == "3":
            self.launch_application("llm")
        elif choice == "4":
            self.launch_application("web")
        else:
            print("设置完成，可以手动启动应用")
    
    def quick_deploy(self):
        """一键部署"""
        print("=== CCGL 一键部署 ===")
        
        # 检查Docker
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            print("✅ Docker 可用")
            
            # 构建Docker镜像
            print("构建Docker镜像...")
            subprocess.run(["docker", "build", "-t", "ccgl-analytics", "."], check=True)
            
            # 启动容器
            print("启动Docker容器...")
            subprocess.run([
                "docker", "run", "-d", 
                "--name", "ccgl-analytics",
                "-p", "8000:8000",
                "ccgl-analytics"
            ], check=True)
            
            print("✅ Docker部署成功")
            print("访问 http://localhost:8000 查看应用")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker不可用，使用本地部署")
            
            # 本地部署
            if self.install_dependencies():
                self.setup_environment()
                self.setup_database()
                print("✅ 本地部署完成")
                
                # 启动Web服务
                self.launch_application("web")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CCGL 仓储管理系统快速启动")
    parser.add_argument('-i', '--interactive', action='store_true', help='交互式设置向导')
    parser.add_argument('-d', '--deploy', action='store_true', help='一键部署')
    parser.add_argument('-c', '--check', action='store_true', help='检查系统要求')
    parser.add_argument('--install-deps', action='store_true', help='安装依赖')
    parser.add_argument('--setup-env', action='store_true', help='设置环境')
    parser.add_argument('--setup-db', action='store_true', help='设置数据库')
    parser.add_argument('--launch', choices=['basic', 'mcp', 'llm', 'web'], help='启动指定模式')
    
    args = parser.parse_args()
    
    quick_start = QuickStart()
    
    try:
        if args.interactive:
            quick_start.interactive_setup()
        elif args.deploy:
            quick_start.quick_deploy()
        elif args.check:
            checks = quick_start.check_requirements()
            quick_start.display_requirements_status(checks)
        elif args.install_deps:
            quick_start.install_dependencies()
        elif args.setup_env:
            quick_start.setup_environment()
        elif args.setup_db:
            quick_start.setup_database()
        elif args.launch:
            quick_start.launch_application(args.launch)
        else:
            # 默认交互式模式
            quick_start.interactive_setup()
            
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"快速启动失败: {e}")


if __name__ == "__main__":
    main()