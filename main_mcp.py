#!/usr/bin/env python3
"""
CCGL 仓储管理系统 - MCP架构主程序

基于Model Context Protocol的分布式服务器集群架构
"""

import argparse
import asyncio
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List

# 确保模块可以被导入
sys.path.insert(0, str(Path(__file__).parent))

from ccgl_mcp_servers.mcp_orchestrator import MCPOrchestrator
from ccgl_mcp_servers.ccgl_preprocessing_mcp_server import PreprocessingMCPServer
from ccgl_mcp_servers.ccgl_ml_mcp_server import MachineLearningMCPServer
from ccgl_mcp_servers.ccgl_dashboard_mcp_server import DashboardMCPServer
from ccgl_mcp_servers.ccgl_llm_mcp_server import LLMMCPServer
from ccgl_analytics.modules.mcp_client_orchestrator import MCPClientOrchestrator
from ccgl_analytics.utils.logger import setup_logger


class CCGLMCPSystem:
    """CCGL MCP分布式系统主类"""
    
    def __init__(self, config_path: str):
        """初始化MCP系统"""
        self.config = self._load_config(config_path)
        self.logger = setup_logger(self.config.get('logging', {}))
        
        # 初始化MCP编排器
        self.orchestrator = MCPOrchestrator(self.config)
        self.client_orchestrator = MCPClientOrchestrator(self.config)
        
        # MCP服务器实例
        self.servers = {}
        
        self.logger.info("CCGL MCP 系统初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            sys.exit(1)
    
    async def start_mcp_servers(self):
        """启动所有MCP服务器"""
        try:
            mcp_config = self.config.get('mcp_servers', {})
            
            # 启动数据预处理服务器
            if mcp_config.get('preprocessing', {}).get('enabled', True):
                preprocessing_server = PreprocessingMCPServer(
                    port=mcp_config['preprocessing']['port'],
                    config=self.config
                )
                self.servers['preprocessing'] = preprocessing_server
                await preprocessing_server.start()
                self.logger.info(f"数据预处理MCP服务器启动在端口 {mcp_config['preprocessing']['port']}")
            
            # 启动机器学习服务器
            if mcp_config.get('ml_analysis', {}).get('enabled', True):
                ml_server = MachineLearningMCPServer(
                    port=mcp_config['ml_analysis']['port'],
                    config=self.config
                )
                self.servers['ml_analysis'] = ml_server
                await ml_server.start()
                self.logger.info(f"机器学习MCP服务器启动在端口 {mcp_config['ml_analysis']['port']}")
            
            # 启动仪表板服务器
            if mcp_config.get('dashboard', {}).get('enabled', True):
                dashboard_server = DashboardMCPServer(
                    port=mcp_config['dashboard']['port'],
                    config=self.config
                )
                self.servers['dashboard'] = dashboard_server
                await dashboard_server.start()
                self.logger.info(f"仪表板MCP服务器启动在端口 {mcp_config['dashboard']['port']}")
            
            # 启动LLM集成服务器
            if mcp_config.get('llm_integration', {}).get('enabled', True):
                llm_server = LLMMCPServer(
                    port=mcp_config['llm_integration']['port'],
                    config=self.config
                )
                self.servers['llm_integration'] = llm_server
                await llm_server.start()
                self.logger.info(f"LLM集成MCP服务器启动在端口 {mcp_config['llm_integration']['port']}")
            
            # 启动编排器
            await self.orchestrator.start()
            self.logger.info("MCP编排器启动完成")
            
        except Exception as e:
            self.logger.error(f"MCP服务器启动失败: {e}")
            raise
    
    async def stop_mcp_servers(self):
        """停止所有MCP服务器"""
        try:
            for name, server in self.servers.items():
                await server.stop()
                self.logger.info(f"{name} MCP服务器已停止")
            
            await self.orchestrator.stop()
            self.logger.info("MCP编排器已停止")
            
        except Exception as e:
            self.logger.error(f"MCP服务器停止失败: {e}")
    
    async def run_distributed_analysis(self, analysis_type: str = "full"):
        """运行分布式分析"""
        try:
            self.logger.info(f"开始分布式 {analysis_type} 分析")
            
            # 使用客户端编排器协调分析任务
            results = await self.client_orchestrator.coordinate_analysis(analysis_type)
            
            self.logger.info("分布式分析完成")
            return results
            
        except Exception as e:
            self.logger.error(f"分布式分析失败: {e}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            status = {
                'servers': {},
                'orchestrator': await self.orchestrator.get_status(),
                'timestamp': asyncio.get_event_loop().time()
            }
            
            for name, server in self.servers.items():
                status['servers'][name] = await server.get_status()
            
            return status
            
        except Exception as e:
            self.logger.error(f"获取系统状态失败: {e}")
            return {'error': str(e)}
    
    async def interactive_mode(self):
        """交互式模式"""
        print("=== CCGL MCP 交互式模式 ===")
        print("可用命令:")
        print("  status - 查看系统状态")
        print("  analyze [type] - 运行分析 (type: full, clustering, anomaly, association)")
        print("  servers - 查看服务器列表")
        print("  help - 显示帮助")
        print("  quit - 退出")
        print()
        
        while True:
            try:
                command = input("CCGL-MCP> ").strip().lower()
                
                if command == "quit":
                    break
                elif command == "status":
                    status = await self.get_system_status()
                    print(f"系统状态: {status}")
                elif command.startswith("analyze"):
                    parts = command.split()
                    analysis_type = parts[1] if len(parts) > 1 else "full"
                    print(f"执行 {analysis_type} 分析...")
                    results = await self.run_distributed_analysis(analysis_type)
                    print(f"分析完成: {results}")
                elif command == "servers":
                    print(f"运行中的服务器: {list(self.servers.keys())}")
                elif command == "help":
                    print("可用命令: status, analyze [type], servers, help, quit")
                else:
                    print("未知命令，输入 'help' 查看帮助")
                    
            except KeyboardInterrupt:
                print("\n退出交互模式")
                break
            except Exception as e:
                print(f"命令执行错误: {e}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CCGL 仓储管理系统 - MCP架构")
    parser.add_argument('-c', '--config', default='config.yml', help='配置文件路径')
    parser.add_argument('--start-mcp-servers', action='store_true', help='启动MCP服务器集群')
    parser.add_argument('--stop-mcp-servers', action='store_true', help='停止MCP服务器集群')
    parser.add_argument('-a', '--analysis', help='运行分布式分析')
    parser.add_argument('-i', '--interactive', action='store_true', help='交互式模式')
    parser.add_argument('--status', action='store_true', help='查看系统状态')
    
    args = parser.parse_args()
    
    try:
        # 初始化MCP系统
        mcp_system = CCGLMCPSystem(args.config)
        
        if args.start_mcp_servers:
            print("启动MCP服务器集群...")
            await mcp_system.start_mcp_servers()
            print("MCP服务器集群启动完成")
            
            # 保持运行
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n正在停止MCP服务器...")
                await mcp_system.stop_mcp_servers()
                
        elif args.stop_mcp_servers:
            print("停止MCP服务器集群...")
            await mcp_system.stop_mcp_servers()
            
        elif args.analysis:
            print(f"运行分布式 {args.analysis} 分析...")
            await mcp_system.start_mcp_servers()
            results = await mcp_system.run_distributed_analysis(args.analysis)
            print(f"分析结果: {results}")
            await mcp_system.stop_mcp_servers()
            
        elif args.status:
            await mcp_system.start_mcp_servers()
            status = await mcp_system.get_system_status()
            print(f"系统状态: {status}")
            await mcp_system.stop_mcp_servers()
            
        elif args.interactive:
            await mcp_system.start_mcp_servers()
            await mcp_system.interactive_mode()
            await mcp_system.stop_mcp_servers()
            
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"系统错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())