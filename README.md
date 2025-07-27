# CCGL 仓储管理系统

[![CI](https://github.com/fttat/test_zhangxiaolei/workflows/CI/badge.svg)](https://github.com/fttat/test_zhangxiaolei/actions)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

🚀 **现代化智能仓储管理与数据分析平台**

## 🌟 核心特性

- **🤖 AI增强分析**: 集成OpenAI、Claude、智谱、通义千问等大模型
- **📊 机器学习**: KMeans聚类、异常检测、关联规则挖掘
- **🔄 MCP架构**: 基于Model Context Protocol的分布式服务器集群
- **📈 实时仪表板**: 交互式Web数据可视化
- **🗄️ 数据管理**: MySQL连接池、数据预处理、质量评估
- **☸️ 云原生**: Docker容器化、Kubernetes部署支持

## 🚀 快速启动

### 基础分析模式
```bash
python main.py -c config.yml
```

### MCP分布式模式
```bash
python main_mcp.py --start-mcp-servers
```

### AI增强交互模式
```bash
python main_llm.py --interactive
```

### 一键快速启动
```bash
python quick_start.py
```

## 📦 安装部署

### 环境要求
- Python 3.8+
- MySQL 8.0+
- Redis 6.0+ (可选)

### 依赖安装
```bash
pip install -r requirements.txt
```

### 数据库初始化
```bash
python scripts/setup_database.py
```

### 配置初始化
```bash
python scripts/init_mcp_config.py
```

## 🏗️ 项目架构

```
ccgl/
├── ccgl_analytics/          # 核心分析模块
│   ├── modules/            # 功能模块
│   └── utils/              # 工具库
├── ccgl_mcp_servers/       # MCP服务器集群
├── config/                 # 配置文件
├── scripts/                # 工具脚本
├── tests/                  # 测试套件
├── docs/                   # 文档系统
├── database/              # 数据库文件
├── docker/                # Docker配置
├── deployments/           # K8s部署配置
├── .github/               # CI/CD工作流
└── examples/              # 示例代码
```

## 🔧 核心模块

### 数据连接管理
- MySQL连接池自动管理
- 连接状态实时监控
- 自动重连与故障恢复

### 数据预处理
- 智能缺失值处理
- 多算法异常检测
- 数据质量评估

### 机器学习分析
- 无监督聚类分析
- 异常行为检测
- 数据降维可视化
- 关联规则挖掘

### MCP服务器集群
- 数据预处理服务器
- 机器学习服务器
- 仪表板服务器
- LLM集成服务器

## 📚 文档导航

- [🚀 安装指南](docs/INSTALL.md)
- [📖 API文档](docs/API.md)
- [🔧 MCP使用指南](docs/MCP_GUIDE.md)
- [🤖 LLM集成指南](docs/LLM_INTEGRATION.md)
- [🎓 教程集合](docs/tutorials/)

## 🤝 贡献指南

我们欢迎所有形式的贡献! 请查看 [CONTRIBUTING.md](docs/CONTRIBUTING.md) 了解详细信息。

## 📄 开源协议

本项目采用 MIT 开源协议 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 技术支持

- 📧 邮箱: support@ccgl.com
- 🐛 问题反馈: [GitHub Issues](https://github.com/fttat/test_zhangxiaolei/issues)
- 💬 讨论交流: [GitHub Discussions](https://github.com/fttat/test_zhangxiaolei/discussions)

---

⭐ **如果这个项目对您有帮助，请给我们一个Star!** ⭐