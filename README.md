# 🏭 CCGL仓储管理系统数据分析工程

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Enabled-orange)](https://modelcontextprotocol.io)

> **企业级仓储管理数据分析平台** - 基于MCP架构的智能数据分析系统

## 🌟 项目特性

- 🤖 **智能数据分析**: 基于机器学习的异常检测、聚类分析和预测模型
- 🔗 **MCP架构设计**: 支持Model Context Protocol的模块化架构
- 🧠 **AI增强分析**: 集成大语言模型，支持自然语言查询
- 📊 **实时仪表板**: 响应式Web界面，实时数据可视化
- 🗄️ **企业级数据库**: 完整的仓储管理数据模型
- 🚀 **容器化部署**: Docker和Kubernetes支持
- 📚 **完整文档**: 详细的API文档和使用指南

## 🚀 快速开始

### 环境要求

- Python 3.8+
- MySQL 8.0+
- Docker (可选)
- Redis (可选，用于缓存)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/fttat/test_zhangxiaolei.git
cd test_zhangxiaolei
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境**
```bash
cp .env.example .env
# 编辑 .env 文件，配置数据库连接等参数
```

4. **初始化数据库**
```bash
python scripts/setup_database.py
```

5. **启动系统**
```bash
# 基础数据分析模式
python main.py

# MCP架构模式
python main_mcp.py

# AI增强模式
python main_llm.py

# 快速演示
python quick_start.py
```

## 📁 项目结构

```
CCGL仓储管理系统/
├── 📄 主程序文件
│   ├── main.py              # 基础分析模式
│   ├── main_mcp.py          # MCP架构模式
│   ├── main_llm.py          # AI增强模式
│   └── quick_start.py       # 快速演示
├── 🧠 核心分析模块
│   └── ccgl_analytics/
│       ├── modules/         # 核心功能模块
│       ├── utils/           # 工具函数
│       └── models/          # 数据模型
├── 🤖 MCP服务器集群
│   └── ccgl_mcp_servers/
│       ├── preprocessing/   # 数据预处理服务器
│       ├── ml/             # 机器学习服务器
│       ├── dashboard/      # 仪表板服务器
│       └── llm/            # 大模型服务器
├── ⚙️ 配置管理
│   └── config/
│       ├── mcp_config.json
│       ├── llm_config.json
│       └── environments/
├── 🗄️ 数据库
│   └── database/
│       ├── schema.sql
│       ├── sample_data.sql
│       └── migrations/
├── 📊 示例和脚本
│   ├── examples/           # 使用示例
│   └── scripts/            # 工具脚本
├── 🧪 测试套件
│   └── tests/
├── 📚 文档系统
│   └── docs/
├── 🚀 部署配置
│   ├── docker/
│   ├── deployments/
│   └── .github/
└── 📋 配置文件
    ├── requirements.txt
    ├── setup.py
    ├── config.yml
    └── .env.example
```

## 🎯 使用模式

### 1. 基础数据分析模式
```bash
python main.py
```
- 标准数据连接和质量评估
- 机器学习分析（聚类、异常检测）
- 生成分析报告

### 2. MCP架构模式
```bash
python main_mcp.py
```
- 启动MCP服务器集群
- 模块化数据处理流程
- 分布式分析架构

### 3. AI增强模式
```bash
python main_llm.py
```
- 大语言模型集成
- 自然语言数据查询
- 智能业务洞察生成

### 4. Web仪表板模式
```bash
python -m ccgl_analytics.modules.web_dashboard
```
- 启动Web服务器
- 访问 http://localhost:8000
- 实时数据可视化

## 🔧 配置说明

### 环境变量配置 (.env)
```env
# 数据库配置
DB_HOST=localhost
DB_PORT=3306
DB_NAME=ccgl_warehouse
DB_USER=your_username
DB_PASSWORD=your_password

# AI模型配置
OPENAI_API_KEY=your_openai_key
CLAUDE_API_KEY=your_claude_key

# MCP配置
MCP_SERVER_PORT=8080
MCP_ENABLE_SSL=false

# Web仪表板
WEB_HOST=0.0.0.0
WEB_PORT=8000
```

## 📊 核心功能

### 数据质量管理
- 自动化数据质量评估
- 缺失值检测和处理
- 数据一致性验证
- 异常值识别

### 机器学习分析
- **聚类分析**: KMeans、DBSCAN、层次聚类
- **异常检测**: Isolation Forest、One-Class SVM
- **降维分析**: PCA、t-SNE、UMAP
- **关联规则**: Apriori算法挖掘

### MCP架构特性
- 模块化服务设计
- 异步数据处理
- 服务编排和监控
- 扩展性架构

### AI增强功能
- 自然语言查询接口
- 智能报告生成
- 对话式数据探索
- 业务洞察自动化

## 🚀 部署方案

### Docker部署
```bash
# 构建镜像
docker build -t ccgl-analytics .

# 运行容器
docker-compose up -d
```

### Kubernetes部署
```bash
# 部署到K8s集群
kubectl apply -f deployments/kubernetes/
```

## 🧪 测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_modules.py

# 覆盖率报告
python -m pytest --cov=ccgl_analytics tests/
```

## 📚 文档

- [安装指南](docs/INSTALL.md)
- [API文档](docs/API.md)
- [MCP使用指南](docs/MCP_GUIDE.md)
- [大模型集成指南](docs/LLM_INTEGRATION.md)
- [部署指南](docs/DEPLOYMENT.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🆘 支持

如有问题，请：
1. 查看 [文档](docs/)
2. 提交 [Issue](https://github.com/fttat/test_zhangxiaolei/issues)
3. 联系维护团队

---

**CCGL仓储管理系统数据分析工程** - 让数据驱动业务决策 🚀