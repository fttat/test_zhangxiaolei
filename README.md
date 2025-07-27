# 🏭 CCGL Warehouse Management Data Analysis Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/fttat/test_zhangxiaolei/workflows/CI/badge.svg)](https://github.com/fttat/test_zhangxiaolei/actions)

An enterprise-level warehouse management data analysis platform featuring MCP (Model Context Protocol) architecture, AI-powered insights, and comprehensive data processing capabilities.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/fttat/test_zhangxiaolei.git
cd test_zhangxiaolei

# Install dependencies
pip install -r requirements.txt

# Quick setup
python quick_start.py
```

### Basic Usage

```bash
# Traditional analysis
python main.py -c config.yml

# MCP architecture analysis
python main_mcp.py -c config.yml --full-mcp --start-mcp-servers

# AI-enhanced analysis
python main_llm.py -c config.yml --interactive --multilingual

# One-click start
./scripts/start_llm_system.sh -i --multilingual -v
```

## 🏗️ Architecture

### MCP (Model Context Protocol) Design

Our platform uses a modular MCP architecture with dedicated servers for:

- **Data Processing MCP Server**: Data preprocessing and quality management
- **Machine Learning MCP Server**: ML analysis and pattern recognition
- **Dashboard MCP Server**: Web visualization and reporting
- **LLM MCP Server**: AI-powered insights and natural language processing

### Core Features

#### 📊 Data Analysis
- **Database Connectivity**: MySQL integration with MCP Alchemy connector
- **Data Quality Assessment**: Completeness, consistency, and timeliness evaluation
- **Smart Preprocessing**: Missing value handling, outlier detection, normalization

#### 🤖 Machine Learning
- **Clustering Analysis**: KMeans, DBSCAN algorithms
- **Anomaly Detection**: Isolation Forest, statistical methods
- **Dimensionality Reduction**: PCA, t-SNE analysis
- **Association Mining**: Market basket analysis

#### 🧠 AI Integration
- **Multi-LLM Support**: OpenAI GPT-4, Anthropic Claude, Zhipu GLM
- **Natural Language Queries**: Conversational data exploration
- **Intelligent Insights**: AI-driven business recommendations
- **Multilingual Reports**: Automated report generation

#### 🌐 Web Dashboard
- **Responsive Interface**: Modern web-based dashboard
- **Interactive Charts**: QuickChart integration
- **Real-time Updates**: Live data visualization
- **Export Capabilities**: PDF, Excel, and API access

## 📁 Project Structure

```
ccgl-analytics/
├── 🚀 main.py                     # Traditional entry point
├── 🚀 main_mcp.py                 # MCP architecture entry
├── 🚀 main_llm.py                 # AI-enhanced entry
├── 🚀 quick_start.py              # Quick setup script
├── 📁 ccgl_analytics/             # Core modules
├── 📁 ccgl_mcp_servers/           # MCP server collection
├── 📁 config/                     # Configuration files
├── 📁 scripts/                    # Setup and deployment scripts
├── 📁 tests/                      # Test suite
├── 📁 docs/                       # Documentation
├── 📁 database/                   # Database schemas
├── 📁 docker/                     # Container configuration
└── 📁 examples/                   # Usage examples
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.9+
- MySQL 8.0+
- Docker (optional)
- Kubernetes (optional)

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Initialize configuration
python scripts/init_mcp_config.py

# Setup database
python scripts/setup_database.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or use individual containers
docker build -t ccgl-analytics .
docker run -p 8080:8080 ccgl-analytics
```

## 📖 Documentation

- [📖 Installation Guide](docs/INSTALL.md)
- [📖 API Documentation](docs/API.md)
- [📖 MCP Guide](docs/MCP_GUIDE.md)
- [📖 LLM Integration](docs/LLM_INTEGRATION.md)
- [📖 Contributing](docs/CONTRIBUTING.md)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_modules.py
pytest tests/test_mcp_clients.py
pytest tests/test_llm_integration.py

# Run with coverage
pytest --cov=ccgl_analytics
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 Email: support@ccgl.com
- 💬 Issues: [GitHub Issues](https://github.com/fttat/test_zhangxiaolei/issues)
- 📚 Wiki: [Project Wiki](https://github.com/fttat/test_zhangxiaolei/wiki)

## 🎯 Roadmap

- [ ] Enhanced ML algorithms
- [ ] Real-time streaming analytics
- [ ] Advanced visualization features
- [ ] Multi-tenant support
- [ ] Cloud deployment automation