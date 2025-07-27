# CCGL 仓储管理系统 - 安装指南

## 系统要求

### 硬件要求
- CPU: 2核心以上
- 内存: 4GB RAM最小，8GB推荐
- 存储: 10GB可用空间
- 网络: 稳定的互联网连接

### 软件要求
- Python 3.8 或更高版本
- MySQL 8.0 或更高版本
- Redis 6.0+ (可选，用于缓存)
- Node.js 16+ (用于前端组件)

## 安装步骤

### 1. 克隆项目
```bash
git clone https://github.com/fttat/test_zhangxiaolei.git
cd test_zhangxiaolei
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置数据库
```bash
# 创建数据库
mysql -u root -p < database/schema.sql

# 加载示例数据
mysql -u root -p ccgl_warehouse < database/sample_data.sql
```

### 5. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，设置数据库连接和API密钥
```

### 6. 初始化系统
```bash
python scripts/init_mcp_config.py
python scripts/setup_database.py
```

### 7. 运行系统
```bash
# 基础模式
python main.py -c config.yml

# MCP分布式模式
python main_mcp.py --start-mcp-servers

# AI增强模式
python main_llm.py --interactive

# 快速启动向导
python quick_start.py
```

## Docker 部署

### 使用 Docker Compose
```bash
docker-compose up -d
```

### 单独构建镜像
```bash
docker build -t ccgl-analytics .
docker run -p 8000:8000 ccgl-analytics
```

## Kubernetes 部署

### 使用 Helm
```bash
helm install ccgl-analytics deployments/helm/ccgl-analytics/
```

### 直接部署
```bash
kubectl apply -f deployments/kubernetes/
```

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查MySQL服务是否运行
   - 验证数据库凭据
   - 确认网络连接

2. **依赖安装失败**
   - 更新pip: `pip install --upgrade pip`
   - 使用国内镜像: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/`

3. **权限错误**
   - 确保用户有足够权限
   - 检查文件所有者和权限

### 日志查看
```bash
tail -f logs/ccgl.log
```

### 性能监控
访问 http://localhost:9090 查看Prometheus指标

## 下一步

安装完成后，请参考：
- [用户指南](README.md)
- [API文档](API.md)
- [教程](tutorials/getting_started.md)