# CCGL仓储管理系统安装指南

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+
- **数据库**: MySQL 8.0+ (可选)
- **容器**: Docker 20.10+ (可选)
- **操作系统**: Linux, macOS, Windows

### 方式一：Python直接安装

#### 1. 克隆项目
```bash
git clone https://github.com/fttat/test_zhangxiaolei.git
cd test_zhangxiaolei
```

#### 2. 创建虚拟环境
```bash
# 使用venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate     # Windows

# 或使用conda
conda create -n ccgl python=3.11
conda activate ccgl
```

#### 3. 安装依赖
```bash
pip install -r requirements.txt
```

#### 4. 配置环境
```bash
cp .env.example .env
# 编辑 .env 文件，配置必要的参数
```

#### 5. 快速体验
```bash
# 快速演示（无需数据库）
python quick_start.py

# 基础分析模式（使用模拟数据）
python main.py

# AI增强模式
python main_llm.py
```

### 方式二：Docker容器部署

#### 1. 构建并启动服务
```bash
cd docker
docker-compose up -d
```

#### 2. 查看服务状态
```bash
docker-compose ps
```

#### 3. 访问服务
- 主应用: http://localhost:8000
- 数据库: localhost:3306
- 监控面板: http://localhost:3000 (Grafana)
- 指标收集: http://localhost:9090 (Prometheus)

### 方式三：生产环境部署

#### 1. 数据库准备
```bash
# 连接MySQL
mysql -u root -p

# 创建数据库和用户
source database/schema.sql
source database/sample_data.sql
```

#### 2. 系统服务配置
```bash
# 复制systemd服务文件
sudo cp deployments/systemd/ccgl-analytics.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ccgl-analytics
sudo systemctl start ccgl-analytics
```

## 🔧 配置说明

### 环境变量配置

创建 `.env` 文件：

```env
# 数据库配置
DB_HOST=localhost
DB_PORT=3306
DB_NAME=ccgl_warehouse
DB_USER=ccgl_user
DB_PASSWORD=your_secure_password

# AI模型配置（可选）
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Web服务配置
WEB_HOST=0.0.0.0
WEB_PORT=8000

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/ccgl_analytics.log
```

### 主配置文件 (config.yml)

```yaml
database:
  host: localhost
  port: 3306
  database: ccgl_warehouse
  user: ccgl_user
  password: your_password

analysis:
  clustering_methods: [kmeans, dbscan, hierarchical]
  anomaly_methods: [isolation_forest, one_class_svm]

web:
  host: 0.0.0.0
  port: 8000
  debug: false
```

## 🗄️ 数据库设置

### MySQL数据库初始化

```bash
# 1. 创建数据库
mysql -u root -p -e "CREATE DATABASE ccgl_warehouse;"

# 2. 创建用户
mysql -u root -p -e "
CREATE USER 'ccgl_user'@'%' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON ccgl_warehouse.* TO 'ccgl_user'@'%';
FLUSH PRIVILEGES;"

# 3. 导入架构
mysql -u ccgl_user -p ccgl_warehouse < database/schema.sql

# 4. 导入示例数据
mysql -u ccgl_user -p ccgl_warehouse < database/sample_data.sql
```

### 使用Docker MySQL

```bash
# 启动MySQL容器
docker run -d \
  --name ccgl-mysql \
  -e MYSQL_ROOT_PASSWORD=rootpass \
  -e MYSQL_DATABASE=ccgl_warehouse \
  -e MYSQL_USER=ccgl_user \
  -e MYSQL_PASSWORD=userpass \
  -p 3306:3306 \
  -v $(pwd)/database:/docker-entrypoint-initdb.d \
  mysql:8.0
```

## 🚀 启动服务

### 开发模式

```bash
# 基础分析
python main.py

# AI增强分析
python main_llm.py

# 交互式Web界面
python -m ccgl_analytics.modules.web_dashboard
```

### 生产模式

```bash
# 使用Gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 "ccgl_analytics.modules.web_dashboard:create_app()"

# 使用uWSGI
uwsgi --http :8000 --module ccgl_analytics.modules.web_dashboard:app --workers 4
```

## 🧪 功能验证

### 基础功能测试

```bash
# 运行快速演示
python quick_start.py

# 运行单元测试
python -m pytest tests/

# 代码覆盖率测试
python -m pytest --cov=ccgl_analytics tests/
```

### Web界面测试

1. 启动Web服务: `python main_llm.py`
2. 在浏览器中访问生成的仪表板文件
3. 检查各个图表和数据展示

### AI功能测试

```bash
# 设置AI API密钥
export OPENAI_API_KEY="your_key"

# 运行AI增强分析
python main_llm.py
```

## 🔍 故障排除

### 常见问题

#### 1. 模块导入错误
```bash
# 解决方案：设置Python路径
export PYTHONPATH=$PWD:$PYTHONPATH
```

#### 2. 数据库连接失败
```bash
# 检查MySQL服务状态
sudo systemctl status mysql

# 验证连接参数
mysql -h localhost -u ccgl_user -p ccgl_warehouse
```

#### 3. 依赖包安装失败
```bash
# 升级pip
pip install --upgrade pip

# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 4. 权限问题
```bash
# 确保目录权限正确
chmod -R 755 /path/to/ccgl
chown -R ccgl:ccgl /path/to/ccgl
```

### 日志查看

```bash
# 查看应用日志
tail -f logs/ccgl_analytics.log

# 查看Docker容器日志
docker-compose logs -f ccgl-analytics

# 查看系统服务日志
sudo journalctl -u ccgl-analytics -f
```

## 📊 性能调优

### 数据库优化

```sql
-- 创建索引
CREATE INDEX idx_inventory_warehouse_product ON inventory(warehouse_id, product_id);
CREATE INDEX idx_movements_date ON inventory_movements(movement_date);

-- 分析表统计信息
ANALYZE TABLE inventory, inventory_movements, products;
```

### 应用优化

```python
# config.yml中的优化配置
data_processing:
  cache_timeout: 3600
  max_workers: 4
  batch_size: 1000
  memory_limit_gb: 4
```

## 🔐 安全配置

### 生产环境安全

1. **更改默认密码**
```bash
# 生成强密码
openssl rand -base64 32
```

2. **配置防火墙**
```bash
# UFW配置
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

3. **SSL证书配置**
```bash
# 使用Let's Encrypt
sudo certbot --nginx -d your-domain.com
```

4. **数据库安全**
```sql
-- 移除测试用户
DROP USER IF EXISTS 'test'@'localhost';

-- 限制root访问
UPDATE mysql.user SET host='localhost' WHERE user='root';
FLUSH PRIVILEGES;
```

## 🔄 更新和维护

### 系统更新

```bash
# 1. 备份数据
mysqldump -u ccgl_user -p ccgl_warehouse > backup_$(date +%Y%m%d).sql

# 2. 拉取最新代码
git pull origin main

# 3. 更新依赖
pip install -r requirements.txt --upgrade

# 4. 重启服务
sudo systemctl restart ccgl-analytics
```

### 定期维护

```bash
# 清理日志文件
find logs/ -name "*.log" -mtime +30 -delete

# 数据库优化
mysql -u ccgl_user -p -e "OPTIMIZE TABLE ccgl_warehouse.inventory;"

# 磁盘空间检查
df -h
```

## 📱 监控告警

### Prometheus + Grafana

1. 访问Grafana: http://localhost:3000
2. 默认账号: admin/admin123
3. 导入预置仪表板配置

### 系统监控

```bash
# 系统资源监控
htop
iotop
nethogs

# 应用监控
ps aux | grep python
netstat -tlnp | grep :8000
```

---

**技术支持**: 如需帮助，请查看[项目文档](../docs/)或提交[Issue](https://github.com/fttat/test_zhangxiaolei/issues)