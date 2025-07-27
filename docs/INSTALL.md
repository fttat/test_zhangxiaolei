# CCGL Analytics - Installation & Deployment Guide

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows

### Optional Dependencies
- **MySQL**: 8.0+ (for database storage)
- **Docker**: 20.10+ (for containerized deployment)
- **Kubernetes**: 1.20+ (for scalable deployment)

## 🚀 Quick Installation

### 1. Clone Repository
```bash
git clone https://github.com/fttat/test_zhangxiaolei.git
cd test_zhangxiaolei
```

### 2. Quick Setup
```bash
# Use the quick start script for automated setup
python quick_start.py

# Or manual setup:
pip install -r requirements.txt
python main.py --create-sample
python main.py
```

### 3. MCP Architecture Setup
```bash
# Start MCP servers and run demo
python main_mcp.py --start-mcp-servers --demo-workflow --list-tools

# Interactive MCP mode
python main_mcp.py --full-mcp --start-mcp-servers
```

## 🗄️ Database Setup

### MySQL Configuration

1. **Create Database**
```sql
CREATE DATABASE ccgl_warehouse;
USE ccgl_warehouse;
```

2. **Create Tables**
```bash
mysql -u root -p ccgl_warehouse < database/schema.sql
```

3. **Load Sample Data**
```bash
mysql -u root -p ccgl_warehouse < database/sample_data.sql
```

4. **Update Configuration**
```yaml
# config.yml
data_source:
  type: mysql
  database:
    host: localhost
    port: 3306
    user: root
    password: your_password
    database: ccgl_warehouse
```

## 🤖 LLM Integration Setup

### 1. API Keys Configuration

Create `.env` file from `.env.example`:
```bash
cp .env.example .env
```

Update with your API keys:
```bash
# .env
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
ZHIPU_API_KEY=your-zhipu-key
```

### 2. Test LLM Integration
```bash
python main_llm.py --interactive
```

## 🐳 Docker Deployment

### 1. Build Container
```bash
docker build -t ccgl-analytics .
```

### 2. Run Container
```bash
docker run -p 8080:8080 -v $(pwd)/data:/app/data ccgl-analytics
```

### 3. Docker Compose (Recommended)
```bash
docker-compose up -d
```

## ☸️ Kubernetes Deployment

### 1. Create Namespace
```bash
kubectl create namespace ccgl-analytics
```

### 2. Deploy Application
```bash
kubectl apply -f deployments/kubernetes/
```

### 3. Access Application
```bash
kubectl port-forward service/ccgl-analytics 8080:8080 -n ccgl-analytics
```

## 🔧 Configuration

### Main Configuration (config.yml)

```yaml
data_source:
  type: file  # or 'mysql'
  file_path: sample_data.csv
  database:
    host: localhost
    port: 3306
    user: root
    password: ""
    database: ccgl_warehouse

preprocessing:
  steps:
    - quality_check
    - handle_missing
    - detect_outliers
    - normalize_data
  missing_strategy: auto
  outlier_method: iqr
  normalization_method: standard

analysis:
  include_clustering: true
  include_anomaly_detection: true
  include_dimensionality_reduction: true

output:
  save_results: true
  output_dir: results
  formats: [csv, json]

logging:
  level: INFO
  log_file: ccgl_analytics.log
```

### MCP Configuration

MCP servers are configured in `config/mcp_config.json`:

```json
{
  "name": "ccgl-mcp-system",
  "servers": {
    "preprocessing": {
      "name": "ccgl-preprocessing-server",
      "command": "python",
      "args": ["ccgl_mcp_servers/ccgl_preprocessing_mcp_server.py"]
    }
  },
  "settings": {
    "log_level": "INFO",
    "max_connections": 10,
    "timeout": 30
  }
}
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure Python path is correct
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Permission Errors
```bash
# Fix file permissions
chmod +x scripts/*.sh
```

#### 3. MySQL Connection Issues
```bash
# Check MySQL service
sudo systemctl status mysql

# Test connection
mysql -u root -p -e "SELECT 1;"
```

#### 4. Memory Issues
```bash
# Monitor memory usage
top -p $(pgrep -f "python.*ccgl")

# Reduce dataset size for testing
python main.py --create-sample --size 100
```

### Debug Mode

Enable verbose logging:
```bash
python main.py -v
python main_mcp.py -v --start-mcp-servers
```

Check log files:
```bash
tail -f logs/ccgl_analytics_*.log
```

## 📊 Performance Tuning

### Database Optimization

1. **Indexes**
```sql
CREATE INDEX idx_product_category ON warehouse_inventory(category);
CREATE INDEX idx_last_updated ON warehouse_inventory(last_updated);
```

2. **Query Optimization**
```sql
EXPLAIN SELECT * FROM warehouse_inventory WHERE category = 'Electronics';
```

### Memory Optimization

1. **Reduce Memory Usage**
```python
# Use chunked processing for large datasets
chunk_size = 1000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

2. **Configure Memory Limits**
```yaml
# config.yml
processing:
  max_memory_mb: 2048
  chunk_size: 1000
```

## 🔄 Backup & Recovery

### Database Backup
```bash
mysqldump -u root -p ccgl_warehouse > backup_$(date +%Y%m%d).sql
```

### Configuration Backup
```bash
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/ .env
```

### Data Backup
```bash
tar -czf data_backup_$(date +%Y%m%d).tar.gz results/ logs/ sample_data.csv
```

## 📈 Monitoring

### Application Monitoring
```bash
# Check application status
python -c "from ccgl_analytics import *; print('✅ Import successful')"

# Monitor log files
tail -f logs/ccgl_analytics.log
```

### System Monitoring
```bash
# CPU and Memory usage
htop

# Disk usage
df -h

# Network connections
netstat -tulpn | grep :3001
```

## 🆘 Support

For additional support:

1. **Documentation**: Check the `docs/` directory
2. **Issues**: [GitHub Issues](https://github.com/fttat/test_zhangxiaolei/issues)
3. **Examples**: See `examples/` directory
4. **Tests**: Run `pytest tests/`

## 🔄 Updates

### Update Application
```bash
git pull origin main
pip install -r requirements.txt --upgrade
python quick_start.py --setup-only
```

### Database Migration
```bash
# Backup before migration
mysqldump -u root -p ccgl_warehouse > pre_migration_backup.sql

# Run migration scripts
python scripts/migrate_database.py
```