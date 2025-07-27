# 🧪 Testing Guide for CCGL Analytics

## 📋 Test Overview

This guide covers testing procedures for the CCGL Analytics platform, including unit tests, integration tests, and end-to-end validation.

## 🚀 Quick Test Run

### Run All Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run complete test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ccgl_analytics --cov-report=html
```

### Quick Functionality Test
```bash
# Test basic functionality
python quick_start.py --basic-only

# Test MCP architecture
python main_mcp.py --start-mcp-servers --demo-workflow

# Test interactive mode
python main_mcp.py --full-mcp --start-mcp-servers
```

## 🔬 Unit Tests

### Core Modules Testing

```bash
# Test data connection module
pytest tests/test_modules.py::test_data_connection -v

# Test preprocessing module
pytest tests/test_modules.py::test_data_preprocessing -v

# Test analysis core
pytest tests/test_modules.py::test_analysis_core -v
```

### MCP Testing

```bash
# Test MCP configuration
pytest tests/test_mcp_clients.py::test_mcp_config -v

# Test MCP servers
pytest tests/test_mcp_clients.py::test_preprocessing_server -v

# Test MCP orchestrator
pytest tests/test_mcp_clients.py::test_orchestrator -v
```

## 🔄 Integration Tests

### End-to-End Workflow

```bash
# Test complete analysis pipeline
pytest tests/test_integration.py::test_complete_workflow -v

# Test MCP workflow
pytest tests/test_integration.py::test_mcp_workflow -v

# Test LLM integration
pytest tests/test_llm_integration.py::test_llm_analysis -v
```

## 🗄️ Database Testing

### MySQL Testing

```bash
# Start test database
docker run --name test-mysql -e MYSQL_ROOT_PASSWORD=test -e MYSQL_DATABASE=ccgl_test -p 3307:3306 -d mysql:8.0

# Run database tests
DB_HOST=localhost DB_PORT=3307 DB_PASSWORD=test pytest tests/test_database.py -v

# Cleanup
docker stop test-mysql && docker rm test-mysql
```

### Test Data

```bash
# Create test dataset
python -c "
from ccgl_analytics.modules.data_connection import create_sample_data
df = create_sample_data()
df.head(100).to_csv('tests/fixtures/test_data_small.csv', index=False)
"
```

## 🐳 Docker Testing

### Container Testing

```bash
# Build test container
docker build -t ccgl-analytics:test .

# Run container tests
docker run --rm ccgl-analytics:test python -m pytest tests/ -v

# Test web dashboard
docker run -p 8080:8080 ccgl-analytics:test &
curl http://localhost:8080/health
docker stop $(docker ps -q --filter ancestor=ccgl-analytics:test)
```

## 📊 Performance Testing

### Load Testing

```bash
# Test with large dataset
python -c "
from ccgl_analytics.modules.data_connection import create_sample_data
import time

# Test with 10k records
start = time.time()
df = create_sample_data()  # Default 1k records
# Scale up to 10k
for i in range(9):
    df = pd.concat([df, create_sample_data()], ignore_index=True)
print(f'Generated {len(df)} records in {time.time() - start:.2f}s')
df.to_csv('large_test_data.csv', index=False)
"

# Run analysis on large dataset
time python main.py -c config.yml
```

### Memory Testing

```bash
# Monitor memory usage
python -c "
import psutil
import subprocess
import time

# Start analysis process
proc = subprocess.Popen(['python', 'main.py'])
pid = proc.pid

# Monitor memory
max_memory = 0
while proc.poll() is None:
    try:
        p = psutil.Process(pid)
        memory_mb = p.memory_info().rss / 1024 / 1024
        max_memory = max(max_memory, memory_mb)
        time.sleep(1)
    except psutil.NoSuchProcess:
        break

print(f'Peak memory usage: {max_memory:.2f} MB')
"
```

## ⚡ Stress Testing

### Concurrent MCP Servers

```bash
# Test multiple MCP instances
for i in {1..5}; do
    python main_mcp.py --start-mcp-servers --demo-workflow &
done

# Wait for all to complete
wait
```

### API Load Testing

```bash
# Install hey for load testing
# go install github.com/rakyll/hey@latest

# Test MCP endpoints (if HTTP exposed)
hey -n 1000 -c 10 http://localhost:3001/health
```

## 🧩 Component Testing

### Individual Component Tests

```bash
# Test logger
python -c "
from ccgl_analytics.utils.logger import get_logger
logger = get_logger('test')
logger.info('Test message')
print('✅ Logger working')
"

# Test data connection
python -c "
from ccgl_analytics.modules.data_connection import DataConnection, create_sample_data
df = create_sample_data()
print(f'✅ Generated {len(df)} sample records')
"

# Test preprocessing
python -c "
from ccgl_analytics.modules.data_preprocessing import DataPreprocessor
from ccgl_analytics.modules.data_connection import create_sample_data
preprocessor = DataPreprocessor()
df = create_sample_data()
processed = preprocessor.preprocess(df, ['quality_check'])
print('✅ Preprocessing working')
"
```

## 🔍 Debugging Tests

### Debug Mode

```bash
# Run tests with debug output
pytest tests/ -v -s --tb=long

# Debug specific test
pytest tests/test_modules.py::test_analysis_core -v -s --pdb
```

### Log Analysis

```bash
# Check test logs
tail -f logs/ccgl_analytics_*.log

# Filter error logs
grep ERROR logs/ccgl_analytics_*.log
```

## 📈 Test Coverage

### Generate Coverage Report

```bash
# HTML coverage report
pytest tests/ --cov=ccgl_analytics --cov-report=html
open htmlcov/index.html

# Terminal coverage
pytest tests/ --cov=ccgl_analytics --cov-report=term-missing

# XML coverage (for CI)
pytest tests/ --cov=ccgl_analytics --cov-report=xml
```

### Coverage Targets

- **Unit Tests**: > 80% coverage
- **Integration Tests**: > 70% coverage
- **Critical Paths**: > 95% coverage

## 🔄 Continuous Testing

### GitHub Actions

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Daily scheduled runs

### Pre-commit Testing

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🐛 Common Test Issues

### Import Errors

```bash
# Fix Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use pytest with path
python -m pytest tests/
```

### Database Connection

```bash
# Check MySQL connection
mysql -h localhost -P 3306 -u root -p -e "SELECT 1;"

# Test with different credentials
DB_HOST=localhost DB_USER=test pytest tests/test_database.py
```

### Memory Issues

```bash
# Reduce test data size
pytest tests/ -k "not large_dataset"

# Run tests individually
pytest tests/test_modules.py::test_data_connection
```

## 📊 Test Metrics

### Performance Benchmarks

| Component | Target Time | Memory Limit |
|-----------|-------------|--------------|
| Data Loading (1k records) | < 1s | < 50MB |
| Preprocessing | < 2s | < 100MB |
| Analysis (basic) | < 5s | < 200MB |
| MCP Startup | < 3s | < 150MB |

### Quality Metrics

| Metric | Target |
|--------|--------|
| Test Coverage | > 80% |
| Test Pass Rate | > 95% |
| Performance Regression | < 10% |
| Memory Growth | < 20% |

## 🆘 Troubleshooting

### Test Failures

1. **Check Dependencies**: Ensure all required packages are installed
2. **Verify Environment**: Check environment variables and configuration
3. **Review Logs**: Check application and test logs for errors
4. **Isolate Issue**: Run failing test individually with debug output

### Common Solutions

```bash
# Clear cache
pytest --cache-clear

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Reset test environment
rm -rf logs/ results/ dashboard/
python quick_start.py --setup-only
```