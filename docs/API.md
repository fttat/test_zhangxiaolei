# CCGL API 文档

## 概述

CCGL 仓储管理系统提供 RESTful API 和 MCP 协议接口，支持多种数据分析和管理功能。

## 基础信息

- **Base URL**: `http://localhost:8000/api/v1`
- **认证方式**: Bearer Token
- **数据格式**: JSON
- **编码**: UTF-8

## 认证

所有API请求需要在Header中包含认证令牌：

```http
Authorization: Bearer YOUR_TOKEN_HERE
```

## 核心API端点

### 数据分析 API

#### 1. 聚类分析
```http
POST /analysis/clustering
Content-Type: application/json

{
  "data": [...],
  "algorithm": "kmeans",
  "n_clusters": 5
}
```

**响应示例:**
```json
{
  "status": "success",
  "result": {
    "n_clusters": 5,
    "labels": [0, 1, 2, 1, 0, ...],
    "centroids": [...],
    "silhouette_score": 0.72
  }
}
```

#### 2. 异常检测
```http
POST /analysis/anomaly
Content-Type: application/json

{
  "data": [...],
  "algorithm": "isolation_forest",
  "contamination": 0.1
}
```

#### 3. 关联规则挖掘
```http
POST /analysis/association
Content-Type: application/json

{
  "data": [...],
  "min_support": 0.01,
  "min_confidence": 0.5
}
```

### 数据管理 API

#### 1. 获取库存数据
```http
GET /inventory?limit=100&offset=0&category=electronics
```

#### 2. 更新库存
```http
PUT /inventory/{product_id}
Content-Type: application/json

{
  "quantity": 100,
  "location": "A1-B2-C3"
}
```

#### 3. 创建商品
```http
POST /products
Content-Type: application/json

{
  "sku": "PROD-001",
  "name": "示例商品",
  "category_id": 1,
  "supplier_id": 1,
  "unit_price": 99.99
}
```

### LLM集成 API

#### 1. 自然语言查询
```http
POST /llm/query
Content-Type: application/json

{
  "query": "显示最近一周的异常数据",
  "model": "openai"
}
```

#### 2. 生成报告
```http
POST /llm/report
Content-Type: application/json

{
  "analysis_results": {...},
  "report_type": "summary",
  "language": "zh"
}
```

### 仪表板 API

#### 1. 创建仪表板
```http
POST /dashboard
Content-Type: application/json

{
  "title": "库存监控仪表板",
  "charts": [...]
}
```

#### 2. 获取仪表板数据
```http
GET /dashboard/{dashboard_id}/data
```

## MCP 协议接口

### 服务器端点

- **预处理服务器**: `http://localhost:8001`
- **机器学习服务器**: `http://localhost:8002`
- **仪表板服务器**: `http://localhost:8003`
- **LLM集成服务器**: `http://localhost:8004`

### MCP 消息格式

```json
{
  "jsonrpc": "2.0",
  "method": "execute_task",
  "params": {
    "task_type": "clustering",
    "data": {...}
  },
  "id": 1
}
```

## 错误处理

### 错误响应格式
```json
{
  "error": {
    "code": 400,
    "message": "Invalid request",
    "details": "Missing required field: data"
  }
}
```

### 状态码说明

- `200 OK` - 请求成功
- `201 Created` - 资源创建成功
- `400 Bad Request` - 请求参数错误
- `401 Unauthorized` - 认证失败
- `403 Forbidden` - 权限不足
- `404 Not Found` - 资源不存在
- `429 Too Many Requests` - 请求频率过高
- `500 Internal Server Error` - 服务器内部错误

## 限流规则

- **用户级别**: 1000 请求/小时
- **IP级别**: 5000 请求/小时
- **API密钥级别**: 10000 请求/小时

## SDK 和示例

### Python SDK
```python
from ccgl_analytics import CCGLClient

client = CCGLClient(api_key="your_api_key")
result = client.clustering_analysis(data, algorithm="kmeans")
```

### JavaScript SDK
```javascript
const client = new CCGLClient('your_api_key');
const result = await client.clusteringAnalysis(data, {algorithm: 'kmeans'});
```

## 版本信息

- **当前版本**: v1.0.0
- **API版本**: v1
- **更新日期**: 2024-01-01

## 支持与反馈

- **文档**: https://docs.ccgl.com
- **GitHub**: https://github.com/fttat/test_zhangxiaolei
- **邮箱**: api-support@ccgl.com