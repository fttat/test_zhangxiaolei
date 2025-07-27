FROM python:3.9-slim

LABEL maintainer="CCGL Team <team@ccgl.com>"
LABEL description="CCGL 仓储管理系统 - 现代化智能仓储管理与数据分析平台"

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建非root用户
RUN useradd --create-home --shell /bin/bash ccgl && \
    chown -R ccgl:ccgl /app
USER ccgl

# 创建必要的目录
RUN mkdir -p logs data models cache

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 暴露端口
EXPOSE 8000 8001 8002 8003 8004

# 设置启动命令
CMD ["python", "main.py", "-c", "config.yml", "-w"]