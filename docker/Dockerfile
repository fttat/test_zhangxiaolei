# 使用Python 3.12官方镜像作为基础镜像
FROM python:3.12-slim as base

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN groupadd -r ccgl && useradd -r -g ccgl ccgl

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p /app/output /app/logs /app/data

# 设置文件权限
RUN chown -R ccgl:ccgl /app

# 切换到非root用户
USER ccgl

# 暴露端口
EXPOSE 8000 8080 9090

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import ccgl_analytics; print('OK')" || exit 1

# 默认启动命令
CMD ["python", "main.py"]

# 构建开发版本
FROM base as development
USER root
RUN pip install --no-cache-dir pytest pytest-cov black flake8 mypy jupyter
USER ccgl

# 构建生产版本  
FROM base as production
# 移除不必要的文件
RUN rm -rf tests/ docs/ examples/ *.md
# 优化Python字节码
RUN python -m compileall /app