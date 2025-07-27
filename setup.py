#!/usr/bin/env python3
"""
CCGL Warehouse Management Data Analysis Platform
Enterprise-level data analysis system with MCP architecture
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ccgl-analytics",
    version="1.0.0",
    author="CCGL Analytics Team",
    author_email="analytics@ccgl.com",
    description="Enterprise warehouse management data analysis platform with MCP architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fttat/test_zhangxiaolei",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "pytest>=7.0.0",
            "pytest-asyncio>=0.19.0",
            "pytest-mock>=3.8.0",
        ],
        "docker": [
            "gunicorn>=20.0.0",
            "uvicorn>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ccgl-analytics=main:main",
            "ccgl-mcp=main_mcp:main",
            "ccgl-llm=main_llm:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ccgl_analytics": ["templates/*", "static/*"],
        "config": ["*.json", "*.yml"],
    },
)