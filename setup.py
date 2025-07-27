#!/usr/bin/env python
"""
CCGL Analytics System Setup Configuration
"""

from setuptools import setup, find_packages
import os
import sys

# Read version from __init__.py
version = {}
with open("ccgl_analytics/__init__.py") as fp:
    exec(fp.read(), version)

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f 
                if line.strip() and not line.startswith('#')]

# Base requirements
requirements = read_requirements('requirements.txt')

# Development requirements
dev_requirements = [
    'pytest>=7.4.0',
    'pytest-cov>=4.1.0',
    'pytest-asyncio>=0.21.0',
    'pytest-mock>=3.11.0',
    'black>=23.7.0',
    'flake8>=6.0.0',
    'mypy>=1.5.0',
    'pre-commit>=3.3.0',
]

# Optional requirements for different features
extras_require = {
    'dev': dev_requirements,
    'mcp': [
        'mcp>=1.0.0',
        'websockets>=11.0.0',
        'aiohttp>=3.8.0',
    ],
    'llm': [
        'openai>=1.0.0',
        'anthropic>=0.25.0',
        'zhipuai>=2.0.0',
        'dashscope>=1.14.0',
    ],
    'web': [
        'streamlit>=1.28.0',
        'fastapi>=0.100.0',
        'uvicorn[standard]>=0.23.0',
        'gradio>=4.0.0',
    ],
    'ml': [
        'scikit-learn>=1.3.0',
        'xgboost>=1.7.0',
        'lightgbm>=3.3.0',
        'statsmodels>=0.14.0',
        'prophet>=1.1.4',
    ],
    'viz': [
        'plotly>=5.15.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'quickchart>=3.1.0',
    ],
    'all': [],
}

# Combine all extras
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    name="ccgl-analytics",
    version=version.get('__version__', '1.0.0'),
    author="CCGL Analytics Team",
    author_email="support@ccgl-analytics.com",
    description="Centralized Control and Group Learning - Advanced Analytics Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fttat/test_zhangxiaolei",
    project_urls={
        "Bug Tracker": "https://github.com/fttat/test_zhangxiaolei/issues",
        "Documentation": "https://github.com/fttat/test_zhangxiaolei/docs",
        "Source Code": "https://github.com/fttat/test_zhangxiaolei",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "ccgl-analyze=ccgl_analytics.cli:main_analysis",
            "ccgl-mcp=ccgl_analytics.cli:main_mcp",
            "ccgl-llm=ccgl_analytics.cli:main_llm",
            "ccgl-web=ccgl_analytics.cli:main_web",
            "ccgl-setup=ccgl_analytics.cli:setup_system",
        ],
    },
    include_package_data=True,
    package_data={
        "ccgl_analytics": [
            "config/*.yml",
            "config/*.json",
            "templates/*.html",
            "static/*.css",
            "static/*.js",
        ],
    },
    zip_safe=False,
    keywords=[
        "analytics", 
        "machine-learning", 
        "data-science", 
        "mcp", 
        "llm", 
        "ai", 
        "database",
        "visualization",
        "clustering",
        "anomaly-detection"
    ],
    platforms=["any"],
    license="MIT",
    test_suite="tests",
    tests_require=dev_requirements,
)