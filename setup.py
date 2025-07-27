from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ccgl-analytics",
    version="1.0.0",
    author="CCGL Team",
    author_email="team@ccgl.com",
    description="Enterprise-level warehouse management data analysis platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fttat/test_zhangxiaolei",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "web": [
            "streamlit>=1.15.0",
            "dash>=2.7.0",
        ],
        "ml": [
            "xgboost>=1.6.0",
            "lightgbm>=3.3.0",
            "catboost>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ccgl-analytics=ccgl_analytics.cli:main",
            "ccgl-mcp=ccgl_mcp_servers.cli:main",
            "ccgl-dashboard=ccgl_analytics.modules.web_dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ccgl_analytics": ["config/*.json", "templates/*.html"],
        "ccgl_mcp_servers": ["config/*.json"],
    },
)