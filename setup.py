"""
Setup configuration for llm-apis package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-apis",
    version="0.1.0",
    author="LLM APIs Team",
    author_email="your-email@example.com",
    description="A unified wrapper for various LLM provider APIs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/llm-apis",
    packages=find_packages(
        exclude=["test*", "tests*", "*.tests", "*.test*"]
    ),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
    },
    keywords="llm, ai, openai, anthropic, perplexity, ollama, api, wrapper",
    project_urls={
        "Bug Reports": "https://github.com/your-username/llm-apis/issues",
        "Source": "https://github.com/your-username/llm-apis",
        "Documentation": "https://github.com/your-username/llm-apis#readme",
    },
)