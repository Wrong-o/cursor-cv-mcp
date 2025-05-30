from setuptools import setup, find_packages
import os

# Read dependencies from pyproject.toml manually (simplified)
# In a production environment, you might want to use more sophisticated parsing
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "mss>=9.0.1",
    "numpy>=1.24.0",
    "opencv-python>=4.9.0.80",
    "pydantic>=2.6.0",
    "python-multipart>=0.0.9",
    "pyautogui>=0.9.54",
    "sse-starlette>=1.8.0",
    "structlog>=24.1.0",
    "tenacity>=8.2.0"
]

# Read long description from README if available
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="cursor-cv-mcp",
    version="0.1.0",
    description="Computer Vision MCP Server for Cursor IDE with cross-platform screen capture capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Otto Rongedal",
    author_email="otto@rongedal.se",
    url="https://github.com/yourusername/cursor-cv-mcp",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=dependencies,
    entry_points={
        "console_scripts": [
            "cursor-cv-server=mcp_cv_tool.server:run_server",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: User Interfaces",
    ],
) 