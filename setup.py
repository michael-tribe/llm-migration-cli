from pathlib import Path

from setuptools import find_packages, setup


setup(
    name="llm_migrate",
    version=Path("VERSION").read_text().strip(),
    description="LLM Migration Evaluation Tool",
    url="https://github.com/michael-tribe/llm-migration-cli",
    author="Michael van Elk",
    author_email="michaelve@tribe.ai",
    entry_points={"console_scripts": ["llm_migrate = src.cli:cli"]},
    include_package_data=True,
    install_requires=Path("requirements.txt").read_text().splitlines(),
    package_data={"llm_migrate": ["src/**/*.txt"]},
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.10",
)
