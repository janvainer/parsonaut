from pathlib import Path

from setuptools import setup

project_root = Path(__file__).parent

install_requires = ["PyYAML==6.0.2"]

tests_require = [
    "pytest==8.1.1",
    "pytest-cov==4.1.0",
    "flake8==7.0.0",
    "black==24.2.0",
    "isort==5.13.2",
    "coverage==7.4.3",
    "codecov==2.1.13",
    "pre-commit==3.6.2",
    "torch",
]

dev_requires = sorted(tests_require + ["pre-commit==3.6.2"])


setup(
    name="parsonaut",
    version="1.0.2",
    url="https://github.com/janvainer/parsonaut.git",
    author="Jan Vainer",
    author_email="vainerjan@gmail.com",
    description="Auto-configure (not only) torch experiments from the CLI.",
    packages=["parsonaut"],
    install_requires=install_requires,
    extras_require={
        "tests": tests_require,
        "dev": dev_requires,
    },
)
