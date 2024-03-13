from pathlib import Path

from setuptools import setup

project_root = Path(__file__).parent

docs_req_path = project_root / "docs" / "requirements.txt"
docs_require = docs_req_path.read_text().splitlines()

tests_require = [
    "pytest==8.1.1",
    "pytest-cov==4.1.0",
    "flake8==7.0.0",
    "black==24.2.0",
    "isort==5.13.2",
    "coverage==7.4.3",
    "pre-commit==3.6.2",
]

dev_requires = sorted(docs_require + tests_require)


setup(
    name="parsonaut",
    version="1.0.0",
    url="https://github.com/janvainer/parsonaut.git",
    author="Jan Vainer",
    author_email="vainerjan@gmail.com",
    description="Description of my package",
    packages=["parsonaut"],
    extras_require={
        "docs": docs_require,
        "tests": tests_require,
        "dev": dev_requires,
    },
)
