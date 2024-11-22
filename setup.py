from setuptools import setup, find_packages

setup(
    name="chess-bench",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "Pillow",
    ],
    python_requires=">=3.8",
)
