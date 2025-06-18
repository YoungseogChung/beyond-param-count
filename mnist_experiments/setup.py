
from setuptools import setup, find_packages

setup(
    name="sparse-moe",
    version="1.0.0",
    description="A package for sparse-moe",
    author="Youngseog Chung, Dhruv Malik",
    author_email="youngsec@cs.cmu.edu",
    packages=find_packages(),
    url="https://github.com/YoungseogChung/sparse-moe",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8"
)
