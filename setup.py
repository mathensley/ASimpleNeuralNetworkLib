from setuptools import setup, find_packages

setup(
    name="ASimpleNeuralNetworkLib",
    version="0.1",
    packages=find_packages(where=".", include=["simple_nn", "simple_nn.*"]),
    install_requires=[
        "numpy"
    ],
    description="A simple neural network library for practicing deep learning and machine learning concepts.",
    author="Matheus Hensley",
    url="https://github.com/mathensley/ASimpleNeuralNetworkLib.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)