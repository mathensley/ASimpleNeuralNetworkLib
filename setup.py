from setuptools import setup, find_packages

setup(
    name="ASimpleNeuralNetworkLib",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy"
    ],
    description="A simple neural network library for practicing deep learning and machine learning concepts.",
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    author="Matheus Hensley",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)