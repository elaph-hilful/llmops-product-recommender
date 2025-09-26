from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="LLMOPS PRODUCT RECOMMENDER",
    version="0.1",
    author="Ashabul Elaph Hilful",
    packages=find_packages(),
    install_requires = requirements,
)