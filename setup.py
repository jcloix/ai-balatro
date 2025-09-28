from setuptools import setup, find_packages

setup(
    name="balatro",
    version="0.1",
    packages=find_packages(),  # automatically finds all packages (annotate_dataset, etc.)
    install_requires=[
        "streamlit",
        "Pillow",
    ],
)
