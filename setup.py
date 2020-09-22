  
from setuptools import find_packages, setup

setup(
    name="text2sql",
    version="0.1",
    author="Yash Bonde",
    author_email="bonde.yash97@gmail.com",
    description="Convert natural language questions to SQL query",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yashbonde/text2sql",
    package_dir={"": "src"},
    packages=find_packages("src"),

    # adding requirements here ensures that we don't go and start compiling torch :P
    install_requires=[
        "numpy",
        # "tokenizers == 0.8.1.rc2", # use the best tokenizers
        "tqdm >= 4.27", # progress bars in model download and training scripts
        "torch", # optimizer library
        "transformers", # huggingface transformer's package
        "tensorboard", # tensorboard supported
        "sentencepiece" # tokeniser, probably already installed
    ]
)