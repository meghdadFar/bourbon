import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rl-nlp",
    version="0.0.0",
    author="meghdadFar",
    author_email="meghdad.farahmand@gmail.com",
    description="Reinforcement Learning for NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meghdadFar/rl-nlp",
    packages=setuptools.find_packages(),
    # install_requires = [],
    # scripts=['bin/downloads.py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)