import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tripletboost",
    version="0.0.1",
    author="mperrot",
    author_email="michael.perrot@tuebingen.mpg.de",
    description="TripletBoost, a comparison-based algorithm for classification.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mperrot/TripletBoost",
    packages=setuptools.find_packages(exclude=['test*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3"
    ],
    install_requires=[
        "numpy>=1.15.4"
    ],
    python_requires="~=3.5"
)
