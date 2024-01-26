from setuptools import setup, find_packages

with open("requirements.txt", 'r') as ifile:
    requirements = ifile.read().splitlines()

nb_requirements = [
    'nbconvert>=6.1.0',
    'nbformat>=5.1.3',
    'notebook>=6.4.11',
    'jupyter>=7.0.0',
    'jupyterlab>=3.4.3',
    'ipython>=7.27.0',
]

setup(
    name="CellStitch",
    version="1.0.0",
    description="Cellstitch: 3D cellular anisotropic image segmentation via optimal transport",
    authors=["Yining Liu", "Yinuo Jin"],
    # url="https://cellstitch.readthedocs.io",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=requirements,
    # package_data = ""; include_package_data=True
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)