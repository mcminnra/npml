# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('./README.md') as f:
    readme = f.read()

with open('./LICENSE') as f:
    license = f.read()

deps = [
    'matplotlib',
    'numpy',
    'pandas',
    'scikit-learn',
    'scipy',
    'seaborn',
    'termcolor'
]

dev_deps = deps + [
    'flake8'
]

setup(
    name='npml',
    version='0.0.0',
    description='No Packages Machine Learning',
    long_description=readme,
    author='Ryder McMinn',
    author_email='mcminnra@gmail.com',
    url='https://github.com/mcminnra/npml',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=deps,
    extras_require={'test': dev_deps},
    python_requires='>=3.6'
)
