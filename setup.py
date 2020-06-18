from setuptools import setup, find_packages
from codecs import open
from os import path
import os

working_dir = path.abspath(path.dirname(__file__))
ROOT = os.path.abspath(os.path.dirname(__file__))

# Read the README.
with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()

setup(name='robograph',
      version='0.1',
      description='Certified Robustness of Graph Convolution Networks for Graph Classification under Topological Attacks',
      long_description=README,
      long_description_content_type='text/markdown',
      packages=find_packages(exclude=['tests*']),
      setup_requires=["numpy", "numba"],
      install_requires=["numpy", "matplotlib", "numba"],
      )
