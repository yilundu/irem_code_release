from setuptools import setup
from distutils.command.build import build
from setuptools.command.install import install

from setuptools.command.develop import develop

import os
BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='metal',
      py_modules=['metal'],
      install_requires=[
          'torchvision==v0.4.2',
          'pillow==v6.2.1',
          'torch==v1.3.1',
          'torchmeta==v1.2.2',
          'tqdm',
      ]
)
