from setuptools import setup

import os

BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='lista_stop',
      py_modules=['lista_stop'],
      install_requires=[
          'torch'
      ],
)
