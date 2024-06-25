from setuptools import setup

setup(
      name='tcrcube',
      version='0.0.1',
      description='Predicting pMHC-TCR interactions using 3D outer product of pre-computed and trained sequence representations',
      packages=['tcrcube'],
      install_requires=[
            'numpy',
            'torch',
            'scipy',
      ],
)