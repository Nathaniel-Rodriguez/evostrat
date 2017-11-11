from setuptools import setup

setup(name='evostrat',
    version='0.1',
    description='Implements ES algorithms for python3',
    author='Nathaniel Rodriguez',
    packages=['evostrat'],
    url='https://github.com/Nathaniel-Rodriguez/evostrat.git',
    install_requires=[
          'numpy',
          'mpi4py'
      ],
    include_package_data=True)