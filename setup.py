from setuptools import setup


setup(
    name='vixstructure',
    version='0.3',
    description=('Exploring the VIX term structure with Machine Learning '
                  'for my Bachelor\'s thesis.'),
    url='https://github.com/leyhline/vix-term-structure',
    author='Thomas Leyh',
    author_email='leyht@informatik.uni-freiburg.de',
    license='MIT',
    keywords='deeplearning machinelearning finance',
    packages=['vixstructure'],
    install_requires=['tensorflow',
                      'pandas',
                      'numpy',
                      'scipy',
                      'jupyter',
                      'matplotlib',
                      'h5py',
                      'lazy',
                      'tables'],
    python_requires='>=3.5',
)
