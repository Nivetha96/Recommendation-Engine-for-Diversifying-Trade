from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.2.5',
                     'h5py==2.7.0',
                     'sklearn',
                     'numpy==1.16.2',
                     'tensorflow',
                     'google-cloud-storage==1.14.0',
                     'gcsfs==0.2.1',
                     'pandas==0.24.1']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='datathon application'
)
