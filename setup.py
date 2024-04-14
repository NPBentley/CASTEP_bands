from setuptools import find_packages
from setuptools import setup, Extension

setup(name="CASTEPbands",
      version="1.3.0",
      description="CASTEP module for plotting band structures and phonon dispersions.",
      packages=find_packages(),
      url="https://github.com/NPBentley/CASTEP_bands.git",
      author="Zachary Hawkhead, Nathan Philip Bentley, Visagan Ravindran",
      author_email="zachary.hawkhead@ymail.com",
      license="MIT",
      python_requires='>=3.6',
      install_requires=["numpy",
                        "matplotlib",
                        "ase>=3.18.1"],
      )
