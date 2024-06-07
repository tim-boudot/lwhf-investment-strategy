from setuptools import find_packages
from setuptools import setup

setup(name='lwhf',
      version="0.0.1",
      description="Le Wagon Hedge Fund",
      license="TBLJ",
      author="Rich people",
      author_email="no contact pls",
      #url="https://github.com/lewagon/taxi-fare",
      #install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
