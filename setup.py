from distutils.core import setup
import setuptools

"""
To upload: 
python setup.py bdist_wheel && conda deactivate && twine upload dist/*.whl
"""

setup(name='potter',
      version='0.0.1',
      description='Some routines for working with simple model potentials',
      author='Ian H. Bell',
      author_email='ian.bell@nist.gov',
      packages = ['potter'],
      zip_safe = False
     )

