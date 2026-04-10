from setuptools import setup, find_packages

setup(name='autograd',
      version='0.0.1',
      description='A lightweight autograd package',
      url='http://github.com/IParsons1000/autograd',
      author='Ira Parsons',
      author_email='iparsons1000+autograd@gmail.com',
      license='BSD-3-Clause',
      packages=find_packages('src', exclude=['test']),
      package_dir = {"": "src"},
      zip_safe=False)