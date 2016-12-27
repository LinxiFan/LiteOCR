'''
@author: jimfan
'''
import os
from setuptools import setup
# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='liteocr',
      version='0.1',
      author='Linxi (Jim) Fan',
      url='http://github.com/LinxiFan/LiteOCR',
      description='Light-weight OCR engine.',
      long_description=read('README.md'),
      keywords='OCR image recognition wrapper Tesseract',
      license='GPLv3',
      packages=['liteocr'],
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Topic :: Scientific/Engineering :: Image Recognition",
          "Topic :: Utilities",
          "Environment :: Console",
          "Programming Language :: Python :: 3.5"
      ],
      zip_safe=False
)