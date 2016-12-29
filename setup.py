'''
@author: jimfan
'''
import os
from setuptools import setup

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


setup(name='liteocr',
      version='0.2.1',
      author='Linxi (Jim) Fan',
      author_email='jimfan@cs.stanford.edu',
      url='http://github.com/LinxiFan/LiteOCR',
      description='Light-weight OCR engine.',
      long_description=read('README.rst'),
      keywords='OCR image recognition Tesseract',
      license='GPLv3',
      packages=['liteocr'],
      entry_points={
        'console_scripts': [
            'liteocr = liteocr.run_ocr:main'
        ]
      },
      classifiers=[
          "Development Status :: 4 - Beta",
          "Topic :: Scientific/Engineering :: Image Recognition",
          "Topic :: Utilities",
          "Environment :: Console",
          "Programming Language :: Python :: 3"
      ],
      install_requires=read('requirements.txt').strip().splitlines(),
      include_package_data=True,
      zip_safe=False
)
