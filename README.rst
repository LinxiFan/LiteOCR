OCR engine
==========

This library provides a clean interface to segment and recognize text in
an image. It's optimized for printed text, e.g. scanned documents and
website screenshots.

|Python version| |Github release| |PyPI version| |PyPI status|

Installation
------------

``pip install liteocr``

The installation includes both the ``liteocr`` Python3 library and a
command line executable.

Usage
-----

``>> liteocr``
~~~~~~~~~~~~~~

Performs OCR on an image file and writes the recognition results to
JSON.

::

    usage: LiteOCR [-h] [-d] [--extra-whitelist str] [--all-unicode] [--lang str]
                   [--min-text-size int] [--max-text-size int]
                   [--uniformity-thresh :0.0<=float<1.0]
                   [--thin-line-thresh :odd int] [--conf-thresh :0<=int<100]
                   [--box-expand-factor :0.0<=float<1.0]
                   [--horizontal-pooling int]
                   str str

    positional arguments:
      str                   image file
      str                   output JSON file

    optional arguments:
      -h, --help            show this help message and exit
      -d, --display         display recognized bounding boxes and text on top of the image

    engine:
      parameters to liteocr.OCREngine constructor

      --extra-whitelist str
                            string of extra chars for Tesseract to consider only
                            takes effect when all_unicode is False
      --all-unicode         if True, Tesseract will consider all possible unicode
                            characters
      --lang str            language in the text. Defaults to English.

    recognition:
      parameters to OCREngine.recognize() method

      --min-text-size int   min text height/width in pixels, below which will be
                            ignored
      --max-text-size int   max text height/width in pixels, above which will be
                            ignored
      --uniformity-thresh :0.0<=float<1.0
                            ignore a region if the number of pixels neither black
                            nor white < [thresh]
      --thin-line-thresh :odd int
                            remove all lines thinner than [thresh] pixels.can be
                            used to remove the thin borders of web page textboxes.
      --conf-thresh :0<=int<100
                            ignore regions with OCR confidence < thresh.
      --box-expand-factor :0.0<=float<1.0
                            expand the bounding box outwards in case certain chars
                            are cutoff.
      --horizontal-pooling int
                            result bounding boxes will be more connected with more
                            pooling, but large pooling might lower accuracy.

Python3 library
~~~~~~~~~~~~~~~

.. code:: python

    from liteocr import OCREngine, load_img, draw_rect, draw_text, disp

    image_file = 'my_img.png'
    img = load_img(image_file)

    # you can either use context manager or call engine.close() manually at the end.
    with OCREngine() as engine:
        # engine.recognize() can accept a file name, a numpy image, or a PIL image.
        for text, box, conf in engine.recognize(image_file):
            print(box, '\tconfidence =', conf, '\ttext =', text)
            draw_rect(img, box)
            draw_text(img, text, box, color='bw')

    # display the image with recognized text boxes overlaid
    disp(img, pause=False)

Notes
-----

I deprecated and moved the old code into a `separate
folder <https://github.com/LinxiFan/LiteOCR/tree/master/old>`__. The old
API calls Tesseract directly on the entire image. The low recall wasn't
trivial to fix at all, as I realized later:

-  The command-line Tesseract makes really weird global page
   segmentation decisions. It ignores certain text regions with no
   apparent patterns. I have tried many different combinations of a
   handful of tuneable parameters, but none of them helps. My hands are
   tied because Tesseract is poorly documented and very few people asks
   such questions on Stackoverflow.
-  `Tesserocr <https://github.com/sirfz/tesserocr/blob/master/tesserocr.pyx>`__
   is a python package that builds a ``.pyx`` wrapper around Tesseract's
   C++ API. There are a few native API methods that can iterate through
   text regions, but they randomly fail with SegFault (ughh!!!). I spent
   a lot of time trying to fix it, but gave up in despair ...
-  Tesseract is the best open-source OCR engine, which means I don’t
   have other choices. I thought about using Google’s online OCR API,
   but we shouldn’t be bothered by internet connection and API call
   limits.

So I ended up using a new workflow:

1. Apply OpenCV magic to produce better text segmentation.
2. Run Tesseract on each of the segmented text box. It’s much more
   transparent than running on the whole image.
3. Collect text result and mean confidence level (``yield`` as a
   generator).

.. |Python version| image:: https://img.shields.io/pypi/pyversions/liteocr.svg
.. |Github release| image:: https://img.shields.io/github/release/LinxiFan/liteocr.svg
.. |PyPI version| image:: https://img.shields.io/pypi/v/liteocr.svg
.. |PyPI status| image:: https://img.shields.io/pypi/status/liteocr.svg

