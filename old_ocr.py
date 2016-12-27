'''
OCR API for Tesseract
'''
import shutil
import sys
import cv2
import tempfile
import bs4 # BeautifulSoup
import re
import subprocess
from functools import lru_cache


class OCRException(Exception):
    pass


class OCREngine():
    def __init__(self):
        self.bin_path = shutil.which('tesseract')
        self.img_file = None
        self.hocr_file = None 
        if not self.bin_path:
            raise OCRException('Tesseract binary not found')
    

    def recognize(self, img, delete=True):
        """
        Params:
          img: - str: file name of the image
               - np array
          delete: True to delete the temporary files
        Returns:
          HOCR.Page object
        """
        is_file = isinstance(img, str)
        with tempfile.NamedTemporaryFile(suffix='.png', 
                                         prefix='ocr_',
                                         delete=delete or is_file) as png_temp:
            if is_file:
                png_file = img
            else:
                # save numpy image to a temp file
                png_file = png_temp.name
                OCREngine.save_np_img(png_file, img)
            self.temp_img_file = png_file
            
            with tempfile.NamedTemporaryFile(suffix='.hocr',
                                             prefix='ocr_',
                                             delete=delete) as hocr_tmp:
                self.temp_hocr_file = hocr_tmp.name
                ret = subprocess.call([self.bin_path, 
                                       self.temp_img_file, 
                                       self.temp_hocr_file[:-5], # remove .hocr
                                       'hocr'])
                if ret != 0:
                    raise OCRException('Call to {} returns non-zero exit code: {}'
                                       .format(self.bin_path, ret))
                # assume only one page
                # tempfile context auto deletes the files upon exit
                self.hocr = HOCR(self.temp_hocr_file)
                return self.hocr.pages[0]
    
    
    @staticmethod
    def save_np_img(png_file, img):
        cv2.imwrite(png_file, img)
        # alternative way to resize and set DPI
#         from PIL import Image
#         import scipy.misc
#         img = scipy.misc.imresize(img, 3.0) # must be float
#         img = Image.fromarray(img)
#         img.save(png_file, dpi=(600, 600))


class HOCR():
    """
    HOCR specification: https://kba.github.io/hocr-spec/
    """
    def __init__(self, html_file):
        try:
            content = open(html_file).read()
        except Exception as e:
            raise OCRException(e)

        self.soup = bs4.BeautifulSoup(content, 'lxml')
        self.pages = [HOCR.Page(elem) for elem 
                      in self.soup.find_all(class_='ocr_page')]
        if not self.pages:
            raise OCRException('No HOCR page found in {}'.format(html_file))
    
    
    @staticmethod
    def correct(text):
        "possible mis-recognition"
        return text.replace('|', 'l')


    class Box():
        def __init__(self, text=None, *, left=0, right=0, top=0, bottom=0):
            # Parse the text string representation if given.
            if text is not None:
                left, top, right, bottom = map(int, text.split())

            self.left = left
            self.right = right
            self.top = top
            self.bottom = bottom

        @property
        def width(self):
            return self.right - self.left

        @property
        def height(self):
            return self.bottom - self.top
        
        @property
        def corners(self):
            """
            (upper_left, bottom_right)
            """
            return (self.left, self.top), (self.right, self.bottom)

        def __repr__(self):
            return '<Box({}, {}, {}, {})>'.format(
                        self.left, self.top, self.right, self.bottom)

    class _Base():
        _included_classes = {}
        
        def __init__(self, element):
            """
            @param[in] element
                XML node for the OCR element.
            """
            # Store the element for later reference.
            self._element = element

            # Create an element cache.
            self._cache = {}
            CLASSES = {
                'words': {'name': 'liteocr.?_word', 'class': HOCR.Word},
                'lines': {'name': 'ocr_line', 'class': HOCR.Line},
                'paragraphs': {'name': 'ocr_par', 'class': HOCR.Paragraph},
                'blocks': {'name': 'ocr_carea', 'class': HOCR.Block}
            }
            for cl in CLASSES:
                CLASSES[cl]['name'] = re.compile(CLASSES[cl]['name'])
            self.CLASSES = CLASSES

            # Parse the properties of the HOCR element.
            properties = element.get('title', '').split(';')
            for prop in properties:
                name, value = prop.split(maxsplit=1)
                if name == 'bbox':
                    self.box = HOCR.Box(value)

                elif name == 'image':
                    self.image = value.strip('" ')

        def __dir__(self):
            return super().__dir__() + list(self._included_classes)

        def __getattr__(self, name):
            if name in self._cache:
                return self._cache[name]

            # Parse the named OCR elements.
            if name in self._included_classes:
                info = self.CLASSES[name]
                nodes = self._element.find_all(class_=info['name'])
                self._cache[name] = elements = list(map(info['class'], nodes))
                return elements
            else:
                raise AttributeError(name)


    class Word(_Base):
        _included_classes = {}

        def __init__(self, element):
            super().__init__(element)

            # A word element is bold if its text node is wrapped in a <strong/>.
            self.bold = bool(element.find('strong'))
            # A word element is italic if its text node is wrapped in a <em/>.
            self.italic = bool(element.find('em'))
            self.text = HOCR.correct(element.text)

        def __str__(self):
            return self.text


    class Line(_Base):
        _included_classes = {'words'}
        
        @lru_cache(maxsize=None)
        def __str__(self):
            return ' '.join(map(str, self.words))


    class Paragraph(_Base):
        _included_classes = {'lines', 'words'}

        @lru_cache(maxsize=None)
        def __str__(self):
            return '\n'.join(map(str, self.lines))


    class Block(_Base):
        _included_classes = {'paragraphs', 'lines', 'words'}

        @lru_cache(maxsize=None)
        def __str__(self):
            return '\n'.join(map(str, self.lines))


    class Page(_Base):
        _included_classes = {'blocks', 'paragraphs', 'lines', 'words'}

        @lru_cache(maxsize=None)
        def __str__(self):
            return '\n'.join(map(str, self.blocks))