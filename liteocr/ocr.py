'''
OCR API for Tesseract
'''
import os
import sys
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL, PSM, image_to_text
from collections import namedtuple
try:
    import cv2
except ImportError:
    print('Please install OpenCV first. `pip install opencv-python` or '
          '`yes | conda install -c https://conda.binstar.org/menpo opencv3` '
          'if you use Anaconda. Make sure `import cv2` works after installation.', 
          file=sys.stderr)
    sys.exit(1)

def is_PIL(img):
    return isinstance(img, Image.Image)


def is_np(img):
    return isinstance(img, np.ndarray)


def np2PIL(img):
    "Numpy array to PIL.Image"
    return Image.fromarray(img)


def PIL2np(img):
    "PIL.Image to numpy array"
    assert is_PIL(img)
    print(img.size)
    return np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)


def load_img(filename, format='np'):
    assert os.path.exists(filename), 'image file {} does not exist'.format(filename)
    format = format.lower()
    if format == 'np':
        return cv2.imread(filename)
    elif format == 'pil':
        return Image.open(filename)
    else:
        raise ValueError('format must be either "np" or "PIL"')


def save_img(img, filename):
    "Save a numpy or PIL image to file"
    if isinstance(img, Image.Image):
        img.save(filename)
    else:
        cv2.imwrite(filename, img)


def get_np_img(obj):
    if isinstance(obj, str):
        return load_img(obj, 'np')
    elif is_PIL(obj):
        return PIL2np(obj)
    elif is_np(obj):
        return obj
    else:
        raise ValueError('{} must be string (filename), ndarray, or PIL.Image'
                         .format(obj))
        
        
def crop(img, box):
    """
    Crop a numpy image with bounding box (x, y, w, h)
    """
    x, y, w, h = box
    return img[y:y+h, x:x+w]


def draw_rect(img, box):
    "Draw a red bounding box"
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,255), 2)


def draw_text(img, text, box, color='bw'):
    """
    FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_COMPLEX_SMALL
    FONT_HERSHEY_DUPLEX
    FONT_HERSHEY_PLAIN
    FONT_HERSHEY_SCRIPT_COMPLEX
    FONT_HERSHEY_SCRIPT_SIMPLEX
    FONT_HERSHEY_SIMPLEX
    FONT_HERSHEY_TRIPLEX
    FONT_ITALIC
    """
    x, y, w, h = box
    font = cv2.FONT_HERSHEY_DUPLEX
    region = crop(img, box)
    if color == 'bw':
        brightness = np.mean(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))
        if brightness > 127:
            font_color = (0,0,0)
        else:
            font_color = (255,255,255)
    elif color == 'color':
        mean_bg = np.round(np.mean(region, axis=(0, 1)))
        font_color = tuple(map(int, np.array((255,255,255)) - mean_bg))
    else:
        font_color = (255, 0, 0) # blue

    cv2.putText(img, text, (x, y+h), font, 1, font_color, 2)
    

def disp(img, pause=True):
    "Display an image"
    save_img(img, '_temp.png')
    os.system('open _temp.png')
    if pause:
        input('continue ...')
        

# Bounding box
Box = namedtuple('Box', ['x', 'y', 'w', 'h'])

# Recognition result
Blob = namedtuple('Blob', ['text', 'box', 'conf'])

# default params
MIN_TEXT_SIZE = 15
MAX_TEXT_SIZE = 200
UNIFORMITY_THRESH = 0.1
THIN_LINE_THRESH = 7
CONF_THRESH = 20
BOX_EXPAND_FACTOR = 0.15
HORIZONTAL_POOLING = 25

PSM_MODE = PSM.AUTO
# PSM_MODE = PSM.SINGLE_BLOCK

class OCREngine():
    def __init__(self, extra_whitelist='', all_unicode=False, lang='eng'):
        """
        Args:
          extra_whitelist: string of extra chars for Tesseract to consider
              only takes effect when all_unicode is False
          all_unicode: if True, Tess will consider all possible unicode characters
          lang: OCR language
        """
        self.tess = PyTessBaseAPI(psm=PSM_MODE, lang=lang)
        self.is_closed = False
        if all_unicode:
            self.whitelist_chars = None
        else:
            self.whitelist_chars = ("abcdefghijklmnopqrstuvwxyz"
                                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                    "1234567890"
                                    r"~!@#$%^&*()_+-={}|[]\:;'<>?,./" '"'
                                    "©") + extra_whitelist
            self.tess.SetVariable('tessedit_char_whitelist', self.whitelist_chars)
    
    
    def check_engine(self):
        if self.is_closed:
            raise RuntimeError('OCREngine has been closed.')
        
    
    def recognize(self, image, 
                  min_text_size=MIN_TEXT_SIZE,
                  max_text_size=MAX_TEXT_SIZE,
                  uniformity_thresh=UNIFORMITY_THRESH,
                  thin_line_thresh=THIN_LINE_THRESH,
                  conf_thresh=CONF_THRESH,
                  box_expand_factor=BOX_EXPAND_FACTOR,
                  horizontal_pooling=HORIZONTAL_POOLING):
        """ 
        Generator: Blob
        http://stackoverflow.com/questions/23506105/extracting-text-opencv
        
        Args:
          input_image: can be one of the following types:
            - string: image file path
            - ndarray: numpy image
            - PIL.Image.Image: PIL image
          min_text_size: 
            min text height/width in pixels, below which will be ignored
          max_text_size: 
            max text height/width in pixels, above which will be ignored
          uniformity_thresh (0.0 < _ < 1.0): 
            remove all black or all white regions
            ignore a region if the number of pixels neither black nor white < [thresh]
          thin_line_thresh (must be odd int): 
            remove all lines thinner than [thresh] pixels.
            can be used to remove the thin borders of web page textboxes. 
          conf_thresh (0 < _ < 100): 
            ignore regions with OCR confidence < thresh.
          box_expand_factor (0.0 < _ < 1.0):
            expand the bounding box outwards in case certain chars are cutoff. 
          horizontal_pooling: 
            result bounding boxes will be more connected with more pooling, 
            but large pooling might lower accuracy. 
        """
        self.check_engine()
        # param sanity check
        assert max_text_size > min_text_size > 0
        assert 0.0 <= uniformity_thresh < 1.0
        assert thin_line_thresh % 2 == 1
        assert 0 <= conf_thresh < 100
        assert 0.0 <= box_expand_factor < 1.0
        assert horizontal_pooling > 0
        
        image = get_np_img(image) 
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_bw = cv2.adaptiveThreshold(img_gray, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 11, 5)
        img = img_gray
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        # cut off all gray pixels < 30.
        # `cv2.THRESH_BINARY | cv2.THRESH_OTSU` is also good, but might overlook certain light gray areas
        _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
        # connect horizontally oriented regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                           (horizontal_pooling, 1))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # remove all thin textbox borders (e.g. web page textbox)
        if thin_line_thresh > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                               (thin_line_thresh, thin_line_thresh))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        # http://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
        _, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, 
                                                  cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = box = Box(*cv2.boundingRect(contour))
            # remove regions that are beyond size limits
            if (w < min_text_size or h < min_text_size
                or h > max_text_size):
                continue
            # remove regions that are almost uniformly white or black
            binary_region = crop(img_bw, box)
            uniformity = np.count_nonzero(binary_region) / float(w * h)
            if (uniformity > 1 - uniformity_thresh 
                or uniformity < uniformity_thresh):
                continue
            # expand the borders a little bit to include cutoff chars
            expansion = int(min(h, w) * box_expand_factor)
            x = max(0, x - expansion)
            y = max(0, y - expansion)
            h, w = h + 2 * expansion, w + 2 * expansion
            if h > w: # further extend the long axis
                h += 2 * expansion
            elif w > h:
                w += 2 * expansion
            # image passed to Tess should be grayscale.
            # http://stackoverflow.com/questions/15606379/python-tesseract-segmentation-fault-11
            box = Box(x, y, w, h)
            ocr_text, conf = self.run_tess(crop(img_gray, box))
            if conf > conf_thresh:
                yield Blob(ocr_text, box, conf)
    
    
    def _experiment_segment(self, img, 
                    min_text_size=MIN_TEXT_SIZE,
                    max_text_size=MAX_TEXT_SIZE,
                    uniformity_thresh=UNIFORMITY_THRESH,
                    horizontal_pooling=HORIZONTAL_POOLING):
        """ 
        PRIVATE: experiment only
        """
        img_init = img # preserve initial image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bw = cv2.adaptiveThreshold(img_gray, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 11, 5)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, morph_kernel)
        disp(img)
#         morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         img = cv2.dilate(img, morph_kernel)
        # OTSU thresholding
#         _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
#         img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,2)
        disp(img)
        # connect horizontally oriented regions
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                 (horizontal_pooling, 1))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, morph_kernel)
        disp(img)
        
        if 0:
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (horizontal_pooling, 3))
            img = cv2.erode(img, morph_kernel, iterations=1)
            disp(img)
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6))
            img = cv2.dilate(img, morph_kernel, iterations=1)
        elif 1:
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, morph_kernel)
        disp(img)

        # http://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
        _, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, 
                                                  cv2.CHAIN_APPROX_SIMPLE)
        img_copy = np.copy(img_init)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            draw_rect(img_copy, x, y, w, h)
        
            if (w < min_text_size or h < min_text_size
                or h > max_text_size):
                continue

            binary_region = img_bw[y:y+h, x:x+w]
            uniformity = np.count_nonzero(binary_region) / float(w * h)
            if (uniformity > 1 - uniformity_thresh 
                or uniformity < uniformity_thresh):
                # ignore mostly white or black regions
#                 print(w, h)
#                 disp(binary_region)
                continue
            # the image must be grayscale, otherwise Tesseract will SegFault
            # http://stackoverflow.com/questions/15606379/python-tesseract-segmentation-fault-11
            draw_rect(img_init, x, y, w, h)
        disp(img_copy)
        disp(img_init, 0)

    
    def run_tess(self, img):
        """
        Tesseract python API source code:
        https://github.com/sirfz/tesserocr/blob/master/tesserocr.pyx
        
        Returns:
          (ocr_text, confidence)
        """
        if isinstance(img, np.ndarray):
            img = np2PIL(img)
        self.tess.SetImage(img)
        ocr_text = self.tess.GetUTF8Text().strip()
        conf = self.tess.MeanTextConf()
        return ocr_text, conf
    
    
    def _deprec_run_tess(self, img):
        "GetComponentImages throws SegFault randomly. No way to fix. :("
        if isinstance(img, np.ndarray):
            img = np2PIL(img)

        components = self.tess.GetComponentImages(RIL.TEXTLINE, True)
        for _, inner_box, block_id, paragraph_id in components:
            # box is a dict with x, y, w and h keys
            inner_box = Box(**inner_box)
            if inner_box.w < MIN_TEXT_SIZE or inner_box.h < MIN_TEXT_SIZE:
                continue
            self.tess.SetRectangle(*inner_box)
            ocr_text = self.tess.GetUTF8Text().strip()
            conf = self.tess.MeanTextConf()
            yield ocr_text, inner_box, conf
    
    
    def close(self):
        self.tess.End()
        self.is_closed = True
        
    
    def __enter__( self ):
        return self
    

    def __exit__( self, type, value, traceback):
        self.close()