'''
OCR API for Tesseract
'''
import os
import sys
import cv2
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL, PSM, image_to_text
from collections import namedtuple
from IPython import embed as REPL

def np2PIL(img):
    "Numpy array to PIL.Image"
    return Image.fromarray(img)


def PIL2np(img):
    "PIL.Image to numpy array"
    assert isinstance(img, Image.Image)
    print(img.size)
    return np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)


def load_img(filename, format='np'):
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
    

def disp(img, pause=True):
    "Display an image"
    save_img(img, '_temp.png')
    os.system('open _temp.png')
    if pause:
        input('continue ...')


Box = namedtuple('Box', ['x', 'y', 'w', 'h'])


class OCREngine():
    def __init__(self, lang='eng'):
        self.tess = PyTessBaseAPI(psm=PSM.SINGLE_BLOCK, lang=lang)
    
    
    def text_region(self, img, 
                    min_text_size=10,
                    horizontal_morph_size=25):
        """ 
        Generator: segment bounding boxes of text regions
        http://stackoverflow.com/questions/23506105/extracting-text-opencv
        
        Args:
          img: numpy array
          min_text_size: minimal text height/width in pixels, below which will be ignored
          horizontal_morph_size: the larger the more connected, but shorter texts
              might be overlooked. Tradeoff between connectedness and recall. 
        """
        img_init = img
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_bw = cv2.threshold(img_bw, 0, 255, 
                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_bw = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2RGB)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, morph_kernel)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # connect horizontally oriented regions
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                 (horizontal_morph_size, 1))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, morph_kernel)
        _, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, 
                                                  cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
        
            if w < min_text_size or h < min_text_size:
                continue

            binary_region = img_bw[y:y+h, x:x+w, 0]
            uniformity = np.count_nonzero(binary_region) / float(w * h)
            if (uniformity > 0.9 or uniformity < 0.1):
                # mostly white or black image will cause SegFault in Tesseract
                continue
            yield img_bw[y:y+h, x:x+w, :], Box(x, y, w, h)
            # draw rectangle around contour on original image
            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

    
    def recognize(self, filename, min_text_size=10):
        img = load_img(filename, 'np')
        idx = 0 
        for region, global_box in self.text_region(img, 
                                                   min_text_size=min_text_size):
            
#             print(idx)
#             region = np2PIL(region)
#             if idx == 13:
#                 disp(region)
#             idx +=1
            
            self.tess.SetImage(region)
            print('succesful set image', region.size)
            boxes = self.tess.GetComponentImages(RIL.TEXTLINE, True)
            print('Found {} textline image components.'.format(len(boxes)))
            for i, (_, box, block_id, paragraph_id) in enumerate(boxes):
                # box is a dict with x, y, w and h keys
                print(box)
                self.tess.SetRectangle(*Box(**box))
                ocrResult = self.tess.GetUTF8Text()
                conf = self.tess.MeanTextConf()
                print((u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
                       "confidence: {1}, text: {2}").format(i, conf, ocrResult, **box))
    
    
    def close(self):
        self.tess.End()
        
    
    def __enter__( self ):
        return self
    

    def __exit__( self, type, value, traceback):
        self.close()


if __name__ == '__main__':
    engine = OCREngine()
    engine.recognize(sys.argv[1])