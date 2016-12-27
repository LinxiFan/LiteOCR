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


def draw_rect(img, x, y, w, h):
    "Draw a red bounding box"
    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,255), 2)
    

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


class OCREngine():
    def __init__(self, lang='eng'):
        self.tess = PyTessBaseAPI(psm=PSM.SINGLE_BLOCK, lang=lang)
    
    def text_region(self, img, 
                    min_text_size=10,
                    max_text_size=60,
                    uniformity_cutoff=0.1,
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
        img_init = img # preserve initial image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bw = cv2.adaptiveThreshold(img_gray, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 11, 5)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, morph_kernel)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # connect horizontally oriented regions
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                 (horizontal_morph_size, 1))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, morph_kernel)
        # http://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
        _, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, 
                                                  cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
        
            if (w < min_text_size or h < min_text_size
                or h > max_text_size):
                continue

            binary_region = img_bw[y:y+h, x:x+w]
            uniformity = np.count_nonzero(binary_region) / float(w * h)
            if (uniformity > 1 - uniformity_cutoff 
                or uniformity < uniformity_cutoff):
                # ignore mostly white or black regions
                continue
            # the image must be grayscale, otherwise Tesseract will SegFault
            # http://stackoverflow.com/questions/15606379/python-tesseract-segmentation-fault-11
            yield img_gray[y:y+h, x:x+w], Box(x, y, w, h)

    
    def recognize(self, filename, min_text_size=10, max_text_size=60):
        img = load_img(filename, 'np')
        idx = 0 
        for region, outer_box in self.text_region(img, 
                                                  min_text_size=min_text_size,
                                                  max_text_size=max_text_size):
            print(idx)
#             if idx == 11: disp(region)
            idx +=1
            region = np2PIL(region)
            with PyTessBaseAPI(psm=PSM.SINGLE_BLOCK) as self.tess:
                self.tess.SetImage(region)
                print(self.tess.GetUTF8Text(), self.tess.BoundingBox(RIL.TEXTLINE), self.tess.MeanTextConf())
                continue
    #             print('succesful set image', region.size)
    #             print(image_to_text(region))
    #             continue
                components = self.tess.GetComponentImages(RIL.TEXTLINE, True)
                print('Found {} textline image components.'.format(len(components)))
                for _, inner_box, block_id, paragraph_id in components:
                    # box is a dict with x, y, w and h keys
                    inner_box = Box(**inner_box)
                    if inner_box.w < min_text_size or inner_box.h < min_text_size:
                        continue
                    self.tess.SetRectangle(*inner_box)
                    ocr_text = self.tess.GetUTF8Text().strip()
                    conf = self.tess.MeanTextConf()
                    # global coordinate in the image
                    global_box = Box(outer_box.x + inner_box.x,
                                     outer_box.y + inner_box.y,
                                     inner_box.w, inner_box.h)
    #                 yield Blob(ocr_text, global_box, conf)
    
    
    def _deprec_recognize(self, filename, min_text_size=10, max_text_size=60):
        img = load_img(filename, 'np')
        idx = 0 
        for region, outer_box in self.text_region(img, 
                                                  min_text_size=min_text_size,
                                                  max_text_size=max_text_size):
#             print(idx)
#             if idx == 11: disp(region)
#             idx +=1
            region = np2PIL(region)
            self.tess.SetImage(region)
            components = self.tess.GetComponentImages(RIL.TEXTLINE, True)
            print('Found {} textline image components.'.format(len(components)))
            for _, inner_box, block_id, paragraph_id in components:
                # box is a dict with x, y, w and h keys
                inner_box = Box(**inner_box)
                if inner_box.w < min_text_size or inner_box.h < min_text_size:
                    continue
                self.tess.SetRectangle(*inner_box)
                ocr_text = self.tess.GetUTF8Text().strip()
                conf = self.tess.MeanTextConf()
                # global coordinate in the image
                global_box = Box(outer_box.x + inner_box.x,
                                 outer_box.y + inner_box.y,
                                 inner_box.w, inner_box.h)
                yield Blob(ocr_text, global_box, conf)
    
    
    def close(self):
        self.tess.End()
        
    
    def __enter__( self ):
        return self
    

    def __exit__( self, type, value, traceback):
        self.close()


if __name__ == '__main__':
    engine = OCREngine()
#     engine.text_region(load_img(sys.argv[1]))

    if False:
        engine.recognize(sys.argv[1])
        sys.exit(0)

    img = load_img(sys.argv[1])
    for text, box, conf in engine._deprec_recognize(sys.argv[1]):
        print(box, conf, '\t\t', text)
        draw_rect(img, *box)
    disp(img)