'''
@author: jimfan
'''
import argparse
import json
from liteocr import *


def run_ocr(image_file, engine_args, method_args, is_display):
    """
    Returns:
      list of blobs (converted to dict)
    """
    engine = OCREngine(**engine_args)
    img = load_img(image_file)
    output = []
    for blob in engine.recognize(image_file, **method_args):
        if is_display:
            text, box, conf = blob
            print(box, '\tconf =', conf, '\t\t', text)
            draw_rect(img, box)
            draw_text(img, text, box, color='bw')
        output.append(blob._asdict())
    if is_display:
        disp(img, pause=False)
    return output


def main():
    parser = argparse.ArgumentParser(prog='LiteOCR',
                                     formatter_class=argparse.MetavarTypeHelpFormatter)
    parser.add_argument('image_file', type=str, help='image file')
    parser.add_argument('output_file', type=str, help='output JSON file')
    parser.add_argument('-d', '--display', action='store_true',
                        help='display recognized bounding boxes and text on top of the image')

    # engine constructor args
    g1 = parser.add_argument_group('engine', 
                                   'parameters to liteocr.OCREngine constructor')
    g1.add_argument('--extra-whitelist', type=str, default='',
                  help='string of extra chars for Tesseract to consider ' 
                    'only takes effect when all_unicode is False')
    g1.add_argument('--all-unicode', action='store_true',
                    help='if True, Tesseract will consider all possible unicode characters')
    g1.add_argument('--lang', type=str, default='eng',
                    help='language in the text. Defaults to English.')
    # recognize() args
    g2 = parser.add_argument_group('recognition', 'parameters to OCREngine.recognize() method')
    g2.add_argument('--min-text-size', type=int, default=MIN_TEXT_SIZE,
                    help='min text height/width in pixels, below which will be ignored')
    g2.add_argument('--max-text-size', type=int, default=MAX_TEXT_SIZE,
                    help='max text height/width in pixels, above which will be ignored')
    g2.add_argument('--uniformity-thresh', type=float, 
                    default=UNIFORMITY_THRESH, metavar=':0.0<=float<1.0',
                    help='ignore a region if the number of pixels neither black nor white < [thresh]')
    g2.add_argument('--thin-line-thresh', type=int, 
                    default=THIN_LINE_THRESH, metavar=':odd int',
                    help='remove all lines thinner than [thresh] pixels.'
                    'can be used to remove the thin borders of web page textboxes.')
    g2.add_argument('--conf-thresh', type=int, 
                    default=CONF_THRESH, metavar=':0<=int<100',
                    help='ignore regions with OCR confidence < thresh.')
    g2.add_argument('--box-expand-factor', type=float, 
                    default=BOX_EXPAND_FACTOR, metavar=':0.0<=float<1.0',
                    help='expand the bounding box outwards in case certain chars are cutoff.')
    g2.add_argument('--horizontal-pooling', type=int, default=HORIZONTAL_POOLING,
                    help='result bounding boxes will be more connected with more pooling,'
                        ' but large pooling might lower accuracy. ')

    args = vars(parser.parse_args())

    image_file = args.pop('image_file')
    output_file = args.pop('output_file')
    is_display = args.pop('display')

    engine_args = {}
    for key in ['extra_whitelist', 'all_unicode', 'lang']:
        engine_args[key] = args.pop(key)
    
    output = run_ocr(image_file, engine_args, args, is_display)
    
    with open(output_file, 'w') as fp:
        json.dump(output, fp, indent=4)


if __name__ == '__main__':
    main()