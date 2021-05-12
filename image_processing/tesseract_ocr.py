import pytesseract
import string

from .utils import deskew

def text_postprocess(txt):
    return ''.join(ch for ch in txt if ch.isalnum() or ch in ['.', ',', ' '])

def preprocess_ocr(img):
    final_img = deskew(img)
    return final_img

def img_to_string(img):
    custom_config = r'--psm 6'
    return text_postprocess(pytesseract.image_to_string(preprocess_ocr(img), config=custom_config, lang='rus'))
