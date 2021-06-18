import cv2
import math
import numpy as np
from deskew import determine_skew
from matplotlib import pyplot as plt

def rotate(image, angle, background):
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def deskew(img):
    angle = determine_skew(img)
    if abs(angle) <= 45:
        img = rotate(img, angle, 255)
    else:
        angle = 0
    return img, angle

def rotate_by_angle(img, angle, fill=255):
    return rotate(img, angle, fill)

def denoise(img):
    dst = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    return dst


def resize(img, target_width=1280):
    """Resize the page so that largest side is target_width and aspec ratio is preserved"""
    w, h = img.shape[1], img.shape[0]
    inter = cv2.INTER_AREA
    if w > target_width:
        inter = cv2.INTER_CUBIC

    r = target_width / w
    height = int(h * r)

    resized = cv2.resize(img, (target_width, height), interpolation=inter)
    return resized


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def get_nonempty_bbox(img, padding=2):
    coords = cv2.findNonZero(img)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box

    y_from = max(0, y - padding)
    y_till = min(img.shape[0], y + h + padding)
    x_from = max(0, x - padding)
    x_till = min(img.shape[1], x + w + padding)
    return (x_from, x_till, y_from, y_till)


def crop_rect(img, x_from, x_till, y_from, y_till):
    cropped = img[y_from:y_till, x_from:x_till]  # Crop the image - note we do this on the original image
    return cropped


def crop_empty(img, img_bin, padding=2):
    """Crops image to non-empty bounding rect"""

    (x_from, x_till, y_from, y_till) = get_nonempty_bbox(img_bin, padding=padding)
    return crop_rect(img, x_from, x_till, y_from, y_till)


def binarize_image(img):
    # Thresholding the image
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Invert the image
    img_bin = 255 - img_bin
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    # img_bin = cv2.dilate(img_bin, kernel, iterations=1)

    return img_bin


def get_lines_images(img_bin):
    """
        img_bin: binarized image
    """

    # Defining a kernel length
    kernel_length = np.array(img_bin).shape[1]//100

    # A vertical kernel of (1 X kernel_length), which will detect all the vertical lines from the image.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=3)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, horiz_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, horiz_kernel, iterations=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    vertical_lines_img = cv2.dilate(vertical_lines_img, kernel, iterations=2)
    horizontal_lines_img = cv2.dilate(horizontal_lines_img, kernel, iterations=2)

    return vertical_lines_img, horizontal_lines_img


def detect_text(img, area_threshold=5000):
    blur = cv2.GaussianBlur(img, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    dilate = cv2.dilate(thresh, kernel, iterations=4)
    
    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    bboxes = []
    ROI_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        
        if area > area_threshold:
            x,y,w,h = cv2.boundingRect(c)
            
            
            bboxes.append((x, y, w, h))
    return bboxes

