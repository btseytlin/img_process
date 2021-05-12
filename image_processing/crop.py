import os
import shutil
from PIL import Image
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from .utils import (resize_to_A4, deskew, denoise, crop_empty, sort_contours, get_lines_images, detect_text, binarize_image)


def extract_table_pos(img, table_contour):
    contour_x_vals = table_contour[:, :, 0].flatten().flatten()
    contour_y_vals = table_contour[:, :, 1].flatten().flatten()
    
    smallest_x, smallest_y = min(contour_x_vals), min(contour_y_vals)
    largest_x, largest_y = max(contour_x_vals), max(contour_y_vals)

    smallest_x, smallest_y, largest_x, largest_y
    
    return smallest_x, largest_x, smallest_y, largest_y


def get_rectangles_image(vetrical_lines_img, horizontal_lines_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(vetrical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)

    img_final_bin = binarize_image(img_final_bin)
    return img_final_bin


def find_table_and_bottom(img, rects_img):
    contours, hierarchy = cv2.findContours(rects_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    table_contour = sorted_contours[1]
    
    table_x_start, table_x_end, table_y_start, table_y_end = extract_table_pos(img, table_contour)
    
    table_img = img[table_y_start:table_y_end, table_x_start:table_x_end]
    
    bottom_img = crop_empty(img[table_y_end:])
    
    return table_img, bottom_img


def extract_table_boxes(img):
    """Cuts the table into lines and each line into cells"""
    vetrical_lines_img, horizontal_lines_img = get_lines_images(img)
    
    vert_contours, hierarchy = cv2.findContours(vetrical_lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    vert_contours, vert_bboxes = sort_contours(vert_contours, method='left-to-right')
    
    vert_bboxes = [bbox for bbox in vert_bboxes if bbox[0] > 20 and bbox[2] <= 10 and bbox[3] >= img.shape[0]-50]
    
    horiz_contours, hierarchy = cv2.findContours(horizontal_lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    horiz_contours, horiz_bboxes = sort_contours(horiz_contours, method='top-to-bottom')
    
    horiz_bboxes = [bbox for bbox in horiz_bboxes if bbox[1] > 20 and bbox[3] <= 10 and bbox[2] >= img.shape[1]-50]
    
    fields = {
        0: 'name',
        1: 'born',
        2: 'address',
        3: 'passport',
        4: 'date',
    }
    lines = []
    for i in range(0, len(horiz_bboxes)-1):
        line = {}
        padding = -2

        line_y_start = max(horiz_bboxes[i][1]-padding, 0)
        line_y_end = min(horiz_bboxes[i+1][1]+padding, img.shape[0])
        
        for j in range(0, len(vert_bboxes)-2):
            box_x_start = max(vert_bboxes[j][0]-padding, 0)
            box_x_end = min(vert_bboxes[j+1][0]+padding, img.shape[1])
        
            line[fields[j]] = img[line_y_start:line_y_end, box_x_start:box_x_end]
        # assert len(line.keys()) == len(fields.keys())
        lines.append(line)
    # assert len(lines) == 5
    return lines


def extract_bottom_boxes(img):
    """Cuts the bottom (подвал) into boxes for each line and one box for account number"""

    vetrical_lines_img, horizontal_lines_img = get_lines_images(img)
    
    horiz_contours, hierarchy = cv2.findContours(horizontal_lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    horiz_contours, horiz_bboxes = sort_contours(horiz_contours, method='top-to-bottom')
    
    horiz_bboxes = [bbox for bbox in horiz_bboxes if bbox[1] > 20 and bbox[2] >= img.shape[1]-img.shape[1]/2 and bbox[3] <= 10]
    
    # Extract images of lines
    padding = 7
    line_imgs = []
    
    approx_line_height = 75
    y_from = max(horiz_bboxes[0][1] - approx_line_height - padding, 0)
    y_to = min(horiz_bboxes[0][1] + padding, img.shape[1])
    line_imgs.append(img[y_from:y_to])
    
    for i in range(len(horiz_bboxes)-1):
        y_from = max(horiz_bboxes[i][1] - padding, 0)
        y_to = min(horiz_bboxes[i+1][1] + padding, img.shape[1])
        line_img = img[y_from:y_to]
        line_imgs.append(line_img)
    
    # Extract account number at the bottom
    top_cutoff = 30
    last_horiz_line_y = horiz_bboxes[-1][1]
    account_number_line_img = img[last_horiz_line_y+top_cutoff:]
    
    text_bboxes = detect_text(account_number_line_img)
    assert len(text_bboxes) == 1
    account_number_bbox = text_bboxes[0]
    account_number_img = account_number_line_img[account_number_bbox[1]:account_number_bbox[1]+account_number_bbox[3], account_number_bbox[0]:account_number_bbox[0]+account_number_bbox[2]]
    return line_imgs, account_number_img


def save_np_img(img, path):
    Image.fromarray(img).save(path)


def extract_blocks(img, debug=False):

    if debug:
        if os.path.exists('tmp_imgs'):
            shutil.rmtree('tmp_imgs')

        os.makedirs('tmp_imgs')


    # img = deskew(img)
    # save_np_img(img, 'tmp_imgs/0_deskewed.png')

    # img = resize_to_A4(img)
    # save_np_img(img, 'tmp_imgs/1_resized.png')

    img = denoise(img)
    save_np_img(img, 'tmp_imgs/2_denoised.png')

    img_blur = cv2.GaussianBlur(img,(5,5),0)
    save_np_img(img_blur, 'tmp_imgs/2_blur.png')

    img_bin = binarize_image(img_blur)
    save_np_img(img_bin, 'tmp_imgs/3_0_binarized_img.png')

    vetrical_lines_img, horizontal_lines_img = get_lines_images(img_bin)
    save_np_img(vetrical_lines_img, 'tmp_imgs/3_1_vertical_lines.png')
    save_np_img(horizontal_lines_img, 'tmp_imgs/3_2_horizontal_lines.png')

    rects_img = get_rectangles_image(vetrical_lines_img, horizontal_lines_img)
    save_np_img(rects_img, 'tmp_imgs/3_3_rectangles.png')

    # Temp contours
    contours, hierarchy = cv2.findContours(rects_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    tmpimg = np.array(img)
    cv2.drawContours(tmpimg, contours, -1, 0, 3)
    plt.imshow(tmpimg)
    plt.show()
    save_np_img(tmpimg, 'tmp_imgs/3_4_contours.png')

    table_img, bottom_img = find_table_and_bottom(img, rects_img)
    save_np_img(table_img, 'tmp_imgs/4_0_table.png')
    save_np_img(bottom_img, 'tmp_imgs/4_1_bottom.png')

    lines = extract_table_boxes(table_img)

    bottom_line_imgs, bottom_account_number = extract_bottom_boxes(bottom_img) 

    result = {
        'table_lines': lines,
        'bottom_lines': bottom_line_imgs,
        'account_number': bottom_account_number
    }
    return result
