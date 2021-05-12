import os
import shutil
from PIL import Image
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from .utils import (resize_to_A4, deskew, denoise, crop_empty, sort_contours, get_lines_images, detect_text,
                    binarize_image, get_nonempty_bbox)


def extract_table_pos(table_contour):
    contour_x_vals = table_contour[:, :, 0].flatten().flatten()
    contour_y_vals = table_contour[:, :, 1].flatten().flatten()
    
    smallest_x, smallest_y = min(contour_x_vals), min(contour_y_vals)
    largest_x, largest_y = max(contour_x_vals), max(contour_y_vals)
    
    return smallest_x, largest_x, smallest_y, largest_y


def get_rectangles_image(vertical_lines_img, horizontal_lines_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)

    img_final_bin = binarize_image(img_final_bin)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_final_bin = cv2.dilate(img_final_bin, kernel, iterations=2)
    return img_final_bin


def find_table_and_bottom(img, img_bin, rects_img):
    contours, hierarchy = cv2.findContours(rects_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = img.shape[0] * img.shape[1]

    # Find first contour thats <= 80% of the whole image
    table_contour = None
    for contour in sorted_contours:
        cont_area = cv2.contourArea(contour)
        if cont_area <= 0.8 * img_area and cont_area >= 0.3 * img_area:
            table_contour = contour

    if table_contour is None:
        raise Exception('Table contour not found on image')

    table_x_start, table_x_end, table_y_start, table_y_end = extract_table_pos(table_contour)

    table_loc = (table_x_start, table_x_end, table_y_start, table_y_end)
    bottom_loc = get_nonempty_bbox(img_bin[table_y_end:])

    bottom_loc = (bottom_loc[0], bottom_loc[1], bottom_loc[2]+table_y_end, bottom_loc[3]+table_y_end)

    return table_loc, bottom_loc


def extract_table_boxes(img, vertical_lines_img, horizontal_lines_img):
    """Cuts the table into lines and each line into cells"""
    
    vert_contours, hierarchy = cv2.findContours(vertical_lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        assert len(line.keys()) == len(fields.keys())
        lines.append(line)
    return lines


def draw_bboxes(img, bboxes):
    temp_img = np.array(img)
    for bbox in bboxes:
        x, y, w, h = bbox
        x1 = x + w
        y1 = y + h
        cv2.rectangle(temp_img, (x, y), (x1, y1), 0, 3)
    return temp_img


def extract_bottom_boxes(img, horizontal_lines_img, approx_line_height=75, padding=7, top_cutoff=30):
    """Cuts the bottom (подвал) into boxes for each line and one box for account number"""
    horiz_contours, hierarchy = cv2.findContours(horizontal_lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    horiz_contours, horiz_bboxes = sort_contours(horiz_contours, method='top-to-bottom')
    
    horiz_bboxes = [bbox for bbox in horiz_bboxes if bbox[1] > 20 and bbox[2] >= img.shape[1]-img.shape[1]/2 and bbox[3] <= 10]

    # Extract images of lines
    line_imgs = []
    
    y_from = max(horiz_bboxes[0][1] - approx_line_height - padding, 0)
    y_to = min(horiz_bboxes[0][1] + padding, img.shape[1])
    line_imgs.append(img[y_from:y_to])
    
    for i in range(len(horiz_bboxes)-1):
        y_from = max(horiz_bboxes[i][1] - padding, 0)
        y_to = min(horiz_bboxes[i+1][1] + padding, img.shape[1])
        line_img = img[y_from:y_to]
        line_imgs.append(line_img)
    
    # Extract account number at the bottom
    last_horiz_line_y = horiz_bboxes[-1][1]
    account_number_line_img = img[last_horiz_line_y+top_cutoff:]
    
    text_bboxes = detect_text(account_number_line_img)

    if not text_bboxes:
        raise Exception("Account number not detected")

    account_number_bbox = text_bboxes[0]
    account_number_img = account_number_line_img[account_number_bbox[1]:account_number_bbox[1]+account_number_bbox[3], account_number_bbox[0]:account_number_bbox[0]+account_number_bbox[2]]
    return line_imgs, account_number_img


def save_np_img(img, path):
    Image.fromarray(img).save(path)


def get_contours_image(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tmp_img = np.array(img)
    cv2.drawContours(tmp_img, contours, -1, 0, 3)
    return tmp_img


def extract_blocks(img, debug=False):

    if debug:
        if os.path.exists('tmp_imgs'):
            shutil.rmtree('tmp_imgs')

        os.makedirs('tmp_imgs')


    img = deskew(img)
    save_np_img(img, 'tmp_imgs/0_0_deskewed.png')

    img = resize_to_A4(img)
    save_np_img(img, 'tmp_imgs/0_1_resized.png')

    img = denoise(img)
    save_np_img(img, 'tmp_imgs/0_2_denoised.png')

    img_blur = cv2.GaussianBlur(img, (3,3), 0)
    save_np_img(img_blur, 'tmp_imgs/0_3_blur.png')

    img_bin = binarize_image(img_blur)
    save_np_img(img_bin, 'tmp_imgs/1_0_binarized_img.png')

    vertical_lines_img, horizontal_lines_img = get_lines_images(img_bin)
    save_np_img(vertical_lines_img, 'tmp_imgs/1_1_vertical_lines.png')
    save_np_img(horizontal_lines_img, 'tmp_imgs/1_2_horizontal_lines.png')

    rects_img = get_rectangles_image(vertical_lines_img, horizontal_lines_img)
    save_np_img(rects_img, 'tmp_imgs/1_3_rectangles.png')


    # Countours only for debug output
    if debug:
        contours_img = get_contours_image(rects_img)
        save_np_img(contours_img, 'tmp_imgs/3_4_contours.png')

    table_loc, bottom_loc = find_table_and_bottom(img, img_bin, rects_img)

    table_x_start, table_x_end, table_y_start, table_y_end = table_loc
    bottom_x_start, bottom_x_end, bottom_y_start, bottom_y_end = bottom_loc

    table_img = img[table_y_start:table_y_end, table_x_start:table_x_end]
    bottom_img = img[bottom_y_start:bottom_y_end, bottom_x_start:bottom_x_end]

    save_np_img(table_img, 'tmp_imgs/4_0_table.png')
    save_np_img(bottom_img, 'tmp_imgs/4_1_bottom.png')

    table_vertical_lines = vertical_lines_img[table_y_start:table_y_end, table_x_start:table_x_end]
    table_horizontal_lines = horizontal_lines_img[table_y_start:table_y_end, table_x_start:table_x_end]
    lines = extract_table_boxes(table_img, table_vertical_lines, table_horizontal_lines)

    os.makedirs('tmp_imgs/lines')
    for i, line in enumerate(lines):
        for key in line.keys():
            save_np_img(line[key], f'tmp_imgs/lines/{i}_{key}.png')

    bottom_horizontal_lines = horizontal_lines_img[bottom_y_start:bottom_y_end, bottom_x_start:bottom_x_end]

    bottom_line_imgs, bottom_account_number = extract_bottom_boxes(bottom_img, bottom_horizontal_lines)

    os.makedirs('tmp_imgs/bottom')
    save_np_img(bottom_account_number, 'tmp_imgs/bottom/account_number.png')
    for i, line_img in enumerate(bottom_line_imgs):
        save_np_img(line_img, f'tmp_imgs/bottom/line_{i}.png')

    result = {
        'table_lines': lines,
        'bottom_lines': bottom_line_imgs,
        'account_number': bottom_account_number
    }
    return result
