import os
import shutil
from PIL import Image
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from .utils import (resize, deskew, denoise, crop_empty, crop_rect, sort_contours, get_lines_images, detect_text,
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

    # Find largest contour thats <= 80% of the whole image

    table_contour = None
    max_contour_area = None
    for contour in sorted_contours:
        cont_area = cv2.contourArea(contour)
        if cont_area <= 0.8 * img_area and cont_area >= 0.4 * img_area:
            if table_contour is None or cont_area > max_contour_area:
                table_contour = contour
                max_contour_area = cont_area

    if table_contour is None:
        raise Exception('Table contour not found on image')

    table_x_start, table_x_end, table_y_start, table_y_end = extract_table_pos(table_contour)

    table_loc = (table_x_start, table_x_end, table_y_start, table_y_end)
    bottom_loc = get_nonempty_bbox(img_bin[table_y_end:, table_x_start:table_x_end])

    bottom_loc = (bottom_loc[0], bottom_loc[1], bottom_loc[2]+table_y_end, bottom_loc[3]+table_y_end)

    return table_loc, bottom_loc


def extract_table_boxes(img, vertical_lines_img, horizontal_lines_img):
    """Cuts the table into lines and each line into cells"""

    vert_contours, hierarchy = cv2.findContours(vertical_lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    vert_contours, vert_bboxes = sort_contours(vert_contours, method='left-to-right')

    # bounding boxes look like this: (x, y, w, h)
    # filter out bboxes for vertical lines:
    # 1. Bbox is not on the left border of image
    # 2. Bbox has small width
    # 3. Bbox has spans the whole image from top to bottom
    def is_vertical_line_box(bbox):
        if bbox[2] >= 0.02 * img.shape[1]:
            # Too wide
            return False
        if bbox[3] < 0.95 * img.shape[0]:
            # Does not span whole image from top to bottom
            return False
        return True

    vert_bboxes = [bbox for bbox in vert_bboxes if is_vertical_line_box(bbox)]

    expected_vert_lines = 8
    assert len(vert_bboxes) == expected_vert_lines,  f'Expected to find {expected_vert_lines} vertical lines in table, found {len(vert_bboxes)}'
    
    horiz_contours, hierarchy = cv2.findContours(horizontal_lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    horiz_contours, horiz_bboxes = sort_contours(horiz_contours, method='top-to-bottom')

    # filter out bboxes for horizontal lines:
    # 1. Bbox is not on the top of image
    # 2. Bbox has small height
    # 3. Bbox has spans the whole image from left to right
    def is_horizontal_line_box(bbox):
        if bbox[3] >= 0.02 * img.shape[0]:
            # Height of box too large
            return False
        if bbox[2] < 0.95 * img.shape[1]:
            # Does not span whole image from left to right
            return False
        # print('yes')
        return True
    horiz_bboxes = [bbox for bbox in horiz_bboxes if is_horizontal_line_box(bbox)]

    expected_horiz_lines = 7
    assert len(horiz_bboxes) == expected_horiz_lines, f'Expected to find {expected_horiz_lines} horizontal lines in table, found {len(horiz_bboxes)}'

    fields = {
        0: 'name',
        1: 'born',
        2: 'address',
        3: 'passport',
        4: 'date',
    }
    lines = []
    for i in range(1, len(horiz_bboxes)-1):
        line = {}
        padding = -2

        line_y_start = max(horiz_bboxes[i][1]-padding, 0)
        line_y_end = min(horiz_bboxes[i+1][1]+padding, img.shape[0])

        for j in range(1, len(vert_bboxes)-2):
            box_x_start = max(vert_bboxes[j][0]-padding, 0)
            box_x_end = min(vert_bboxes[j+1][0]+padding, img.shape[1])
            box_loc = (box_x_start, box_x_end, line_y_start, line_y_end)
            line[fields[j-1]] = crop_rect(img, *box_loc)
        assert len(line.keys()) == len(fields.keys()), f'Expected to find all required columns for line, found only: {str(line.keys())}'
        lines.append(line)
    assert len(lines) == 5, f'Expected to find 5 lines in table, found: {len(lines)}'
    return lines


def draw_bboxes(img, bboxes):
    temp_img = np.array(img)
    for bbox in bboxes:
        x, y, w, h = bbox
        x1 = x + w
        y1 = y + h
        cv2.rectangle(temp_img, (x, y), (x1, y1), 0, 3)
    return temp_img


def extract_bottom_boxes(img, horizontal_lines_img, padding=5):
    """Cuts the bottom (подвал) into boxes for each line and one box for account number"""
    horiz_contours, hierarchy = cv2.findContours(horizontal_lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    horiz_contours, horiz_bboxes = sort_contours(horiz_contours, method='top-to-bottom')


    # filter out bboxes for horizontal lines:
    # 1. Bbox is not on the top of image
    # 2. Bbox has small height
    # 3. Bbox has spans most of the image from left to right
    def is_horizontal_line_box(bbox):
        if bbox[0] >= 0.3 * img.shape[1]:
            # Does not start on the left of image
            return False
        if bbox[3] >= 0.1 * img.shape[0]:
            # Height of box too large
            return False
        if bbox[2] < 0.7 * img.shape[1]:
            # Does not span whole image from left to right
            return False
        # print('yes')
        return True

    horiz_bboxes = [bbox for bbox in horiz_bboxes if is_horizontal_line_box(bbox)]
    expected_lines = 5
    assert len(horiz_bboxes) == expected_lines, f'Expected {expected_lines} lines in botton, found: {len(horiz_bboxes)}'

    # Extract images of lines
    line_imgs = []
    
    y_from = 0
    y_to = min(horiz_bboxes[0][1] + padding, img.shape[1])

    line_imgs.append(img[y_from:y_to])
    
    for i in range(len(horiz_bboxes)-1):
        y_from = max(horiz_bboxes[i][1], 0)
        y_to = min(horiz_bboxes[i+1][1] + padding, img.shape[1])
        line_img = img[y_from:y_to]
        line_imgs.append(line_img)

    return line_imgs


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

    img = resize(img)
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

    nonempty_cords = get_nonempty_bbox(rects_img)

    img = crop_rect(img, *nonempty_cords)
    img_bin = crop_rect(img_bin, *nonempty_cords)
    rects_img = crop_rect(rects_img, *nonempty_cords)
    vertical_lines_img = crop_rect(vertical_lines_img, *nonempty_cords)
    horizontal_lines_img = crop_rect(horizontal_lines_img, *nonempty_cords)
    save_np_img(img, 'tmp_imgs/1_4_img_cropped_nonempty.png')
    save_np_img(vertical_lines_img, 'tmp_imgs/1_5_vertical_lines_img_cropped_nonempty.png')
    save_np_img(horizontal_lines_img, 'tmp_imgs/1_5_horizontal_lines_img_cropped_nonempty.png')

    # Countours only for debug output
    if debug:
        contours_img = get_contours_image(rects_img)
        save_np_img(contours_img, 'tmp_imgs/3_0_contours.png')

    table_loc, bottom_loc = find_table_and_bottom(img, img_bin, rects_img)

    table_img = crop_rect(img, *table_loc)
    bottom_img = crop_rect(img, *bottom_loc)

    save_np_img(table_img, 'tmp_imgs/4_0_table.png')
    save_np_img(bottom_img, 'tmp_imgs/4_1_bottom.png')

    table_img = deskew(table_img)
    save_np_img(table_img, 'tmp_imgs/4_2_table_deskew.png')
    bottom_img = deskew(bottom_img)
    save_np_img(bottom_img, 'tmp_imgs/4_3_bottom_deskew.png')

    table_vertical_lines = crop_rect(vertical_lines_img, *table_loc)
    table_horizontal_lines = crop_rect(horizontal_lines_img, *table_loc)

    save_np_img(table_vertical_lines, 'tmp_imgs/5_0_table_vertical_lines.png')
    save_np_img(table_horizontal_lines, 'tmp_imgs/5_1_table_horizontal_lines.png')

    lines = extract_table_boxes(table_img, table_vertical_lines, table_horizontal_lines)

    os.makedirs('tmp_imgs/lines')
    for i, line in enumerate(lines):
        for key in line.keys():
            save_np_img(line[key], f'tmp_imgs/lines/{i}_{key}.png')

    bottom_horizontal_lines = crop_rect(horizontal_lines_img, *bottom_loc)

    save_np_img(bottom_horizontal_lines, 'tmp_imgs/5_2_bottom_horizontal_lines.png')

    bottom_line_imgs = extract_bottom_boxes(bottom_img, bottom_horizontal_lines)

    os.makedirs('tmp_imgs/bottom')
    for i, line_img in enumerate(bottom_line_imgs):
        save_np_img(line_img, f'tmp_imgs/bottom/line_{i}.png')

    result = {
        'table_lines': lines,
        'bottom_lines': bottom_line_imgs,
    }
    return result
