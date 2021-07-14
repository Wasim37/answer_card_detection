'''
Author: wangxin
Date: 2021-05-25 10:31:01
LastEditTime: 2021-07-01 13:55:35
LastEditors: Please set LastEditors
Description: 检测考号
'''

# coding=utf-8
import cv2
import numpy as np
from PIL import Image
from utils import sort_contours


def get_exam_num_area(image_path):
    """ 获取图片中待检测的考号填充区域

    Args:
        image_path (String): 图片地址

    Returns:
        [type]: [description]
    """
    image = Image.open(image_path)
    image_width = image.width

    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 先腐蚀与膨胀, 高亮化学生填充的考号
    kernel = np.ones((9, 9), np.uint8)
    erode_img = cv2.erode(threshold_img, kernel, iterations=1)
    kernel = np.ones((9, 9), np.uint8)
    dilate_img = cv2.dilate(erode_img, kernel, iterations=1)

    # 学生填充的考号，最左边边缘的x轴坐标
    exam_number_left_x = float("inf")
    # 学生填充的考号，最右边边缘的x轴坐标
    exam_number_right_x = 0
    cnts, _ = cv2.findContours(dilate_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if x > image_width / 2:
            if x < exam_number_left_x:
                exam_number_left_x = x
            if x + w > exam_number_right_x:
                exam_number_right_x = x + w

    # 通过x轴坐标，缩小待检测区域的范围
    threshold_img = threshold_img[:, exam_number_left_x - 15:exam_number_right_x + 15]

    # 再通过检测图片中面积最大的轮廓（考号手写区域, 而不是填充区域）, 进一步缩小范围
    cnts, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    mix_y = None
    num_card_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        (x, y, w, h) = cv2.boundingRect(approx)
        if len(approx) == 4:
            cv2.imwrite('out/num_card.jpg', threshold_img[y:y + h, x:x + w])
            num_card_cnt = c
            mix_y = y + h
            break

    threshold_img = threshold_img[mix_y:, :]
    return threshold_img, num_card_cnt


def get_exam_num_height(img):
    """ 获取考号填充区域, 行中心与行中心的y轴坐标间隔

    Args:
        img ([type]): 图片

    Returns:
        [float]: 行中心与行中心的y轴坐标间隔
    """

    # 膨胀
    kernel = np.ones((5, 5), np.uint8)
    dilate_img = cv2.dilate(img, kernel, iterations=1)

    # 第一行待填充考号的中心的x轴坐标
    first_line_center_y = None
    # 第二行待填充考号的中心的x轴坐标
    second_line_center_y = None
    # 第一行待填充考号的底部边缘的y坐标
    first_line_bottom_y = None
    cnts, _ = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts, _ = sort_contours(cnts, 'top-to-bottom')
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        (x, y, w, h) = cv2.boundingRect(approx)
        center_y = (2 * y + h) / 2

        if h > 10:
            if first_line_center_y is None:
                first_line_center_y = center_y
                first_line_bottom_y = y + h

            if center_y > first_line_bottom_y and second_line_center_y is None:
                second_line_center_y = center_y
                break

    print(type(second_line_center_y))
    return second_line_center_y - first_line_center_y


def detection_exam_num(image_path):
    """ 识别图片中学生填充的考号

    Args:
        image_path (String): 图片地址

    Returns:
        [list]: 识别的考号结果
    """
    # 获取图片中考号填充区域范围
    thresh_img, _ = get_exam_num_area(image_path)

    # 获取考号填充区域, 每2行的中心y轴坐标间隔
    line_y_height = get_exam_num_height(thresh_img)

    # 腐蚀与膨胀
    kernel = np.ones((9, 9), np.uint8)
    erode_img = cv2.erode(thresh_img, kernel, iterations=1)
    kernel = np.ones((9, 9), np.uint8)
    dilate_img = cv2.dilate(erode_img, kernel, iterations=1)

    # 学生填充考号的识别结果
    num_card = []
    cnts, _ = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts, boundingBoxes = sort_contours(cnts, 'left-to-right')
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        (x, y, w, h) = cv2.boundingRect(approx)
        num_card.append(int(y / line_y_height))
    return num_card


if __name__ == '__main__':
    num_card = detection_exam_num('out/sub_answer_card_0.jpg')
    print('num_card: ', num_card)
