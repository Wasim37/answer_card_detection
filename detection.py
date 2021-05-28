'''
Author: wangxin
Date: 2021-05-25 10:31:01
LastEditTime: 2021-05-28 17:52:17
LastEditors: Please set LastEditors
Description: In User Settings Edit
'''

# coding=utf-8
import cv2
import numpy as np
from utils import sort_contours


def get_candidate_area_cnts(img_path):
    """[summary] 读取图片中的选择题的候选项轮廓

    Args:
        img_path ([type]): 图片地址

    Returns:
        [type]: 候选项轮廓
    """
    
    # 1、图片预处理：灰度、二值化
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_option_answer = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 2、对高亮部分膨胀
    # 因为候选区域由三部分组成（左括号、右括号、大写的英文字母），通过膨胀将三个区域连成一片
    kernel = np.ones((7, 7), np.uint8)
    dilate_choice_answer = cv2.dilate(thresh_option_answer, kernel, iterations=1)

    # 3、提取膨胀后的轮廓
    dilate_option_cnts, _ = cv2.findContours(dilate_choice_answer.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(len(dilate_option_cnts))

    # 4、筛选轮廓中的选择题候选项
    choiceAnswerCnts = []
    for c in dilate_option_cnts:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        # 计算轮廓的边界框
        (x, y, w, h) = cv2.boundingRect(approx) 
        # 计算宽高比
        ar = w / float(h)
        
        # 筛选轮廓为四边形的目前轮廓
        # if len(approx) < 4:  # 不通过边框数进行筛选，会误伤
        if w >= 35 and w <= 60 and ar >= 0.5 and ar <= 2:
            choiceAnswerCnts.append(approx)
    return choiceAnswerCnts


def get_answer_area_cnts(img_path):
    """[summary] 读取图片中的选择题的答案轮廓

    Args:
        img_path ([type]): 图片地址

    Returns:
        [type]: 答案轮廓
    """
    
    # 1、图片预处理：灰度、二值化
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_answer = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 2、提取轮廓
    thresh_answer_cnts, _ = cv2.findContours(thresh_answer.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 3、筛选轮廓中的答案轮廓
    answer_cnts = []
    for c in thresh_answer_cnts:

        # 计算轮廓的边界框
        (x, y, w, h) = cv2.boundingRect(c) 
        # 计算宽高比
        ar = w / float(h)
    
        # 筛选目标轮廓
        if w >= 30 and w <= 40 and ar >= 1 and ar <= 3:
            answer_cnts.append(c)
    return answer_cnts


def get_answer_card_cnts(img):
    """ 获得答题卡的左右答题区域
    # findContours 函数详解：https://blog.csdn.net/laobai1015/article/details/76400725
    # approxPolyDP 多边形近似 https://blog.csdn.net/kakiebu/article/details/79824856
    
    Args:
        img ([type]): 图片
    Returns:
        [type]: 答题卡的左右答题区域轮廓
    """

    # 检测图片中的最外围轮廓
    cnts, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    print("原始图片检测的轮廓总数：", len(cnts))
    if len(cnts) == 0:
        return None

    # 提取的轮廓总数
    contour_size = 0
    # 检测到的左右答题区域轮廓
    answer_cnts = []

    # 将轮廓按大小, 降序排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        # arcLength 计算周长
        peri = cv2.arcLength(c, True)
        # print("轮廓周长：", peri)

        # 之前寻找到的轮廓可能是多边形，现在通过寻找近似轮廓，得到期望的四边形
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print('原始轮廓的边数:', len(c), ', 近似轮廓的边数:', len(approx))

        # 当近似轮廓为4时，代表是需要提取的矩形区域
        if len(approx) == 4:
            contour_size = contour_size + 1
            answer_cnts.append(approx)

        # 只提取答题卡中的最大两个轮廓
        if contour_size == 2:
            break

    answer_cnts = sort_contours(answer_cnts, method="left-to-right")[0]
    return answer_cnts


def get_sub_answer_card_cnts(img_path):
    """ 获得答题卡的子区域
    # findContours 函数详解：https://blog.csdn.net/laobai1015/article/details/76400725
    # approxPolyDP 多边形近似 https://blog.csdn.net/kakiebu/article/details/79824856
    
    Args:
        img ([type]): 图片
    Returns:
        [type]: 答题卡的左右答题区域轮廓
    """
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # warped_answer_image_1 = four_point_transform(gray, answer_contour_1.reshape(4, 2))

    # 二值化
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # 在二值图像中查找轮廓，包括内围、外围轮廓，但是检测到的轮廓不建立等级关系，彼此之间独立
    thresh_cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_SIMPLE)

    cnt_size = 0
    sub_answer_cnts = []
    if len(thresh_cnts) > 0:
        # 将轮廓按大小, 降序排序
        thresh_cnts = sorted(thresh_cnts, key=cv2.contourArea, reverse=True)
        for c in thresh_cnts:
            cnt_size = cnt_size + 1

            # arcLength 计算周长
            peri = cv2.arcLength(c, True)

            # 计算轮廓的边界框
            (x, y, w, h) = cv2.boundingRect(c)
            print((x, y, w, h))

            # 之前寻找到的轮廓可能是多边形，现在通过寻找近似轮廓，得到期望的四边形
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # 只提取近似轮廓为四边形的区域, 且轮廓长度大于指定长度
            if len(approx) == 4 and w > 1300:
                print("轮廓周长：", peri, '宽:', w)
                print('原始轮廓的边数:', len(c), ', 近似轮廓的边数:', len(approx))
                sub_answer_cnts.append(approx)

            # 只处理前20个最大轮廓
            if cnt_size >= 20:
                break

    # 从上到下，将轮廓排序
    print(type(sub_answer_cnts))
    sub_answer_cnts = sort_contours(sub_answer_cnts, method="top-to-bottom")[0]
    print(type(sub_answer_cnts))
    return sub_answer_cnts
