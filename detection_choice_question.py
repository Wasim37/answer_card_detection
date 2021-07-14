'''
Author: wangxin
Date: 2021-05-25 10:31:01
LastEditTime: 2021-07-01 14:11:58
LastEditors: Please set LastEditors
Description: 选择题自动识别与批改
'''

# coding=utf-8
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from settings import ANSWER_CARD_MIN_WIDTH, ANSWER_CARD_SIZE
from utils import sort_contours, save_img_by_cnts


def detection_choice_question(images_path, ocr):
    """ 选择题自动识别与批改

    Args:
        images_path (list): 图片地址列表
    Returns:
        [list]: 每张图片的识别结果
    """

    sub_answer_cnt_szie = 0
    question_answers = []
    for img_path in images_path:
        image = cv2.imread(img_path)
        if not is_choice_question(image):
            continue
        
        # 获取图片中填充的全部答案轮廓
        answer_option_cnts = get_answer_option_cnts(image)
        if len(answer_option_cnts) > 0:
            save_img_by_cnts('out/answer_cnt_' + str(sub_answer_cnt_szie) + '.png', image.shape[:2], answer_option_cnts)

        # 所有被填充的选择项的中心的x坐标
        answer_options_center_x = get_cnt_center_x(answer_option_cnts)
        # 所有未被填充的选择项的中心的x坐标
        choice_options_center_x = get_choice_option_center_x(img_path)
        # 所有选择项的中心的x坐标
        all_options_center_x = answer_options_center_x + choice_options_center_x

        # 获取所有选择项的轮廓及其题序轮廓
        all_choice_option_cnts, question_number_cnts = get_choice_option_cnts(image, all_options_center_x)
        if len(all_choice_option_cnts) > 0:
            save_img_by_cnts('out/choice_cnt_' + str(sub_answer_cnt_szie) + '.png', image.shape[:2], all_choice_option_cnts)
            save_img_by_cnts('out/ques_num_' + str(sub_answer_cnt_szie) + '.png', image.shape[:2], question_number_cnts)

        sub_answer_cnt_szie = sub_answer_cnt_szie + 1

        # 选择题自动批改
        if len(all_choice_option_cnts) > 0:
            question_answer_dict = get_choice_question_answer_index(image, all_choice_option_cnts, answer_option_cnts, question_number_cnts, ocr)
            question_answers.append(question_answer_dict)
    return question_answers


def get_answer_option_cnts(img):
    """ 识别图片中的填充的全部答案轮廓

    Args:
        img_path (String): 图片

    Returns:
        [list]: 候选项轮廓
    """
    
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # OTSU二值化（黑底白字）
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 腐蚀
    kernel = np.ones((5, 5), np.uint8)
    erode_img = cv2.erode(thresh_img, kernel, iterations=1)
    # 膨胀
    kernel = np.ones((9, 9), np.uint8)
    dilate_img = cv2.dilate(erode_img, kernel, iterations=1)

    # 提取答案的轮廓
    answer_cnts, _ = cv2.findContours(dilate_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 减少答案轮廓的边数
    answer_option_cnts = []
    for cnt in answer_cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.06 * peri, True)
        answer_option_cnts.append(approx)

    # self.assertTrue(choiceAnswerCnts % 4 == 0, "候选框提取异常, 提取的数量不是4的整数")
    return answer_option_cnts


def get_choice_option_cnts(img, all_options_center_x):
    """识别图片中的所有的选择项轮廓与题序轮廓

    Args:
        img ([type]): [description]
        all_option_center_x ([type]): [description]

    Returns:
        [type]: [description]
    """
    # 灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化（黑底白字）
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 对高亮部分膨胀
    # 因为候选区域由三部分组成（左括号、右括号、大写的英文字母），通过膨胀将三个区域连成一片
    kernel = np.ones((11, 11), np.uint8)
    dilate_img = cv2.dilate(thresh_img, kernel, iterations=1)

    # 提取膨胀后的轮廓
    option_cnts, _ = cv2.findContours(dilate_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 所有候选框的轮廓
    choice_option_cnts = []
    # 每道选择题的题序
    question_number_cnts = []
    for c in option_cnts:
        peri = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = h / float(w)

        # 筛选轮廓为四边形的目前轮廓
        #     if y >= 60 and w >= 20 and w <= 60 and ar >= 1 and ar <= 2 and area > 700:
        if y >= 60 and ar > 0.5 and ar < 2:
            if is_choice_option(x, w, all_options_center_x) and area > 400:
                choice_option_cnts.append(c)
            elif not is_choice_option(x, w, all_options_center_x) and area > 100:
                question_number_cnts.append(c)
    return choice_option_cnts, question_number_cnts


def is_choice_option(x, w, all_option_center_x):
    for center_x in all_option_center_x:
        if center_x > x and center_x < x + w:
            return True
    return False


def get_cnt_center_x(cnts):
    """返回轮廓中心的x轴坐标

    Args:
        cnts (list): 轮廓列表

    Returns:
        [list]: 中心x轴坐标
    """
    center_x = []
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        center_x.append((2 * x + w) / 2)
    return center_x


def get_choice_option_center_x(img):
    """ 识别所有未被填充的选择项的中心的x坐标

    Args:
        img ([type]): [description]

    Returns:
        [type]: [description]
    """
    img = cv2.imread(img)
    ocr_reslut = pytesseract.image_to_data(img, output_type=Output.DICT, lang='chi_sim')

    choice_option_center_x = []
    for i in range(len(ocr_reslut['text'])):
        text_i = ocr_reslut['text'][i]
        (x, y, w, _) = (ocr_reslut['left'][i], ocr_reslut['top'][i], ocr_reslut['width'][i], ocr_reslut['height'][i])
        if y > 60 and ('A' in text_i or 'B' in text_i or 'C' in text_i or 'D' in text_i):
            choice_option_center_x.append((2 * x + w) / 2)
    return choice_option_center_x


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
    # print("原始图片检测的轮廓总数：", len(cnts))
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
        if contour_size == ANSWER_CARD_SIZE:
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

            # 之前寻找到的轮廓可能是多边形，现在通过寻找近似轮廓，得到期望的四边形
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # 只提取近似轮廓为四边形的区域, 且轮廓长度大于指定长度
            # if len(approx) == 4 and w > ANSWER_CARD_MIN_WIDTH:

            # print("轮廓周长：", peri, '宽:', w)
            # print('原始轮廓的边数:', len(c), ', 近似轮廓的边数:', len(approx))
            if w > ANSWER_CARD_MIN_WIDTH:
                sub_answer_cnts.append(approx)

            # 只处理前20个最大轮廓
            if cnt_size >= 20:
                break

    # 从上到下，将轮廓排序
    sub_answer_cnts = sort_contours(sub_answer_cnts, method="top-to-bottom")[0]
    return sub_answer_cnts


def get_question_num_dict(image, question_number_cnts, ocr):
    """获取图片中所有的选择题的题序

    Args:
        image ([type]): 图片
        question_number_cnts ([type]): 图片中的所有的选择题的题序轮廓
        ocr ([type]): ocr识别工具

    Returns:
        [dict]: key: 题序, value: 题序轮廓的坐标
    """
    question_num_dict = {}
    for question_number_cnt in question_number_cnts:
        peri = cv2.arcLength(question_number_cnt, True)
        approx = cv2.approxPolyDP(question_number_cnt, 0.1 * peri, True)
        (x, y, w, h) = cv2.boundingRect(approx)
        
        # ocr识别题型轮廓区域的文本
        text = ocr.ocr_for_single_line(image[y:y + h, x:x + w])
        question_num = ''.join(text)
        question_num = question_num.replace('.', '')
        
        # 文本是否为数字
        if question_num.isdigit():
            (x, y, w, h) = cv2.boundingRect(question_number_cnt)
            question_num_dict[int(question_num)] = (x, y, w, h)
    
    # 按照题序从小到大排序
    question_num_list = sorted(question_num_dict.items(), key=lambda item: item[0])
    return dict(question_num_list)


def get_choice_question_answer_index(image, choice_option_cnts, answer_option_cnts, question_number_cnts, ocr):
    """自动批改, 返回每道试题对应的答案索引. \
       注意：(1)用户可能没有填充答案 (2)选择题的答案数量可能大于1

    Args:
        choice_option_cnts (list): 试题的选择项轮廓
        answer_option_cnts (list): 用户填充的答案轮廓
        question_number_cnts (list): 试题的题序轮廓
    Returns:
        [dict]: key  题序, value 答案索引列表
    """
    
    # 获取所有选择题的题序
    question_num_dict = get_question_num_dict(image, question_number_cnts, ocr)
    
    question_answer_dict = {}
    for key in question_num_dict.keys():
        (num_x, num_y, num_w, num_h) = question_num_dict[key]
        num_center_x = (2 * num_x + num_w) / 2
        num_center_y = (2 * num_y + num_h) / 2

        # 获取同一行中，本题序右侧第一个题序的中心x坐标
        min_num_center_x = float("inf")  # 无穷大
        for question_number_cnt in question_number_cnts:
            (x, y, w, h) = cv2.boundingRect(question_number_cnt)
            right_num_center_x = (2 * x + w) / 2
            if num_center_y > y and num_center_y < y + h and right_num_center_x > num_center_x and right_num_center_x < min_num_center_x:
                min_num_center_x = right_num_center_x
        # print(min_num_center_x)

        # 获取本题的全部答案轮廓的中心x坐标列表
        # 一道选择题题可能有多个答案， 所以answers_center_x为列表
        answers_center_x = []
        for answer_option_cnt in answer_option_cnts:
            (x, y, w, h) = cv2.boundingRect(answer_option_cnt)
            answer_cnt_center_x = (2 * x + w) / 2
            if num_center_y > y and num_center_y < y + h and answer_cnt_center_x > num_center_x and answer_cnt_center_x < min_num_center_x:
                answers_center_x.append(answer_cnt_center_x)
        # print('answers_center_x', answers_center_x)

        # 获取本题的全部选择项轮廓
        question_choice_option_cnts = []
        for choice_option_cnt in choice_option_cnts:
            # print(len(question_choice_option_cnts))
            (x, y, w, h) = cv2.boundingRect(choice_option_cnt)
            choice_option_center_x = (2 * x + w) / 2
            if num_center_y > y and num_center_y < y + h and choice_option_center_x > num_center_x and choice_option_center_x < min_num_center_x:
                question_choice_option_cnts.append(choice_option_cnt)

        question_choice_option_cnts, _ = sort_contours(question_choice_option_cnts, 'left-to-right')
        # print('question_choice_option_cnts', len(question_choice_option_cnts))

        # 答案列表
        answer_indexes = []
        # 答案索引
        answer_index = 0
        for choice_option_cnt in question_choice_option_cnts:
            answer_index = answer_index + 1
            (x, y, w, h) = cv2.boundingRect(choice_option_cnt)
            # print((x, y, w, h), answers_center_x)
            for answer_center_x in answers_center_x:
                if answer_center_x > x and answer_center_x < x + w:
                    answer_indexes.append(answer_index)
                    break
        question_answer_dict[key] = answer_indexes

    # 返回每道试题对应的答案索引
    question_answer_dict = sorted(question_answer_dict.items(), key=lambda item: item[0])
    return dict(question_answer_dict)


def is_choice_question(img):
    """判断当前图片是否属于选择题

    Args:
        image_path ([type]): 图片

    Returns:
        [boolean]: false 不是  true 是
    """
    ocr_result = pytesseract.image_to_data(img, output_type=Output.DICT, lang='chi_sim')
    ocr_text = ocr_result['text']
    return '[A]' in ocr_text or '[B]' in ocr_text or '[C]' in ocr_text or '[D]' in ocr_text
