'''
Author: wangxin
Date: 2021-05-25 10:29:58
LastEditTime: 2021-05-28 17:56:20
LastEditors: Please set LastEditors
Description: In User Settings Edit
'''


import numpy as np
import cv2
import matplotlib.pyplot as plt
from detection import get_candidate_area_cnts, get_answer_area_cnts, get_answer_card_cnts, get_sub_answer_card_cnts
from utils import get_init_process_img, capture_img, ocr_single_line_img
from cnocr import CnOcr


def demo(origin_image_path):
    # 获取答题卡区域
    image = get_init_process_img(origin_image_path)
    answer_cnts = get_answer_card_cnts(image)
    answer_card_images_path = []
    if len(answer_cnts) > 0:
        len_answer_cnts = 0
        for c in answer_cnts:
            len_answer_cnts = len_answer_cnts + 1
            answer_card_image_path = 'out/answer_card_' + str(len_answer_cnts) + '.jpg'
            answer_card_images_path.append(answer_card_image_path)
            capture_img(origin_image_path, answer_card_image_path, c)
    print(answer_card_images_path)
    
    # 切分答题卡子模块
    sub_answer_card_images_path = []
    sub_answer_cnt_szie = 0
    for answer_card_image in answer_card_images_path:
        sub_answer_cnts = get_sub_answer_card_cnts(answer_card_image)
        if len(sub_answer_cnts) > 1:
            sub_answer_cnts = sub_answer_cnts[1:len(sub_answer_cnts)]
        
        if len(sub_answer_cnts) > 0:
            for c in sub_answer_cnts:
                sub_answer_card_image_path = 'out/sub_answer_card_' + str(sub_answer_cnt_szie) + '.jpg'
                sub_answer_card_images_path.append(sub_answer_card_image_path)
                capture_img(answer_card_image, sub_answer_card_image_path, c)
                sub_answer_cnt_szie = sub_answer_cnt_szie + 1

    # 获取每个大标题的索引
    ocr = CnOcr()
    title_index = []
    title_num = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二', '十三']
    for img in sub_answer_card_images_path:
        res = ocr_single_line_img(img, ocr)
        print(res)
        if len(res) > 0 and res[0] in title_num:
            title_index.append(sub_answer_card_images_path.index(img))
    print(title_index)
    
    # 识别选择题的序号
    # image = cv2.imread('pic/ti.png')
    # print(image.shape)
    # res = ocr.ocr_for_single_lines([image])
    # print(res)
    # res = ocr.ocr_for_single_line(image)
    # print(res)
    
    image_path = 'out/sub_answer_card_1.jpg'
    image = cv2.imread(image_path)
    # 提取候选框
    candidate_area_cnts = get_candidate_area_cnts(image_path)
    black_background = np.ones(image.shape[:2], np.uint8) * 0
    
    # 显示候选框
    cv2.drawContours(black_background, candidate_area_cnts, -1, (255, 255, 255), 2)
    plt.figure(figsize=(10, 5))
    plt.imshow(black_background)
    plt.axis('off')
    plt.savefig('out/cnt_candidate.png')
    plt.show()
    
    # 提取答案框
    answer_area_cnts = get_answer_area_cnts(image_path)
    black_background = np.ones(image.shape[:2], np.uint8) * 0
    
    # 显示答案框
    cv2.drawContours(black_background, answer_area_cnts, -1, (255, 255, 255), 2)
    plt.figure(figsize=(10, 5))
    plt.imshow(black_background)
    plt.axis('off')
    plt.savefig('out/cnt_answer.png')
    plt.show()
    

if __name__ == '__main__':
    demo('pic/answer_card.jpg')