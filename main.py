'''
Author: wangxin
Date: 2021-05-25 10:29:58
LastEditTime: 2021-07-01 14:12:15
LastEditors: Please set LastEditors
Description: In User Settings Edit
'''

from detection_choice_question import get_answer_card_cnts, get_sub_answer_card_cnts, detection_choice_question
from detection_exam_num import detection_exam_num
from settings import TITLE_NUM
from utils import get_init_process_img, capture_img, ocr_single_line_img
from cnocr import CnOcr


def demo(origin_image_path):
    # 获取答题卡左右区域
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
    print('答题卡左右区域切分结果：', answer_card_images_path)

    # 将答题卡切分为一道道试题
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
    print('试题切分结果：', sub_answer_card_images_path)

    # 获取每个大标题的索引
    ocr = CnOcr()
    title_index = []
    for img in sub_answer_card_images_path:
        res = ocr_single_line_img(img, ocr)
        if len(res) > 0 and res[0] in TITLE_NUM:
            title_index.append(sub_answer_card_images_path.index(img))
    print('每道大题的起始图片索引: ', title_index)
    
    # 学生考号自动识别
    num_card = detection_exam_num(sub_answer_card_images_path[0])
    print('学生考号: ', num_card)

    # 选择题自动识别与批改
    question_answer_dict = detection_choice_question(sub_answer_card_images_path, ocr)
    print('每道选择题答案（key 题序, value: 对应题序的答案列表）：', question_answer_dict)
    

if __name__ == '__main__':
    demo('pic/answer_card1.jpg')
