'''
Author: your name
Date: 2021-06-02 17:55:10
LastEditTime: 2021-06-03 15:40:43
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \teaching-answer-card-tool\test.py
'''
import cv2
from cnocr import CnOcr
 

if __name__ == '__main__':
    img = cv2.imread('out/sub_answer_card_1.jpg')
    ocr = CnOcr()
    res = ocr.ocr_for_single_line(img)
    print(res)
