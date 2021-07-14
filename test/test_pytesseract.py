'''
Author: your name
Date: 2021-06-02 17:55:10
LastEditTime: 2021-06-03 15:40:43
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \teaching-answer-card-tool\test.py
'''
import cv2
import pytesseract
from utils import get_init_process_img
from pytesseract import Output
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
 
# https://livezingy.com/pytesseract-image_to_data_locate_text/
# pytesseract.pytesseract.tesseract_cmd = r'D:\ProgramData\Tesseract-OCR\tesseract.exe'


def recoText(im):
    """
    识别字符并返回所识别的字符及它们的坐标
    :param im: 需要识别的图片
    :return data: 字符及它们在图片的位置
    """
    data = {}
    # im = get_init_process_img(im)
    # d = pytesseract.image_to_string(im, output_type=Output.DICT, lang='chi_sim')
    # print(d['text'])
    
    dd = pytesseract.image_to_string(im, output_type=Output.DICT, lang="eng")
    print(dd['text'])
    # print(d['text'])
    # for i in range(len(d['text'])):
    #     print(d['text'][i])
    #     if 0 < len(d['text'][i]):
    #         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #         data[d['text'][i]] = ([d['left'][i], d['top'][i], d['width'][i], d['height'][i]])
 
    #         cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 1)
    #         # 使用cv2.putText不能显示中文，需要使用下面的代码代替
    #         # cv2.putText(im, d['text'][i], (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
 
    #         pilimg = Image.fromarray(im)
    #         draw = ImageDraw.Draw(pilimg)
    #         # 参数1：字体文件路径，参数2：字体大小
    #         # Hiragino Sans GB.ttc 为mac下的简体中文
    #         font = ImageFont.truetype("Hiragino Sans GB.ttc", 15, encoding="utf-8")
    #         # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    #         draw.text((x, y - 10), d['text'][i], (255, 0, 0), font=font)
    #         im = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
 
    # cv2.imshow("recoText", im)
    return data


def recognize_text(image):
    # 边缘保留滤波  去噪
    blur = cv2.pyrMeanShiftFiltering(image, sp=8, sr=60)
    cv2.imshow('dst', blur)
    # 灰度图像
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # 二值化  设置阈值  自适应阈值的话 黄色的4会提取不出来
    ret, binary = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY_INV)
    print(f'二值化设置的阈值：{ret}')
    cv2.imshow('binary', binary)
    # 逻辑运算  让背景为白色  字体为黑  便于识别
    cv2.bitwise_not(binary, binary)
    cv2.imshow('bg_image', binary)
    # 识别
    test_message = Image.fromarray(binary)
    text = pytesseract.image_to_string(test_message)
    print(f'识别结果：{text}')
 
 
if __name__ == '__main__':
    # img = cv2.imread('out/sub_answer_card_3.jpg')
    
    img = cv2.imread('20210623184007.jpg')
    
    # 转灰度
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # 二值化
    
    # height, width, deep = img.shape                 # cropImg是从图片里截取的,只包含一行数字
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # 转灰度图
    # dst = np.zeros((height, width, 1), np.uint8)        
    # for i in range(0, height):                          # 反相 转白底黑字
    #     for j in range(0, width):
    #         grayPixel = gray[i, j]
    #         dst[i, j] = 255 - grayPixel
    # ret, canny = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # 二值化

    
    # img = cv2.imread('out/test.png')
    # cv2.imshow("src", img)
    # data = recoText(img)
    recognize_text(img)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
