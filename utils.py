import numpy as np
import cv2
import matplotlib.pyplot as plt
from settings import TITLE_TOP_LEFT_CORNER_WIDTH, TITLE_TOP_LEFT_CORNER_HEIGTH
from imutils import auto_canny


def order_points(pts):
    """4边形4点排序函数

    Args:
        pts ([type]): 4边形任意顺序的4个顶点

    Returns:
        [type]: 按照一定顺序的4个顶点
    """
    
    rect = np.zeros((4, 2), dtype="float32")  # 按照左上、右上、右下、左下顺序初始化坐标

    s = pts.sum(axis=1)  # 计算点xy的和
    rect[0] = pts[np.argmin(s)]  # 左上角的点的和最小
    rect[2] = pts[np.argmax(s)]  # 右下角的点的和最大

    diff = np.diff(pts, axis=1)  # 计算点xy之间的差
    rect[1] = pts[np.argmin(diff)]  # 右上角的差最小
    rect[3] = pts[np.argmax(diff)]  # 左下角的差最小
    return rect  # 返回4个顶点的顺序


def four_point_transform(image, pts):
    """4点变换

    Args:
        image ([type]): 原始图像
        pts ([type]): 4个顶点

    Returns:
        [type]: 变换后的图像
    """
    
    rect = order_points(pts)  # 获得一致的顺序的点并分别解包他们
    (tl, tr, br, bl) = rect

    # 计算新图像的宽度(x)
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))  # 右下和左下之间距离
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))  # 右上和左上之间距离
    maxWidth = max(int(widthA), int(widthB))  # 取大者

    # 计算新图像的高度(y)
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))  # 右上和右下之间距离
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))  # 左上和左下之间距离
    maxHeight = max(int(heightA), int(heightB))

    # 有了新图像的尺寸, 构造透视变换后的顶点集合
    dst = np.array(
        [
            [0, 0],  # -------------------------左上
            [maxWidth - 1, 0],  # --------------右上
            [maxWidth - 1, maxHeight - 1],  # --右下
            [0, maxHeight - 1]
        ],  # ------------左下
        dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)  # 计算透视变换矩阵
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))  # 执行透视变换

    return warped  # 返回透视变换后的图像


def sort_contours(cnts, method="left-to-right"):
    """轮廓排序

    Args:
        cnts ([type]): 轮廓
        method (str, optional): 排序方式. Defaults to "left-to-right".

    Returns:
        [type]: 排序好的轮廓
    """
    
    if cnts is None or len(cnts) == 0:
        return [], []
    
    # 初始化逆序标志和排序索引
    reverse = False
    i = 0

    # 是否需逆序处理
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # 是否需要按照y坐标函数
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 构造包围框列表，并从上到下对它们进行排序
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(
        zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    # 返回已排序的轮廓线和边框列表
    return cnts, boundingBoxes


def get_init_process_img(img_path):
    """
    对图片进行初始化处理，包括灰度，高斯模糊，腐蚀，膨胀和边缘检测等
    :param roi_img: ndarray
    :return: ndarray
    """
    image = cv2.imread(img_path)
    # 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 腐蚀erode与膨胀dilate
    # kernel = np.ones((3, 3), np.uint8)
    # blurred = cv2.erode(blurred, kernel, iterations=1) # 腐蚀
    # blurred = cv2.dilate(blurred, kernel, iterations=2) # 膨胀
    # blurred = cv2.erode(blurred, kernel, iterations=1) # 腐蚀
    # blurred = cv2.dilate(blurred, kernel, iterations=2) # 膨胀

    # 边缘检测
    # edged = cv2.Canny(blurred, 75, 200)
    edged = auto_canny(blurred)
    return edged


def capture_img(origin_image_path, target_image_path, contour):
    """根据轮廓截取图片

    Args:
        origin_image_path ([type]): 原始图片路径
        target_image_path ([type]): 目标图片路径
        contour ([type]): 截取轮廓

    Returns:
        [type]: [description]
    """
    # 根据轮廓或者坐标
    x, y, w, h = cv2.boundingRect(contour)
    # 截图
    image = cv2.imread(origin_image_path)
    cv2.imwrite(target_image_path, image[y:y + h, x:x + w])


def save_img_by_cnts(save_image_path, image_size, cnts):
    """通过提取的轮廓绘制图片并保存

    Args:
        save_image_path ([type]): 图片存储路径
        image ([type]): 绘制的图片尺寸, 长与宽
        cnts ([type]): 轮廓列表
    """
    black_background = np.ones(image_size, np.uint8) * 0
    cv2.drawContours(black_background, cnts, -1, (255, 255, 255), 2)
    plt.figure(figsize=(10, 5))
    plt.imshow(black_background)
    plt.axis('off')
    plt.savefig(save_image_path)


def ocr_single_line_img(image_path, ocr):
    """ocr识别图片

    Args:
        origin_image_path ([type]): 原始图片路径
        ocr ([type]): ocr

    Returns:
        [type]: [description]
    """

    image = cv2.imread(image_path)
    res = ocr.ocr_for_single_line(image[0:TITLE_TOP_LEFT_CORNER_WIDTH, 0:TITLE_TOP_LEFT_CORNER_HEIGTH])
    if len(res) > 0 and res[0] == '-':
        res[0] = '一'
    return res

