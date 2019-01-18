import cv2
def bi_demo(image):#高斯双边滤波
    if image == None:
        return
    print("he")
    dst = cv2.bilateralFilter(src=image, d=0, sigmaColor=100, sigmaSpace=15)
    cv2.namedWindow('bi_demo', 0)
    cv2.resizeWindow('bi_demo', 300, 400)
    cv2.imshow("bi_demo", dst)
    cv2.waitKey(0)

'''
    其中各参数所表达的意义：
    src：原图像；
    d：像素的邻域直径，可有sigmaColor和sigmaSpace计算可得；
    sigmaColor：颜色空间的标准方差，一般尽可能大；
    sigmaSpace：坐标空间的标准方差(像素单位)，一般尽可能小。'''

src = cv2.imread("images/1.png")
bi_demo(src)

# cv2.namedWindow('src', 0)
# cv2.resizeWindow('src', 300, 400)
# cv2.imshow('src',src)
# cv2.waitKey(0)
