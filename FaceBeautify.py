import cv2
filepath = "images/1.png"

#第一步：读取图像
img = cv2.imread(filepath) # 读取图片
rows,cols,channel=img.shape
# for i in range(rows):
#     for j in range(cols):
#         print(img[i][j][0],img[i][j][1],img[i][j][2])
print(rows,cols,channel)

#第四步：使用cvtColor()函数将彩色图转变成灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# OpenCV人脸识别分类器 
classifier = cv2.CascadeClassifier( "haarcascade_frontalface_default.xml" )
color = (0, 255, 0) # 定义绘制颜色 
# 调用识别人脸 
faceRects = classifier.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(faceRects): # 大于0则检测到人脸 
    for faceRect in faceRects: # 单独框出每一张人脸 
        x, y, w, h = faceRect 
        # 框出人脸 
        cv2.rectangle(img, (x, y), (x + h, y + w), color, 2) 
        # # 左眼
        # cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
        # #右眼
        # cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
        # #嘴巴
        # cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4), (x + 5 * w // 8, y + 7 * h // 8), color)
# cv2.imshow("image", img) # 显示图像

#第五步：显示灰度图
# cv2.imshow("image", gray) # 显示图像
# c = cv2.waitKey(10)
#
# #防止窗口关闭
# cv2.waitKey(0)
# cv2.destroyAllWindows()