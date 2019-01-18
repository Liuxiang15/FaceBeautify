# import cv2
#
# # c_weight_table = []
# # s_weight_table = []
# # radius = 10
#
# def bi_demo(image):#高斯双边滤波
#     # if image == None:
#     #     return
#     print("he")
#     dst = cv2.bilateralFilter(src=image, d=0, sigmaColor=100, sigmaSpace=15)
#     cv2.namedWindow('bi_demo', 0)
#     cv2.resizeWindow('bi_demo', 300, 400)
#     cv2.imshow("bi_demo", dst)
#     cv2.waitKey(0)
#
# '''
#     其中各参数所表达的意义：
#     src：原图像；
#     d：像素的邻域直径，可有sigmaColor和sigmaSpace计算可得；
#     sigmaColor：颜色空间的标准方差，一般尽可能大；
#     sigmaSpace：坐标空间的标准方差(像素单位)，一般尽可能小。'''
#
# src = cv2.imread("images/1.png")
# bi_demo(src)
#
# cv2.namedWindow('src', 0)
# cv2.resizeWindow('src', 300, 400)
# cv2.imshow('src',src)
# cv2.waitKey(0)
import numpy as np
import math
import cv2
from copy import deepcopy

# filepath = "images/1.png"
# #第一步：读取图像
# src = cv2.imread(filepath) # 读取图片
# rows,cols,channel=src.shape

def myBialteralFilter(src, dst, N, sigmaColor,sigmaSpace):
    print("进入myBialteralFilter")
    size = 2*N + 1
    #分别计算空间权值和相似度权值
    rows, cols, channels = src.shape
    colorArray = getColorArray(size,channels, sigmaColor)
    print("colorArray是————————————————————————：")
    print(colorArray)
    spaceArray = getSpaceArray(size,channels, sigmaSpace)
    print("spaceArray是————————————————————————：")
    print(spaceArray)
    #滤波
    dst = bilateralFilter(dst, N, colorArray, spaceArray)
    return dst

def getColorArray(size, channels, sigmaColor):
    print("进入getColorArray")
    total_num = 255*channels
    colorArray = [0]*(total_num+2)  #为什么不是+1
    for i in range(total_num+1):
        colorArray[i] = math.exp(-i*i / (2*sigmaColor*sigmaColor))
        colorArray[total_num+1] += colorArray[i]
    return colorArray

def getSpaceArray(size,channels, sigmaSpace):
    print("进入getSpaceArray")
    spaces = [[0.0 for col in range(size+1)] for row in range(size+1)]
    spaceArray = np.array(spaces)

    center_i = center_j = size / 2
    for i in range(0, size-1):
        for j in range(0,size-1):
            spaceArray[i][j] = math.exp( -((i-center_i)*(i-center_i)+(j-center_j)*(j-center_j)) /(2*sigmaSpace*sigmaSpace))
            spaceArray[size][0] += spaceArray[i][j] #最后一行的第一个数据放总值
    return spaceArray


def bilateralFilter(src, N, colorArray, spaceArray):
    print("进入bilateralFilter")
    size = 2*N+1
    srcCopy = deepcopy(src)
    rows, cols, channel = src.shape
    print(rows, cols, channel)
    for i in range(rows):
        print("外层循环"+str(i))
        for j in range(cols):
            if i>N-1 and j>N-1 and i<rows-N and j<cols-N:
                spaceColorSum = 0.0
                sum = [0.0, 0.0, 0.0]
                for k in range(size):
                    for l in range(size):
                        x = i-k+N
                        y = j-l+N
                        values = abs(int(src[i][j][0])+int(src[i][j][1])+int(src[i][j][2])-int(src[x][y][0])-int(src[x][y][1])-int(src[x][y][2]))
                        spaceColorSum += colorArray[values]*spaceArray[k][l]
                # print("第93行输出spaceColorSum"+str(spaceColorSum))
                for k in range(size):
                    for l in range(size):
                        x = i-k+N
                        y = j-l+N
                        values = abs(int(src[i][j][0])+int(src[i][j][1])+int(src[i][j][2])-int(src[x][y][0])-int(src[x][y][1])-int(src[x][y][2]))
                        for c in range(3):
                            # print(src[x][y][c])
                            # print(colorArray[values])
                            # print(spaceArray[k][l])
                            #
                            # print("-----------------------------------------------")
                            if spaceColorSum != spaceColorSum:
                                # 先这样特殊处理，之后修改
                                sum[c] = 0

                            else:
                                # print("spaceColorSum是："+str(spaceColorSum))
                                if spaceColorSum != 0:
                                    sum[c] += int((src[x][y][c] * colorArray[values] * spaceArray[k][l]) / spaceColorSum)
                for c in range(3):
                    srcCopy[i][j][c] = sum[c]
    src = deepcopy(srcCopy)
    return src
def whiteFace(src, alpha, beta):
    # 参数2为对比度，参数3为亮度
    rows, cols, channel = src.shape
    for y in range(rows):
        for x in range(cols):
            for c in range(3):
                newValue =  (alpha*src[y][x][c]+beta)% 255
                # print("旧值"+str(src[y][x][c])+"新值"+str(newValue))
                if newValue < src[y][x][c]:
                    src[y][x][c] = 255
                else:
                    src[y][x][c] = newValue

    cv2.imshow("whiteFace", src)
    return src

def getGuassionArray(size, sigma):
    spaces = [[0.0 for col in range(size)] for row in range(size)]
    GuassionArray = np.array(spaces)
    center = size/2
    sum = 0.0
    for i in range(size):
        icenterSquare = ((i-center)*(i-center)
        for j in range(size):

            GuassionArray[i][j] = math.exp(-(icenterSquare+(j-center)*(j-center))/(2*sigma*sigma))
            sum += GuassionArray[i][j]

    for i in range(size):
        for j in range(size):
            GuassionArray[i][j] /= sum
    return  GuassionArray

def myGaussianFilter(src, size,sigma):
    tmp = deepcopy(src)
    GuassionArray = getGuassionArray(size,sigma)
    rows, cols, channel = src.shape
    for i in range(rows):
        for j in range(cols):
            if i-1>0 and i+1<rows and j-1>0 and j+1<cols:
                tmp[i][j][0] = 0
                tmp[i][j][1] = 0
                tmp[i][j][2] = 0
                for x in range(3):
                    for y in range(3):
                        tmp[i][j][0] += GuassionArray[x][y] * src[i + 1 - x][j + 1 - y][0]
                        tmp[i][j][1] += GuassionArray[x][y] * src[i + 1 - x][j + 1 - y][1]
                        tmp[i][j][2] += GuassionArray[x][y] * src[i + 1 - x][j + 1 - y][2]
    return tmp

def main():
    print("进入main")
    filepath = "images/4.jpg"
    src = cv2.imread(filepath)  # 读取图片
    cv2.imshow('srcPic', src)

    dst = deepcopy(src)
    whiteDst = whiteFace(dst, 1.1, 30)
    # GuassionArray = getGuassionArray(3,1.5)
    # print(GuassionArray)
    # dst = myBialteralFilter(src,dst, 25,12.5,50)
    myGuassionPic = myGaussianFilter(whiteDst, 5, 0.5)
    cv2.imshow('myGuassionPic_1', myGuassionPic)

    myGuassionPic = myGaussianFilter(whiteDst, 5, 1.1)
    cv2.imshow('myGuassionPic_2', myGuassionPic)

    myGuassionPic = myGaussianFilter(whiteDst, 5, 10)
    cv2.imshow('myGuassionPic_3', myGuassionPic)

    myGuassionPic = myGaussianFilter(whiteDst, 5, 30)
    cv2.imshow('myGuassionPic_4', myGuassionPic)


    # GaussianBlurPic = cv2.GaussianBlur(whiteDst,(9,9),5)
    # cv2.imshow('GaussianBlurPic', GaussianBlurPic)



    # cv2.imshow('dstPic', dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("End main")
    return
# if __name__ == '__main__':
main()

