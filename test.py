import numpy as np
import math
image =[[0 for col in range(5)] for row in range(3)]
a = np.array(image)
print(a)

double **getGuassionArray(int size,double sigma) {
	int i, j;
	double sum = 0.0;
	int center = size; //以第一个点的坐标为原点，求出中心点的坐标
 
	double **arr = new double*[size];//建立一个size*size大小的二维数组
	for (i = 0; i < size; ++i)
		arr[i] = new double[size];
	
	for (i = 0; i < size; ++i)
		for (j = 0; j < size; ++j) {
			arr[i][j] = exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (sigma*sigma * 2));
			sum += arr[i][j];
		}
	for (i = 0; i < size; ++i)
		for (j = 0; j < size; ++j)
			arr[i][j] /= sum;
	return arr;
}
 
void myGaussian(const Mat _src, Mat &_dst) {
	if (!_src.data) return;
	double **arr;
	Mat tmp(_src.size(), _src.type());
	for (int i = 0; i < _src.rows; ++i)
		for (int j = 0; j < _src.cols; ++j) {
			//边缘不进行处理
			if ((i - 1) > 0 && (i + 1) < _src.rows && (j - 1) > 0 && (j + 1) < _src.cols) {
				arr = getGuassionArray(3, 1);//自定义得到的权值数组
				tmp.at<Vec3b>(i, j)[0] = 0;
				tmp.at<Vec3b>(i, j)[1] = 0;
				tmp.at<Vec3b>(i, j)[2] = 0;
				for (int x = 0; x < 3; ++x) {
					for (int y = 0; y < 3; ++y) {
							tmp.at<Vec3b>(i, j)[0] += arr[x][y] * _src.at<Vec3b>(i + 1 - x, j + 1 - y)[0];
							tmp.at<Vec3b>(i, j)[1] += arr[x][y] * _src.at<Vec3b>(i + 1 - x, j + 1 - y)[1];
							tmp.at<Vec3b>(i, j)[2] += arr[x][y] * _src.at<Vec3b>(i + 1 - x, j + 1 - y)[2];
					}
				}
			}
		}
	tmp.copyTo(_dst);
}
 
void main() {
	Mat image = imread("路飞.jpg");
	Mat salt_image;
	image.copyTo(salt_image);
 
	Mat _gaussian;
	Mat image1;
	salt(salt_image, 1000);
	myGaussian(salt_image, _gaussian);
	GaussianBlur(salt_image, image1, Size(3, 3), 1);
 
	imshow("原图", image);
	imshow("椒盐化图", salt_image);
	imshow("opencv自带的高斯滤波", image1);
	imshow("自定义高斯滤波", _gaussian);
	waitKey();
}
--------------------- 
作者：林小默 
来源：CSDN 
原文：https://blog.csdn.net/weixin_37720172/article/details/72843238 
版权声明：本文为博主原创文章，转载请附上博文链接！