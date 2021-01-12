import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QListWidgetItem, QPushButton
from flags import *
from matplotlib import pyplot as plt


class MyItem(QListWidgetItem):
    def __init__(self, name=None, parent=None):
        super(MyItem, self).__init__(name, parent=parent)
        self.setIcon(QIcon('icons/color.png'))
        self.setSizeHint(QSize(100, 60))

    def get_params(self):
        protected = [v for v in dir(self) if v.startswith('_') and not v.startswith('__')]
        param = {}
        for v in protected:
            param[v.replace('_', '', 1)] = self.__getattribute__(v)
        return param

    def update_params(self, param):
        for k, v in param.items():
            if '_' + k in dir(self):
                self.__setattr__('_' + k, v)


class ExpTranItem(MyItem):
    def __init__(self, parent=None):
        super(ExpTranItem, self).__init__('指数灰度变换', parent=parent)
        self._param1 = 0  # esp
        self._param2 = 1  # gamma

    def __call__(self, img):
        height, width = img.shape[0], img.shape[1]
        for i in range(height):
            for j in range(width):
                for k in range(3):
                    tmp = img[i, j, k] / 255
                    tmp = int(pow(tmp + self._param1, self._param2) * 255)
                    if 0 <= tmp <= 255:
                        img[i, j, k] = tmp
                    elif tmp > 255:
                        img[i, j, k] = 255
                    else:
                        img[i, j, k] = 0
        return img


class GammaItem(MyItem):
    def __init__(self, parent=None):
        super(GammaItem, self).__init__('伽马校正', parent=parent)
        self._gamma = 1

    def __call__(self, img):
        gamma_table = [np.power(x / 255.0, self._gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)


class FilterItem(MyItem):

    def __init__(self, parent=None):
        super().__init__('平滑处理', parent=parent)
        self._ksize = 3
        self._kind = MEAN_FILTER
        self._sigmax = 0

    def __call__(self, img):
        # 均值滤波
        if self._kind == MEAN_FILTER:
            img = cv2.blur(img, (self._ksize, self._ksize))
        # 高斯滤波
        elif self._kind == GAUSSIAN_FILTER:
            img = cv2.GaussianBlur(img, (self._ksize, self._ksize), self._sigmax)
        # 中值滤波
        elif self._kind == MEDIAN_FILTER:
            img = cv2.medianBlur(img, self._ksize)
        return img


class HisBalanceItem(MyItem):
    def __init__(self, parent=None):
        super(HisBalanceItem, self).__init__('直方图均衡化', parent=parent)
        self._param1 = 256

    def __call__(self, img):
        dmax = 255
        height, width = img.shape[0], img.shape[1]
        tran_func = dict(zip([i for i in range(round(self._param1))], [0 for i in range(round(self._param1))]))
        hist = dict(zip([i for i in range(round(self._param1))], [0 for i in range(round(self._param1))]))

        for i in range(height):
            for j in range(width):
                hist[img[i, j, 0]] += 1

        total = 0
        for i in range(round(self._param1)):
            total += hist[i]
            tran_func[i] = round(dmax * total / height / width)

        for i in range(height):
            for j in range(width):
                for k in range(3):
                    img[i, j, k] = tran_func[img[i, j, k]]
        return img


class ImcompItem(MyItem):
    def __init__(self, parent=None):
        super(ImcompItem, self).__init__('彩色负片', parent=parent)

    def __call__(self, img):
        table = np.array([255 - i for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)


class LaplaceSharpItem(MyItem):
    def __init__(self, parent=None):
        super(LaplaceSharpItem, self).__init__('拉普拉斯锐化', parent=parent)
        self._param1 = 1

    def __call__(self, img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        img = cv2.filter2D(img, -1, kernel=kernel)
        return img


class BorderDetectItem(MyItem):
    def __init__(self, parent=None):
        super(BorderDetectItem, self).__init__('边缘检测', parent=parent)
        self._kind = SCHARR

    def __call__(self, img):
        if self._kind == SCHARR:
            x = cv2.Sobel(img, cv2.CV_16S, 1, 0, -1)
            y = cv2.Sobel(img, cv2.CV_16S, 0, 1, -1)
            img = cv2.addWeighted(cv2.convertScaleAbs(x), 0.5, cv2.convertScaleAbs(y), 0.5, 0)
        elif self._kind == LAPLACIAN:
            laplacian = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
            img = cv2.convertScaleAbs(laplacian)
        elif self._kind == CANNY:
            img = cv2.Canny(img, 50, 150)
        return img


class DFTItem(MyItem):
    def __init__(self, parent=None):
        super(DFTItem, self).__init__('傅里叶变换频谱', parent=parent)

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dftShift = np.fft.fftshift(dft)
        img = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))
        img = img.astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img


class HighPassFilterItem(MyItem):
    def __init__(self, parent=None):
        super(HighPassFilterItem, self).__init__('高通滤波', parent=parent)
        self._d = 10

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        def make_transform_matrix(d):
            transfor_matrix = np.zeros(img.shape)
            center_point = tuple(map(lambda x: (x - 1) / 2, img.shape))
            for i in range(transfor_matrix.shape[0]):
                for j in range(transfor_matrix.shape[1]):
                    def cal_distance(pa, pb):
                        from math import sqrt
                        dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                        return dis

                    dis = cal_distance(center_point, (i, j))
                    if dis <= d:
                        transfor_matrix[i, j] = 0
                    else:
                        transfor_matrix[i, j] = 1
            return transfor_matrix
        d_matrix = make_transform_matrix(self._d)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
        img = new_img.astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img


class LowPassFilterItem(MyItem):
    def __init__(self, parent=None):
        super(LowPassFilterItem, self).__init__('低通滤波', parent=parent)
        self._d = 90

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        def make_transform_matrix(d):
            transfor_matrix = np.zeros(img.shape)
            center_point = tuple(map(lambda x: (x - 1) / 2, img.shape))
            for i in range(transfor_matrix.shape[0]):
                for j in range(transfor_matrix.shape[1]):
                    def cal_distance(pa, pb):
                        from math import sqrt
                        dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                        return dis

                    dis = cal_distance(center_point, (i, j))
                    if dis <= d:
                        transfor_matrix[i, j] = 1
                    else:
                        transfor_matrix[i, j] = 0
            return transfor_matrix
        d_matrix = make_transform_matrix(self._d)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
        img = new_img.astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img


class MakeBlurredItem(MyItem):
    def __init__(self, parent=None):
        super(MakeBlurredItem, self).__init__('加噪声', parent=parent)
        self._eps = 1e-3

    def __call__(self, img):
        def motion_process(image_size, motion_angle):
            import math
            PSF = np.zeros(image_size)
            center_position = (image_size[0] - 1) / 2
            slope_tan = math.tan(motion_angle * math.pi / 180)
            slope_cot = 1 / slope_tan
            if slope_tan <= 1:
                for i in range(15):
                    offset = round(i * slope_tan)
                    PSF[int(center_position + offset), int(center_position - offset)] = 1
                return PSF / PSF.sum()
            else:
                for i in range(15):
                    offset = round(i * slope_cot)
                    PSF[int(center_position - offset), int(center_position + offset)] = 1
                return PSF / PSF.sum()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        PSF = motion_process((img.shape[0], img.shape[1]), 60)
        input_fft = np.fft.fft2(img)
        PSF_fft = np.fft.fft2(PSF) + self._eps
        blurred = np.fft.ifft2(input_fft * PSF_fft)
        blurred = np.abs(np.fft.fftshift(blurred))
        img = blurred.astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img


class ReverseFilterItem(MyItem):
    def __init__(self, parent=None):
        super(ReverseFilterItem, self).__init__('逆滤波复原', parent=parent)
        self._eps = 1e-3

    def __call__(self, img):
        def motion_process(image_size, motion_angle):
            import math
            PSF = np.zeros(image_size)
            center_position = (image_size[0] - 1) / 2

            slope_tan = math.tan(motion_angle * math.pi / 180)
            slope_cot = 1 / slope_tan
            if slope_tan <= 1:
                for i in range(15):
                    offset = round(i * slope_tan)
                    PSF[int(center_position + offset), int(center_position - offset)] = 1
                return PSF / PSF.sum()
            else:
                for i in range(15):
                    offset = round(i * slope_cot)
                    PSF[int(center_position - offset), int(center_position + offset)] = 1
                return PSF / PSF.sum()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        PSF = motion_process((img.shape[0], img.shape[1]), 60)
        input_fft = np.fft.fft2(img)
        PSF_fft = np.fft.fft2(PSF) + self._eps
        result = np.fft.ifft2(input_fft / PSF_fft)
        result = np.abs(np.fft.fftshift(result))
        img = result.astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img


class WienerFilterItem(MyItem):
    def __init__(self, parent=None):
        super(WienerFilterItem, self).__init__('维纳滤波复原', parent=parent)
        self._eps = 1e-3
        self._K = 0.01

    def __call__(self, img):
        def motion_process(image_size, motion_angle):
            import math
            PSF = np.zeros(image_size)
            center_position = (image_size[0] - 1) / 2
            slope_tan = math.tan(motion_angle * math.pi / 180)
            slope_cot = 1 / slope_tan
            if slope_tan <= 1:
                for i in range(15):
                    offset = round(i * slope_tan)
                    PSF[int(center_position + offset), int(center_position - offset)] = 1
                return PSF / PSF.sum()
            else:
                for i in range(15):
                    offset = round(i * slope_cot)
                    PSF[int(center_position - offset), int(center_position + offset)] = 1
                return PSF / PSF.sum()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        PSF = motion_process((img.shape[0], img.shape[1]), 60)
        input_fft = np.fft.fft2(img)
        PSF_fft = np.fft.fft2(PSF) + self._eps
        PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + self._K)
        result = np.fft.ifft2(input_fft * PSF_fft_1)
        result = np.abs(np.fft.fftshift(result))
        img = result.astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img
