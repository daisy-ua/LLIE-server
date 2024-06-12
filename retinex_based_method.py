import cv2
import os
from utils import *

from retinex_model import load_retinex_model

class RetinexBasedMethod:

    def __init__(self):
        self.__tmp_dir = "tmp_files"
        self.__tmp_filename = "low_space.png"
          
        self.__retinex_model = load_retinex_model()

        self.__tmp_path = os.path.join(self.__tmp_dir, self.__tmp_filename)
        self.__retinex_dir = "results"
        self.__ckpt_dir = "data/ckpt_dir"

     
    def run(self, img):
        mean = get_mean_value(img)
        preprocessed = []

        if (mean <= 35):
            preprocessed = self.__process_dark_image(img)
        elif (mean <= 60):
            preprocessed = self.__process_bright_image(img, gamma=1.2)
        else:
            preprocessed = self.__process_bright_image(img, gamma=1.5)

        return self.__process_retinex(preprocessed, self.__retinex_model)


    def __process_retinex(self, img, retinex_model):
        cv2.imwrite(self.__tmp_path, img)

        if not os.path.exists(self.__retinex_dir):
            os.makedirs(self.__retinex_dir)

        retinex_model.predict([self.__tmp_path], self.__retinex_dir, self.__ckpt_dir)

        res = cv2.imread(os.path.join(
            self.__retinex_dir, self.__tmp_filename.split('.')[0] + '.jpg'))

        res = res[:, int(res.shape[1]/2):]

        return res


    def __process_dark_image(self, low_image, gamma=0.8):
        image = cv2.cvtColor(low_image, cv2.COLOR_BGR2YCrCb)
        image[:, :, 0] = apply_contrast_stretching(image[:, :, 0])
        image[:, :, 0] = apply_gamma_correction(image[:, :, 0], gamma)
        image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        return image


    def __process_bright_image(self, low_image, gamma=1.2):
        image = cv2.cvtColor(low_image, cv2.COLOR_BGR2HSV)
        image[:, :, 2] = apply_piecewise_stretching(image[:, :, 2])
        image[:, :, 2] = apply_gamma_correction(image[:, :, 2], gamma)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

