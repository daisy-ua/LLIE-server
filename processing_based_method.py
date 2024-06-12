
import cv2
from utils import *


class ProcessingBasedMethod:

    def run(self, img):
        mean = get_mean_value(img)

        if (mean <= 35):
            return self.__process_dark_image(img)
        elif (mean <= 60):
            return self. __process_bright_image(img)
        else:
            return self.__process_extra_bright_image(img)


    def __process_dark_image(self, low_image, gamma=0.5):
        image = cv2.cvtColor(low_image, cv2.COLOR_BGR2YCrCb)
        image[:, :, 0] = apply_contrast_stretching(image[:, :, 0])
        image[:, :, 0] = apply_gamma_correction(image[:, :, 0], gamma)
        image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        return image


    def __process_bright_image(self, low_image, gamma=0.4):
        image = cv2.cvtColor(low_image, cv2.COLOR_BGR2LAB)
        image[:, :, 0] = apply_contrast_stretching(image[:, :, 0])
        image[:, :, 0] = apply_gamma_correction(image[:, :, 0], gamma)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        return image


    def __process_extra_bright_image(self, low_image, gamma=0.7):
        image = cv2.cvtColor(low_image, cv2.COLOR_BGR2LAB)
        image[:, :, 0] = apply_gamma_correction(image[:, :, 0], gamma)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        return image
