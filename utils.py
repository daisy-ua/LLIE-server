import cv2
import numpy as np


def get_mean_value(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    return mean


def apply_gamma_correction(image, gamma):
    gamma_corrected = np.power(image / 255.0, gamma) * 255
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
    return gamma_corrected


def apply_contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)

    new_min = 0
    new_max = 255

    stretched_image = ((image - min_val) / (max_val - min_val)
                       ) * (new_max - new_min) + new_min

    return np.uint8(stretched_image)


def pixelVal(pix, r1, s1):
    if 0 <= pix <= r1:
        return (s1 / r1) * pix
    else:
        return pix


def apply_piecewise_stretching(image, r1=5, s1=10):
    pixelVal_vec = np.vectorize(pixelVal)
    return pixelVal_vec(image, r1, s1)
