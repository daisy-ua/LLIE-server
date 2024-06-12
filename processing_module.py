import cv2
from utils import *
from processing_based_method import ProcessingBasedMethod
from retinex_based_method import RetinexBasedMethod

class ProcessingModule:
    
     def __init__(self):
          self.__processing_based_method = ProcessingBasedMethod()
          self.__retinex_based_method = RetinexBasedMethod()


     def run_processing_based_method(self, image):
          try:
               return self.__processing_based_method.run(image)

          except Exception as e:
               print(e)        


     def run_retinex_based_method(self, image):
          try:
               image = self.__resize_image(image)

               print(image.shape[:2])

               return self.__retinex_based_method.run(image)

          except Exception as e:
               print(e) 


     def __resize_image(self, image):
          height, width = image.shape[:2]

          if max(height, width) > 600:
             scale_factor = 600 / max(height, width)
             return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

          else:
               return image




