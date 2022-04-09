# __author__ = 'Prakhar MISRA'
# Created 08/21/2017
# Last edit 08/21/2017
'''
#Purpose:
#-----------------
To provide a template to the code and get matching location in the image. TO be used primarily for identifying brick kilns

Reference:
https://pythonprogramming.net/template-matching-python-opencv-tutorial/
http://www.acgeospatial.co.uk/blog/template-matching-eo/

# EDIT History



#Output expected:
#-----------------
# locations of brick kilns


#Terminology used:
#-----------------
#       df_gdpcor - gdp and aq correlation series
#       st_name - state name
'''

# import
import cv2
import numpy as np


#-----------------


img_rgb = cv2.imread('opencv-template-matching-python-tutorial.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('opencv-template-for-matching.jpg',0)
w, h = template.shape[::-1]


from newspaper import Article




