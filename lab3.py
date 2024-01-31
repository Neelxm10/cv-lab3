import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

##  BIG MICH

#Read in image

#convert RGB to HSV

#Use subplots from matplotlib to show HSV and original photo


#Seperate channels Hue (H), Saturation (S), Value (V)

img = cv.imread('leaves.jpg')

cv.imshow("Leaves", img)

imgHSV= cv.cvtColor(img, cv.COLOR_BGR2HSV)
imgHue = imgHSV[:, :, 0]
imgSat = imgHSV[:, :, 1]
imgVal= imgHSV[:, :, 2]

Fenetre= cv.hconcat([imgHSV, imgHue , imgSat , imgVal])
cv.imshow('HSV Image+ Hue Channel + Saturation Channel'+ 'Value Channel', Fenetre)


