
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

lexi = cv2.imread('is300.jpg')

# Convert the image to HSV
imgHSV2 = cv2.cvtColor(lexi, cv2.COLOR_BGR2HSV)
imgHue2 = imgHSV2[:, :, 0]
imgSat2 = imgHSV2[:, :, 1]
imgVal2 = imgHSV2[:, :, 2]

# Threshold the hue channel
binret, Bimg = cv2.threshold(imgHue2, 70, 180, cv2.THRESH_BINARY)

# Define a structuring element for erosion
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Erosion operation
eroded_image = cv2.erode(Bimg, disc, iterations=1)

# Edge detection using absdiff
edgydetection = cv2.absdiff(Bimg, eroded_image)

# Display results
plt.subplot(221), plt.imshow(cv2.cvtColor(lexi, cv2.COLOR_BGR2RGB)), plt.title("Original")
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(Bimg, cmap='gray'), plt.title("Binary Image")
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(edgydetection, cmap='gray'), plt.title("Edge Detection")
plt.xticks([]), plt.yticks([])
plt.show()