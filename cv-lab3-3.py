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

#convert to binary

path = r'/home/mich/Documents/Michel/Opencv/cv-lab3/leaves.jpg'
img = cv.imread(path)



imgHSV= cv.cvtColor(img, cv.COLOR_BGR2HSV)

imgSat = imgHSV[:, :, 1]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Normal images')


# HSV Image
axes[0, 1].imshow(imgSat, cmap='hsv')
axes[0, 1].set_title('HSV Image')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()

cv.waitKey(0) 
cv.destroyAllWindows()