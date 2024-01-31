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

path = r'/home/mich/Documents/Michel/Opencv/Labs/cv-lab3/leaves.jpg'
img = cv.imread(path)



imgHSV= cv.cvtColor(img, cv.COLOR_BGR2HSV)
imgHue = imgHSV[:, :, 0]
imgSat = imgHSV[:, :, 1]
imgVal= imgHSV[:, :, 2]


fig, axes = plt.subplots(2, 5, figsize=(10, 10))

# Original Image
axes[0, 0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axes[0, 0].set_title('Original Image')

# HSV Image
axes[0, 1].imshow(cv.cvtColor(imgHSV, cv.COLOR_HSV2RGB))
axes[0, 1].set_title('HSV Image')

# Hue Channel
axes[0, 2].imshow(imgHue, cmap='hsv')
axes[0, 2].set_title('Hue Channel')

# Saturation Channel
axes[0, 3].imshow(imgSat, cmap='gray')
axes[0, 3].set_title('Saturation Channel')

# Value Channel
# Normalize the Value channel to be between 0 and 1
imgVal_normalized = imgVal / 255.0
axes[0, 4].imshow(imgVal_normalized, cmap='gray')
axes[0, 4].set_title('Value Channel')

# Remove axis labels
for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()

cv.waitKey(0) 
cv.destroyAllWindows()