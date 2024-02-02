import cv2 
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

img = cv2.imread('leaves.jpg')

cv2.imshow("image Leaves ", img)

imgHSV= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imgHue = imgHSV[:, :, 0]
imgSat = imgHSV[:, :, 1]
imgVal= imgHSV[:, :, 2]

fig, axes = plt.subplots(2, 5, figsize=(20, 20))

# Original Image
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original Image')

# HSV Image
axes[0, 1].imshow(imgHSV, cmap='hsv')
axes[0, 1].set_title('HSV Image')

# Hue Channel
axes[0, 2].imshow(imgHue, cmap='gray')
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

ret, mask = cv2.threshold(imgSat, 50, 180,cv2.THRESH_BINARY)
plt.imshow( mask)

disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
print(disc)
fig = plt.figure();
plt.subplot(111), plt.imshow(disc), plt.title("Disc structuring element")
plt.xticks([]), plt.yticks([])

eroded = cv2.erode(mask, disc)
plt.imshow(eroded), plt.title("Eroded Image")
plt.xticks([]), plt.yticks([])

dilated = cv2.dilate(mask, disc)
plt.imshow(dilated), plt.title("Dilated Image")
plt.xticks([]), plt.yticks([])

opening_manual = cv2.dilate(eroded, disc)
plt.imshow(opening_manual), plt.title("Image with opening mask")
plt.xticks([]), plt.yticks([])

closing_manual = cv2.erode(dilated, disc)
plt.imshow(closing_manual), plt.title("Image with closing mask")
plt.xticks([]), plt.yticks([])

opening_auto = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disc)
plt.imshow(opening_auto), plt.title("Image with opening mask")
plt.xticks([]), plt.yticks([])

closing_auto = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disc)
plt.imshow(closing_auto), plt.title("Image with closing mask")
plt.xticks([]), plt.yticks([])

#plot everything
plt.subplot(431), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Original")
plt.xticks([]), plt.yticks([])
plt.subplot(432), plt.imshow(mask, cmap = plt.cm.gray), plt.title("binary mask")
plt.xticks([]), plt.yticks([])
plt.subplot(433), plt.imshow(eroded, cmap = plt.cm.gray), plt.title("Eroded image")
plt.xticks([]), plt.yticks([])
plt.subplot(434), plt.imshow(dilated, cmap = plt.cm.gray), plt.title("Dilated image")
plt.xticks([]), plt.yticks([])
plt.subplot(435), plt.imshow(opening_manual, cmap = plt.cm.gray), plt.title("Open Masked")
plt.xticks([]), plt.yticks([])
plt.subplot(436), plt.imshow(closing_manual, cmap = plt.cm.gray), plt.title("Close Masked")
plt.xticks([]), plt.yticks([])
plt.show()

mask.dtype = 'uint8'
imgSegment = img
imgSegment[:,:,0] = img[:,:,0]*mask
imgSegment[:,:,1] = img[:,:,1]*mask
imgSegment[:,:,2] = img[:,:,2]*mask

plt.imshow(imgSegment)

####PART 2 CODE
lexi = cv2.imread('is300.jpg')
plt.imshow(cv2.cvtColor(lexi, cv2.COLOR_BGR2RGB))

imgHSV2= cv2.cvtColor(lexi, cv2.COLOR_BGR2HSV)
imgHue2= imgHSV2[:, :, 0]
imgSat2 = imgHSV2[:, :, 1]
imgVal2= imgHSV2[:, :, 2]

plt.subplot(221), plt.imshow(cv2.cvtColor(lexi, cv2.COLOR_BGR2RGB)), plt.title("Original")
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(imgHue2), plt.title("Hue image of is300")
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(imgSat2), plt.title("Saturation image of is300")
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(imgVal2), plt.title("Intensity Value image of is300")
plt.xticks([]), plt.yticks([])
plt.show()

#Convert the HSV images to gray 
plt.subplot(221), plt.imshow(imgHue2, cmap = plt.cm.gray), plt.title("Hue image of is300")
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(imgSat2, cmap = plt.cm.gray), plt.title("Saturation image of is300")
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(imgVal2, cmap = plt.cm.gray), plt.title("Intensity Value image of is300")
plt.xticks([]), plt.yticks([])
plt.show()


binret, Bimg = cv2.threshold(imgSat2, 90, 170,cv2.THRESH_BINARY)
disc2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

plt.imshow( Bimg, cmap = plt.cm.gray)

# adding the SE for our erosion, and erode for n number of iterations 
eroded_image = cv2.erode(Bimg, disc2, iterations=9)

#dilating the eroded image to perform a opening mask
dilated_lexi = cv2.dilate(eroded_image, disc2)


# show the binary image to delete the open masked image from the original binary img created
edgydetection = cv2.absdiff(Bimg, dilated_lexi) # deleting the erroded binary from the original

plt.subplot(221), plt.imshow(cv2.cvtColor(lexi, cv2.COLOR_BGR2RGB)), plt.title("Original")
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(dilated_lexi, cmap = plt.cm.gray), plt.title("Image with opening mask")
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(edgydetection, cmap = plt.cm.gray), plt.title("edge detection on morphed image")
plt.xticks([]), plt.yticks([])
plt.show()

#waits for user to press a key
cv2.waitKey(0)
cv2.destroyAllWindows()


