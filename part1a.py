# ========================================
# ENPM673 Spring 2021: Perception for Autonomous Robotics
# Tag detection using FFT
#
# Author: Siddharth Telang(stelang@umd.edu)
# ========================================
# Run as 'python3 part1a.py'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

#create a circular high pass filter
def hpf(d,imgShape):
    hpfilter = np.ones(imgShape[:2])
    x, y = imgShape[:2]
    for i in range(y):
        for j in range(x):
            if ( ((j-int(x/2))**2 + (i-int(y/2))**2)**0.5 < d):
                hpfilter[j,i] = 0
    return hpfilter

# read image and convert to grayscale
image = cv2.imread('frame.jpg')
image = cv2.resize(image, (800,600))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# calculate FFT, shift it's center and get the magnitude spectrum
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))


plt.subplot(121),plt.imshow(gray, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

# multiply fft with LPF mask
hpf_mask = fshift * hpf(30,gray.shape)
# re-shift center and perform IFFT
back_shift = np.fft.ifftshift(hpf_mask)
hpf_final = abs(np.fft.ifft2(back_shift))

plt.figure()
plt.subplot(121), plt.imshow((20*np.log(np.abs(hpf_mask))),"gray")
plt.title('With Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(hpf_final,"gray")
plt.title('Inverse FFT'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imsave('ifft.jpg', hpf_final, cmap="gray")

# read the saved image and start to detect the corners of AR Tag
image = cv2.imread('ifft.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 50, 255, 0)

# find the contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy_list, contour_list = [],[]
for contour,i in zip(contours,hierarchy[0]):
    if cv2.contourArea(contour) > 500:
        hierarchy_list.append(i)
        contour_list.append(contour)

# once the AR tag is detected, do a approximation, get the corners and draw the contour
im = copy.copy(image)
approx = cv2.approxPolyDP(contour_list[0], 0.03 * cv2.arcLength(contour_list[0], True), True)
cv2.drawContours(im, [approx], 0, (0, 0, 255),2)
cv2.imshow('Detected AR Tag', im)

cv2.waitKey(0)
cv2.destroyAllWindows()