# ========================================
# ENPM673 Spring 2021: Perception for Autonomous Robotics
# Decode image tag
#
# Author: Siddharth Telang(stelang@umd.edu)
# ========================================
# Run as 'python3 part1b.py'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

def decodeImage(image):
    s = int(image.shape[0]/8)
    # map to store values
    value = {True:1, False:0}

    # -1 means no rotation ; degrees value to be used in cv2.rotate
    degrees = {'BR':-1,'BL':0, 'TL':1, 'TR':2}
    _, thresh1 = cv2.threshold(image, 220, 255, 0)

    # conditions for Bottom Left, Top Left, Top Right, and Bottom Right
    # slice the image as per the step size (s) and calculate median
    bl = np.median(thresh1[5*s:6*s, 2*s:3*s])
    tl = np.median(thresh1[2*s:3*s, 2*s:3*s])
    tr = np.median(thresh1[2*s:3*s, 5*s:6*s])
    br = np.median(thresh1[5*s:6*s, 5*s:6*s])

    # calculate the binary values
    b1 = value.get(np.median(thresh1[4*s:5*s, 3*s:4*s]) == 255.0)
    b2 = value.get(np.median(thresh1[3*s:4*s, 3*s:4*s]) == 255.0)
    b3 = value.get(np.median(thresh1[3*s:4*s, 4*s:5*s]) == 255.0)
    b4 = value.get(np.median(thresh1[4*s:5*s, 4*s:5*s]) == 255.0)

    # orient the binary values as per orientation : left->right: Most significant -> Least significant
    orientation = ''
    val_list = [0,0,0,0]
    if bl == 255.0:
        orientation = 'BL'
        val_list = [b2,b1,b4,b3]
    if tl == 255.0:
        orientation = 'TL'
        val_list = [b3,b2,b1,b4]
    if tr == 255.0:
        orientation = 'TR'
        val_list = [b4,b3,b2,b1]
    if br == 255.0:
        orientation = 'BR'
        val_list = [b1,b4,b3,b2]
    
    orientation_list = [bl,br,tl,tr]
    print(orientation)
    print(val_list)
    print(bl,tl,br,tr)

    # a flag value more than 1 signifies that the current orientation can not be properly determined due to improper frame
    flag = orientation_list.count(255.0)
    print(flag)

    # decode the binary information inside the AR TAG
    value_tag = 8*val_list[0] + 4*val_list[1] + 2*val_list[2] + 1*val_list[3]
    return orientation, value_tag, degrees.get(orientation), flag

def main():
    ref = cv2.imread('AR_REF.png')
    shape = ref.shape[:2]
    # define a max size
    size = 160
    ref = cv2.resize(ref, (size, size))
    # convert to grayscale
    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    orientation, value_tag, degrees, flag = decodeImage(gray)
    ref = cv2.resize(ref, shape)
    ref = cv2.putText(ref, 'Orientation: '+str(orientation), (20,270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
    ref = cv2.putText(ref, 'TAG Value: '+str(value_tag), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)

    cv2.imshow('TAG', ref)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if (__name__ == '__main__'):
    main()