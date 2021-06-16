# ========================================
# ENPM673 Spring 2021: Perception for Autonomous Robotics
# Utils file providing various helper functions
#
# Author: Siddharth Telang(stelang@umd.edu)
# ========================================
#

from imutils import paths
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

# function to warp turtle to the main frame
def warpTurtle(img,h,dest):
    row,col = dest.shape[:2]
    H = np.linalg.inv(h)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            mat = np.dot(H,[i,j,1])
            # normalize
            k,l,_ = (mat/mat[2]).astype(int)
            if l < col and k < row:
                dest[int(l),int(k)] = img[j,i]
    return dest

#function to warp the AR tag
def warpARTag(img, h, x, y):
    H = np.linalg.inv(h)
    dest = np.zeros((x,y),dtype =np.uint8)
    for i in range(x):
        for j in range(y):
            mat = np.dot(H,[i,j,1])
            k,l,_ = (mat/mat[2]).astype(int)
            if l < img.shape[1] and k < img.shape[0]:
                      dest[j,i] = img[int(l),int(k)]

    return dest

# function to find the homography matrix
def findHomography(src, dst):
    S = []
    for i in range(len(src)):
        x,y = src[i][0], src[i][1]
        a,b = dst[i][0], dst[i][1]
        S.append([x, y, 1, 0, 0, 0, -a*x, -a*y, -a])
        S.append([0, 0, 0, x, y, 1, -b*x, -b*y, -b])
    A = np.array(S)
    # calculate SVD
    U, S, Vt = np.linalg.svd(A)
    # take the last row of Vt, divide by last element and reshape to 3x3 to obtain homography matrix
    H = (Vt[-1]/Vt[-1,-1]).reshape(3,3)
    return H

def decodeImage(image):
    s = int(image.shape[0]/8)
    value = {True:1, False:0}
    degrees = {'BR':-1,'BL':0, 'TL':1, 'TR':2}
    _, thresh1 = cv2.threshold(image, 220, 255, 0)

    bl = np.median(thresh1[5*s:6*s, 2*s:3*s])
    tl = np.median(thresh1[2*s:3*s, 2*s:3*s])
    tr = np.median(thresh1[2*s:3*s, 5*s:6*s])
    br = np.median(thresh1[5*s:6*s, 5*s:6*s])

    b1 = value.get(np.median(thresh1[4*s:5*s, 3*s:4*s]) == 255.0)
    b2 = value.get(np.median(thresh1[3*s:4*s, 3*s:4*s]) == 255.0)
    b3 = value.get(np.median(thresh1[3*s:4*s, 4*s:5*s]) == 255.0)
    b4 = value.get(np.median(thresh1[4*s:5*s, 4*s:5*s]) == 255.0)

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
    #print(orientation)
    #print(val_list)
    print(bl,tl,br,tr)
    flag = orientation_list.count(255.0)
    #print(flag)
    value_tag = 8*val_list[0] + 4*val_list[1] + 2*val_list[2] + 1*val_list[3]
    return orientation, value_tag, degrees.get(orientation), flag

def calcProjectionMatrix(H, K):
    K = K.T
    # take the first two column vectors of homography matrix to calculate lamba
    h1 = H[:,0]
    h2 = H[:,1]

    k_inv = np.linalg.inv(K)
    # calculate the value of lamba - scale factor
    Lambda = 2 / (np.linalg.norm(np.matmul(k_inv,h1))+ np.linalg.norm(np.matmul(k_inv,h2)))
    # calculate b tilda
    b_tilda = Lambda * (np.matmul(k_inv, H))
    # calculate the determinant to assign sign - if negative, make it positive
    det = np.linalg.det(b_tilda)
    b = b_tilda
    if det < 0 : b = -1*b_tilda

    # next, calculate the rotation vectors r1, r2, r3
    r1 = b[:, 0]
    r2 = b[:, 1]
    r3 = np.cross(r1, r2)

    # get the translation vector
    tr = b[:, 2]

    # stack the rotation and translation vectors to rotation matrix
    Rot = np.stack((r1,r2,r3,tr), axis=1)

    # calculate the final Projection matrix
    Pr = np.matmul(K, Rot)

    return Pr, Rot, tr


def drawCube(P, size, image):

    # define points and multiply by Projection matrix
    x1,y1,z1 = np.matmul(P,[0,0,0,1])
    x2,y2,z2 = np.matmul(P,[0,size,0,1])
    x3,y3,z3 = np.matmul(P,[size,0,0,1])
    x4,y4,z4 = np.matmul(P,[size,size,0,1])
    x5,y5,z5 = np.matmul(P,[0,0,-size,1])
    x6,y6,z6 = np.matmul(P,[0,size,-size,1])
    x7,y7,z7 = np.matmul(P,[size,0,-size,1])
    x8,y8,z8 = np.matmul(P,[size,size,-size,1])
    
    # Scale by z axis and join lines of cube
    cv2.line(image,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (7,122,36), 2)
    cv2.line(image,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (7,122,36), 2)
    cv2.line(image,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (7,122,36), 2)
    cv2.line(image,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (7,122,36), 2)

    cv2.line(image,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (53,5,245), 2)
    cv2.line(image,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (53,5,245), 2)
    cv2.line(image,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (53,5,245), 2)
    cv2.line(image,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (53,5,245), 2)
    
    cv2.line(image,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (102, 13, 49), 2)
    cv2.line(image,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (102, 13, 49), 2)
    cv2.line(image,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (102, 13, 49), 2)
    cv2.line(image,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (102, 13, 49), 2)

    return image
