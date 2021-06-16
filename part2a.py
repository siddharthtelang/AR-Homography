# ========================================
# ENPM673 Spring 2021: Perception for Autonomous Robotics
# Superimpose Turtle on AR Tag
#
# Author: Siddharth Telang(stelang@umd.edu)
# ========================================
# Run as 'python3 part2a.py'

from imutils import paths
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from myUtils import warpTurtle, warpARTag, findHomography, decodeImage

def main():
    # define max size for turtle warping and min size for AR tag warping 
    maxSize = 200
    minSize = 64

    #take input from user and start capturing video
    MAP = {1:'Tag0.mp4', 2:'Tag1.mp4', 3:'Tag2.mp4', 4:'multipleTags.mp4'}
    print('Input the file number to be rendered: \n 1: Tag0\n 2: Tag1\n 3: Tag2\n 4: MultipleTags\n')
    filenumber = int(input())
    filename = MAP.get(filenumber)
    if filename == 'None':
        print('Incorrect file number entered. Run again')
        return 0
 
    # set the output avi file
    index = filename.index('.')
    outputFile = filename[:index] + '_Warped.avi'
    result = cv2.VideoWriter(outputFile,cv2.VideoWriter_fourcc(*'XVID'), 30,(1280,720))    
    count = 1

    testudo = cv2.imread('test.png')
    testudo = cv2.resize(testudo, (maxSize,maxSize))
    dst1 = np.array([[0,0],[minSize,0],[minSize,minSize],[0,minSize]])
    dst2 = np.array([[0,0],[maxSize,0],[maxSize,maxSize],[0,maxSize]])

    # initialize lastDegree to be used when the TAG can not be decoded properly
    lastDegree = -2
    multiTag = False

    cap = cv2.VideoCapture(filename)
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (600,600))
        img = copy.copy(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 240, 255, 0)

        # find the contours
        img_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # initialize the lists for each frame
        contour_list, hierarchy_list, approx_list, count  = [], [], [], -1
        hierarchy_ = list(hierarchy[0])

        # find only the contours which have a parent and a child
        for contour,i in zip(img_contours,hierarchy_):
            # as soon as hierarchy hits a outermost parent contour, reset the count to 0
            if i[0] != -1 or i[1] != -1:
                count = 0
            else:
                count+=1
            # add the AR TAG contour in list
            if i[2] != -1 and i[3] != -1:
                if count > 1:
                    continue
                contour_list.append(contour)
                hierarchy_list.append(i)

        #im2 = copy.copy(img)

        for j in contour_list:
            #area = cv2.contourArea(j)
            # get the approx points of the contour
            approx = cv2.approxPolyDP(j, 0.025 * cv2.arcLength(j, True), True)
            if len(approx) != 4:
                #print('approx is not giving 4 points :/')
                #print('Contour count = ', str(len(contour_list)))
                #print(hierarchy_)
                continue
            #cv2.drawContours(im2, [approx], 0, (0, 0, 255),2)
            approx_list.append(approx)

        # approx list contains the points of AR TAG
        if approx_list == []:
            continue
        final = copy.copy(img)

        # set the multiTag flag is there are multiple tags - we won't be using lastDegree logic in multitag
        if len(approx_list) > 1 : multiTag = True

        # start processing for each AR TAG
        for k in range(len(approx_list)):
        #for k in range(0,1):
            src = approx_list[k].reshape(4,2)

            # warp the AR TAG
            h = findHomography(src, dst1)
            warped = warpARTag(gray, h, minSize,minSize)
            # find H for Turtle - different size than warped AR TAG
            h = findHomography(src, dst2)
            # decode the orientation
            orientation, valueTag, degrees, flag = decodeImage(warped)
            turtle = copy.copy(testudo)

            if (flag != 1 and not multiTag):
                print('Can not decode correct pose ; take the last well known pose')
                # take the last well known degree
                if lastDegree >= 0:
                    turtle = cv2.rotate(turtle, lastDegree)
            else:
                # orientation is found correctly, save it
                if degrees != -1:
                    turtle = cv2.rotate(turtle, degrees)
                lastOrientation = orientation
                lastDegree = degrees

            # warp the turtle
            warpedTurtle = warpTurtle(turtle, h, final)

            #cv2.imshow('turtle', turtle)
            tag = 'Tag' + str(k) + ' : ' + str(valueTag)
            warpedTurtle = cv2.putText(warpedTurtle, tag, (10, 30+(50*k)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
            final = copy.copy(warpedTurtle)

        cv2.imshow('video rendering', warpedTurtle)
        warpedTurtle = cv2.resize(warpedTurtle, (1280,720))
        result.write(warpedTurtle)

        if ret:
            if cv2.waitKey(25) & 0XFF == ord('q'):
                break
        else:
            break

    result.release()
    cap.release()
    cv2.destroyAllWindows()


if (__name__ == '__main__'):
    main()

