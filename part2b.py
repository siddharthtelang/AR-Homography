# ========================================
# ENPM673 Spring 2021: Perception for Autonomous Robotics
# Projecting 3D cube on AR Tag
#
# Author: Siddharth Telang(stelang@umd.edu)
# ========================================
# Run as 'python3 part2b.py'

from imutils import paths
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from myUtils import calcProjectionMatrix, drawCube, findHomography
    
def main():

    maxSize = 200

    K =np.array([[1406.08415449821,0,0],
       [ 2.20679787308599, 1417.99930662800,0],
       [ 1014.13643417416, 566.347754321696,1]])

    dst = np.array([[0,0],[maxSize,0],[maxSize,maxSize],[0,maxSize]])

    #take input from user and start capturing video
    MAP = {1:'Tag0.mp4', 2:'Tag1.mp4', 3:'Tag2.mp4', 4:'multipleTags.mp4'}
    print('Input the file number to be rendered: \n 1: Tag0\n 2: Tag1\n 3: Tag2\n 4: MultipleTags\n')
    filenumber = int(input())
    filename = MAP.get(filenumber)
    if filename == None:
        print('Incorrect file number entered. Run again')
        exit()

    # set the output avi file
    index = filename.index('.')
    outputFile = filename[:index] + '_Cube.avi'
    result = cv2.VideoWriter(outputFile,cv2.VideoWriter_fourcc(*'XVID'), 30,(1280,720))    
    print(outputFile)
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

        # initializethe lists for each frame
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

        for j in contour_list:
            area = cv2.contourArea(j)
            approx = cv2.approxPolyDP(j, 0.025 * cv2.arcLength(j, True), True)
            if len(approx) != 4:
                print('approx is not giving 4 points :/')
                print('Contour count = ', str(len(contour_list)))
                print(hierarchy_)
                continue
            approx_list.append(approx)
        if approx_list == []:
            continue

        final = copy.copy(img)

        # start processing for each AR TAG
        for k in range(len(approx_list)):
            src = approx_list[k].reshape(4,2)

            # find the homography matrix
            h = findHomography(dst, src)

            # calculate projection matrix
            Pr, Rot, tr = calcProjectionMatrix(h,K)

            # draw the cube
            final = drawCube(Pr, maxSize, final)

        cv2.imshow('Cube', final)
        final = cv2.resize(final, (1280,720))
        # write in video file
        result.write(final)


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

