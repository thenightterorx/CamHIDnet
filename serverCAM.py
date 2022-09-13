 
from collections import deque
import cv2

import numpy as np

import time

import os

from matplotlib import pyplot as plt

# define a video capture object
vid = cv2.VideoCapture(2)
#id.set(cv2.CAP_PROP_EXPOSURE, 40)
#print(vid.get(cv2.CAP_PROP_EXPOSURE))
c = 1

crop = 100

corner1xls=deque()
corner1yls=deque()
corner2xls=deque()
corner2yls=deque()
corner3xls=deque()
corner3yls=deque()
corner4xls=deque()
corner4yls=deque()

xout=-1
yout=-1

corner1x= 0
corner1y= 0
corner2x = 0
corner2y = 0
corner3x = 0
corner3y = 0
corner4x = 0
corner4y = 0

drawCorner1x = 0
drawCorner1y = 0
drawCorner2x = 0
drawCorner2y = 0
drawCorner3x = 0
drawCorner3y = 0
drawCorner4x = 0
drawCorner4y = 0

def lineLister(corner1xin, corner1yin, corner2xin, corner2yin, corner3xin, corner3yin, corner4xin, corner4yin):
    corner1xls.append(corner1xin)
    corner1yls.append(corner1yin)
    corner2xls.append(corner2xin)
    corner2yls.append(corner2yin)
    corner3xls.append(corner3xin)
    corner3yls.append(corner3yin)
    corner4xls.append(corner4xin)
    corner4yls.append(corner4yin)

    if (len(corner1xls)>99):
        corner1xls.popleft()
    if (len(corner1yls)>99):
        corner1yls.popleft()    
    if (len(corner2xls)>99):
        corner2xls.popleft()
    if (len(corner2yls)>99):
        corner2yls.popleft()
    if (len(corner3xls)>99):
        corner3xls.popleft()
    if (len(corner3yls)>99):
        corner3yls.popleft()
    if (len(corner4xls)>99):
        corner4xls.popleft()
    if (len(corner4yls)>99):
        corner4yls.popleft()
    #print(corner1xin)
    return int(np.mean(corner1xls)),int(np.mean(corner1yls)),int(np.mean(corner2xls)),int(np.mean(corner2yls)),int(np.mean(corner3xls)),int(np.mean(corner3yls)),int(np.mean(corner4xls)),int(np.mean(corner4yls))


while(True):
    time1=time.time()
    _, frame = vid.read()
    
    
    # paper mask
    """low_ong = np.array([0, 0, 120])
    high_ong = np.array([180, 133, 255])
    mask = cv2.inRange(hsv, low_ong, high_ong)"""

    frameCrop= frame[crop:(480-crop),0:640]
    #paper line finder 2.0
    frameGrey = cv2.cvtColor(frameCrop, cv2.COLOR_BGR2GRAY)
    frameOBlur = cv2.GaussianBlur(frameGrey, (5,5), 0)
    frameOEdge = cv2.Canny(frameOBlur, 50, 150, apertureSize=3)

    #print(frameCrop.shape)


    
    """#shadow mask 1
    low_shd1 = np.array([0, 0, 51])
    high_shd1 = np.array([5, 92, 77])
    shd_mask1 = cv2.inRange(hsv, low_shd1, high_shd1)

    #shadow mask 2
    low_shd2 = np.array([120, 0, 51])
    high_shd2 = np.array([180, 92, 77])
    shd_mask2 = cv2.inRange(hsv, low_shd2, high_shd2)

    shd_mask = cv2.bitwise_or(shd_mask2,shd_mask1, mask=None)"""

    

    #fingernail detection
    """low_fing = np.array([16,114,176])
    high_fing = np.array([20,140,196])
    fing_mask = cv2.inRange(hsv, low_fing, high_fing)"""

    #try to get frame of paper
    """maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    maskRes = cv2.resize(maskBGR, (1280, 960))
    maskBlur = cv2.GaussianBlur(maskRes, (5,5),0)
    maskBlur1 = cv2.GaussianBlur(maskBlur, (5,5), 0)
    maskBlur2 = cv2.GaussianBlur(maskBlur1, (5,5), 0)
    maskGray = cv2.cvtColor(maskBlur2, cv2.COLOR_BGR2GRAY)
    maskEdge = cv2.Canny(maskGray, 50, 150, apertureSize=3)"""
    #corner detection
    corners = ()
    

    
    lineColor = (0,0,255)
    trackCorner = 150>c
    
    if trackCorner:
        try:
            #print (c)
            corners = cv2.goodFeaturesToTrack(frameOEdge, 4, 0.2, 170)
            corners = np.int0(corners)
            for i in corners:
                x, y = i.ravel()
                cv2.circle(frameCrop, (x, y), 4, (100,100,100), -1)
                if (x<320 and y>(240-crop)):
                    corner2x = x
                    corner2y = y
                if (x<320 and y<(240-crop)):
                    corner3x = x
                    corner3y = y
                if (x>320 and y>(240-crop)):
                    corner1x = x
                    corner1y = y
                if (x>320 and y<(240-crop)):
                    corner4x = x
                    corner4y = y
            drawCorner1x,drawCorner1y,drawCorner2x,drawCorner2y,drawCorner3x,drawCorner3y,drawCorner4x,drawCorner4y = lineLister(corner1x,corner1y,corner2x,corner2y,corner3x,corner3y,corner4x,corner4y)
        except:
            print("corner detection failed")
    

    #draw connecting lines
    """drawCorner1 = np.array([int(corner1[0]/i,corner1[1]/i)])
    drawCorner2 = np.array([int(corner2[0]/i,corner2[1]/i)])
    drawCorner3 = np.array([int(corner3[0]/i,corner3[1]/i)])
    drawCorner4 = np.array([int(corner4[0]/i,corner4[1]/i)])"""
    
    
    cv2.line(frameCrop, (drawCorner1x, drawCorner1y), (drawCorner2x, drawCorner2y), lineColor, 15)
    cv2.line(frameCrop, (drawCorner2x, drawCorner2y), (drawCorner3x, drawCorner3y), lineColor, 15)
    cv2.line(frameCrop, (drawCorner3x, drawCorner3y), (drawCorner4x, drawCorner4y), lineColor, 15)
    cv2.line(frameCrop, (drawCorner4x, drawCorner4y), (drawCorner1x, drawCorner1y), lineColor, 15)
    #lines crap
    """rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 90  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(maskEdge, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),5)"""
    
    #cv2.imshow('frame', maskBlur2)
    #shadow mask

    hsv = cv2.cvtColor(frameCrop, cv2.COLOR_BGR2HSV)

    low_shd = np.array([100, 77, 66])
    high_shd = np.array([120, 142, 153])
    shd_maskraw = cv2.inRange(hsv, low_shd, high_shd)

    #remove outside of paper boundary
    paper_bound = np.zeros(shd_maskraw.shape).astype(shd_maskraw.dtype)
    contours = np.array([[drawCorner1x,drawCorner1y],[drawCorner2x,drawCorner2y],[drawCorner3x,drawCorner3y],[drawCorner4x,drawCorner4y]])
    cv2.fillPoly(paper_bound, pts = [contours],color=(255))

    shd_mask = cv2.bitwise_and(shd_maskraw, paper_bound, mask = None)
    #shd_mask = shd_maskraw


    #find finger
    pixels = np.where(shd_mask == 255)
    fingerx = -1
    fingery = -1
    try:
        fingerx = int(np.mean(pixels[1]))
        fingery = int(np.mean(pixels[0]))
    except:
        print("no finger")


    if fingerx >=0 and fingery >=0:
        cv2.circle(frameCrop, (fingerx, fingery), 4, (0,255,0), -1)
        cv2.circle(shd_mask, (fingerx, fingery), 4, (255), -1)

    #cv2.imshow('shadow', shd_mask)
    #cv2.imshow('paper', paper_bound)
    cv2.imshow('original', frameCrop)

    #cv2.imshow('finger', fing_mask)
    #cv2.imshow('frameBlur', maskBlur1)
    #cv2.imshow('egde', maskEdge)
    #cv2.imshow('line', frameOEdge)

    c=c+1

    #scale to paper
    A11= np.zeros((2508,3840))
    cv2.circle(A11, (fingerx, fingery), 4, (255), -1)

    xout = fingerx
    yout = fingery
    if yout>=0 and xout >=0:
        print (xout)
        print (yout)

    cv2.imshow('paper',A11)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        c=0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        status = cv2.imwrite('/home/lee/Pictures/save.png',frameCrop)
        print(status)
        break
    time2 = time.time()
    fps = 1/(time2-time1)
    #print(fps)
    

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()