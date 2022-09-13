 
from collections import deque
import cv2

import numpy as np

import time

import socket



# define a video capture object
vid = cv2.VideoCapture(0)
#id.set(cv2.CAP_PROP_EXPOSURE, 40)
#print(vid.get(cv2.CAP_PROP_EXPOSURE))

s = socket.socket()
print("A")

port = 12354
try:
  s.bind(('',port))
except:
  s.bind(('',port+1))
print("B")

s.listen(5)
print("C")

count = 1

testfingerx= 67
testfingery= 100


iterations = 7

examplex=100

crop = 100

corner1xls=deque()
corner1yls=deque()
corner2xls=deque()
corner2yls=deque()
corner3xls=deque()
corner3yls=deque()
corner4xls=deque()
corner4yls=deque()

"""xout=-1
yout=-1"""

corner1x = 0
corner1y = 0
corner2x = 0
corner2y = 0
corner3x = 0
corner3y = 0
corner4x = 0
corner4y = 0
# Establish connection with client.
connection = False
while 1==1:
    c, addr = s.accept()  
    
      

    print ('Got connection from', addr )
    break

xprev = -1
yprev = -1

#send stuff
def sendData(x, y, xsize, ysize):
    global xprev
    global yprev
    xstr = 'x'+str(x)
    ystr = 'y'+str(y)
    if(xprev!=x or yprev!=y):
        
        c.send(xstr.encode())
        c.send(ystr.encode())
    xprev = x
    yprev = y

def revisedScaler(inputx, inputy, corner1xin, corner1yin, corner2xin, corner2yin, corner3xin, corner3yin, corner4xin, corner4yin):

    corner1x=corner1xin
    corner1y=corner1yin
    corner2x=corner2xin
    corner2y=corner2yin
    corner3x=corner3xin
    corner3y=corner3yin
    corner4x=corner4xin
    corner4y=corner4yin

    go = 5
    locationx=32
    while go>0:
        topMiddle=(((corner1x+corner2x)/2),((corner1y+corner2y)/2))
        bottomMiddle=(((corner3x+corner4x)/2),((corner3y+corner4y)/2))
        slope,intercept=lineFinder(topMiddle[0],bottomMiddle[0],topMiddle[1],bottomMiddle[1])
        pointx=(fingery-intercept)/slope

        go=go-1

        if(inputx>pointx):
            corner2x=topMiddle[0]
            corner2y=topMiddle[1]
            corner3x=bottomMiddle[0]
            corner3y=bottomMiddle[1]
            locationx=locationx+2**go
        else:
            corner1x=topMiddle[0]
            corner1y=topMiddle[1]
            corner4x=bottomMiddle[0]
            corner4y=bottomMiddle[1]
            locationx=locationx-2**go

    go = 5
    locationy=32
    while go>0:
        leftMiddle=(((corner2x+corner3x)/2),((corner2y+corner3y)/2))
        rightMiddle=(((corner4x+corner1x)/2),((corner4y+corner1y)/2))
        slope,intercept=lineFinder(leftMiddle[0],rightMiddle[0],leftMiddle[1],rightMiddle[1])
        pointy=(fingery-intercept)/slope

        go=go-1

        if(inputy>pointy):
            corner3x=leftMiddle[0]
            corner3y=leftMiddle[1]
            corner4x=rightMiddle[0]
            corner4y=rightMiddle[1]
            locationy=locationy+2**go
        else:
            corner1x=rightMiddle[0]
            corner1y=rightMiddle[1]
            corner2x=leftMiddle[0]
            corner2y=leftMiddle[1]
            locationy=locationy-2**go

    return locationx, locationy
    


#scale finger from physical paper to virtual paper? Idk
def fingerScaler(iterations, inputx, inputy, corner1xin, corner1yin, corner2xin, corner2yin, corner3xin, corner3yin, corner4xin, corner4yin):
    iteration = 1

    loop = True
    direction = 1

    previousTopx = corner2xin
    previousTopy = corner2yin
    previousBottomx = corner3xin
    previousBottomy = corner3yin

    limitxTopRight = corner1xin
    limitxTopLeft = corner2xin
    limitxBottomRight = corner4xin
    limitxBottomLeft = corner3xin

    limityTopRight = corner1yin
    limityTopLeft = corner2yin
    limityBottomRight = corner4yin
    limityBottomLeft = corner3yin

    #find x
    index =0
    while loop==True:
        splitTopx= midpointFinder(direction, previousTopx, limitxTopLeft, limitxTopRight, iteration)
        splitTopy= midpointFinder(direction, previousTopy, limityTopLeft, limityTopRight, iteration)
        splitBottomx = midpointFinder(direction, previousBottomx, limitxBottomLeft, limitxBottomRight, iteration)
        splitBottomy = midpointFinder(direction, previousBottomy, limityBottomLeft, limityBottomRight, iteration)
        
        intercept, slope = lineFinder(splitBottomx, splitTopx, splitBottomy, splitTopy)
        if(fingerx):
            direction = 1
            index = index + 2**iteration
        else:
            direction = -1
            index = index - 2**iteration

        previousTopx = splitTopx 



        iteration=iteration+1

        if (iteration>=iterations):
            loop = False

    #find point on edge 3

    #find point on edge 4

    #convert edge points to x and y values
    x=1
    y=1
    return index

#function takes 2 values (function used for each edge) and finds a kind of weighted midpoint idk how to explain it
def midpointFinder(direction, previous, limit1, limit2, iteration):
    width = limit2-limit1
    move = direction*(width/(2**iteration))
    new= previous+move

    if(new>previous):
        return previous,new
    else:
        return new,previous

#find intercept and slope from 2 points
def lineFinder(x1,x2,y1,y2):
    try:
        slope= (y1-y2)/(x1-x2)
        intercept= y2-(slope*x2)

        return (slope,intercept)
    except:
        return (0,0)


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

drawCorner1x,drawCorner1y,drawCorner2x,drawCorner2y,drawCorner3x,drawCorner3y,drawCorner4x,drawCorner4y = 0,0,0,0,0,0,0,0

while(True):
    papershow = np.zeros((64,64))
    time1=time.time()
    #_, frame = vid.read()
    frameO = cv2.imread(r"/home/lee/images/c1.png")

    frame = cv2.resize(frameO,(640,480))

    
    
    # paper mask
    """low_ong = np.array([0, 0, 120])
    high_ong = np.array([180, 133, 255])
    mask = cv2.inRange(hsv, low_ong, high_ong)"""

    #frameCrop= frame[crop:(480-crop),0:640]
    frameCrop = frame
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
    trackCorner = 150>count
    
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

    cornersArray= np.array([[drawCorner1x,drawCorner2x, drawCorner3x, drawCorner4x],[drawCorner1y,drawCorner2y,drawCorner3y,drawCorner4y]])
    center = np.array([np.mean(cornersArray[0]),np.mean(cornersArray[1])])
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

#test finger location

    fingerx = testfingerx
    fingery = testfingery


    if fingerx >=0 and fingery >=0:
        cv2.circle(frameCrop, (fingerx, fingery), 4, (0,255,0), -1)
        cv2.circle(shd_mask, (fingerx, fingery), 4, (255), -1)

    
    #cv2.imshow('shadow', shd_mask)
    #cv2.imshow('paper', paper_bound)
    cv2.circle(frameCrop, (int(center[0]),int(center[1])), 4, (0,0,255), -1)
    

    #cv2.imshow('finger', fing_mask)
    #cv2.imshow('frameBlur', maskBlur1)
    #cv2.imshow('egde', maskEdge)
    #cv2.imshow('line', frameOEdge)

    count=count+1

    

    edge1 =(int(np.mean((drawCorner1x,drawCorner2x))),int(np.mean((drawCorner1y,drawCorner2y))))
    edge2 =(int(np.mean((drawCorner2x,drawCorner3x))),int(np.mean((drawCorner2y,drawCorner3y))))
    edge3 =(int(np.mean((drawCorner3x,drawCorner4x))),int(np.mean((drawCorner3y,drawCorner4y))))
    edge4 =(int(np.mean((drawCorner4x,drawCorner1x))),int(np.mean((drawCorner4y,drawCorner1y)))) 
    #cv2.imshow('paper', paper)
    #get mean point between corners and draw line to opposide side mean point between corners
    cv2.line(frameCrop, edge1,edge3, (255,0,0),2)
    cv2.line(frameCrop, edge2,edge4, (255,0,0),2)

    #cv2.line(frameCrop, (int(np.mean()),int(np.mean())),(int(np.mean()),int(np.mean())),(255,0,0),2)
    """ cv2.line(frameCrop, (int(np.mean(corner1x))), (480-2*crop)),(int(center[0]),0),(255,0,0),2)
    cv2.line(frameCrop, (640, int(center[1])),(0, int(center[1])),(255,0,0),2)"""
    #find x and y of point
    """
    #convert paper outline to polar coordinates

    #rotate a line around
    #start at 0 radians
    #progress ray until true
    #distance from center is distance

    #divide distance to edge in actual paper by distance to edge on paper_bound to get an array of angles and ratios

    #multiply finger point by ratio to get actual finger location"""

    #convert point in 16384 regions

    #find x area
    """left = 0
    weight = 0
    while iterations>0:
        if (fingerx>((corner3x+corner4x)/2) or fingerx>((corner1x+corner2x)/2)):
            left = 0
            weight = weight + 2**(iterations-1)
        else:
            left = 1


        iterations=iterations-1"""

    """#alternate method: convert edges to functions
    slope,intercept = lineFinder(corner2x,corner3x,corner2y,corner3y)
    print(slope,intercept)

    examplex = examplex+1
    if (examplex>160):
        examplex=0
        """
    
    """excircley = int(((examplex*slope))+intercept)
    cv2.circle(frameCrop, (int(examplex), excircley), 10, (200,200,200), -1)
    cv2.circle(frameCrop, (100, 100), 10, (200,200,200), -1)
    print("examplex")
    print(examplex)
    print("exampley")
    print(excircley)"""

    cv2.imshow('original', frameCrop)
    
    """xout = fingerx
    yout = fingery

    if yout>=0 and xout >=0:
        print (xout)
        print (yout)"""

    testx, testy= revisedScaler(fingerx, fingery, drawCorner1x, drawCorner1y, drawCorner2x, drawCorner2y, drawCorner3x, drawCorner3y, drawCorner4x, drawCorner4y)
    
    print("rawinput: "+str(fingerx)+", "+str(fingery))
    print("scaledoutput: "+str(testx)+", "+str(testy))
    papershow[testy, testx]=1
    cv2.imshow('paper', papershow)

    if cv2.waitKey(1) & 0xFF == ord('r'):
        count=0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        status = cv2.imwrite('/home/lee/Pictures/save.png',frameCrop)
        print(status)
        break
    time2 = time.time()
    fps = 1/(time2-time1)

    #outputting
    sendData(testx, testy, 64, 64)
    #print(fps)



# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
