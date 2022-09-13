 
from collections import deque
import cv2

import numpy as np

import time

import socket



# define a video capture object
vid = cv2.VideoCapture(0)

#initalize socket stuff
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

#initalize counter for number of frames
count = 1

#how much you want to crop top and bottom
crop=0

#make lists of corners
corner1xls=deque()
corner1yls=deque()
corner2xls=deque()
corner2yls=deque()
corner3xls=deque()
corner3yls=deque()
corner4xls=deque()
corner4yls=deque()

#global corners
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

#splits location of finger into 64x64 grid
def revisedScaler(inputx, inputy, corner1xin, corner1yin, corner2xin, corner2yin, corner3xin, corner3yin, corner4xin, corner4yin):

    corner1x=corner1xin
    corner1y=corner1yin
    corner2x=corner2xin
    corner2y=corner2yin
    corner3x=corner3xin
    corner3y=corner3yin
    corner4x=corner4xin
    corner4y=corner4yin

    #x finder thing
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

    #y finder thing
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

#average paper edges to make things smoother
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

#initalize global average corner locations
drawCorner1x,drawCorner1y,drawCorner2x,drawCorner2y,drawCorner3x,drawCorner3y,drawCorner4x,drawCorner4y = 0,0,0,0,0,0,0,0

while(True):
    #initial time read
    time1=time.time()

    #make blank paper grid
    papershow = np.zeros((64,64))

    #get frame from webcam
    _, frameO = vid.read()
    
    #testimage = "/home/<Username/images/c1.png"
    #frameO = cv2.imread(r(testimage))

    frame = cv2.resize(frameO,(640,480))    

    #crop if necessary
    #frameCrop= frame[crop:(480-crop),0:640]
    frameCrop = frame

    #apply filters to make edges more visible
    frameGrey = cv2.cvtColor(frameCrop, cv2.COLOR_BGR2GRAY)
    frameOBlur = cv2.GaussianBlur(frameGrey, (5,5), 0)
    frameOEdge = cv2.Canny(frameOBlur, 50, 150, apertureSize=3)

    #corner detection
    corners = ()
    
    lineColor = (0,0,255)
    trackCorner = 150>count

    #track corners
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
    
    #make lines on edges
    cv2.line(frameCrop, (drawCorner1x, drawCorner1y), (drawCorner2x, drawCorner2y), lineColor, 15)
    cv2.line(frameCrop, (drawCorner2x, drawCorner2y), (drawCorner3x, drawCorner3y), lineColor, 15)
    cv2.line(frameCrop, (drawCorner3x, drawCorner3y), (drawCorner4x, drawCorner4y), lineColor, 15)
    cv2.line(frameCrop, (drawCorner4x, drawCorner4y), (drawCorner1x, drawCorner1y), lineColor, 15)

    #find center
    cornersArray= np.array([[drawCorner1x,drawCorner2x, drawCorner3x, drawCorner4x],[drawCorner1y,drawCorner2y,drawCorner3y,drawCorner4y]])
    center = np.array([np.mean(cornersArray[0]),np.mean(cornersArray[1])])
    
    #shadow mask

    #convert frame to something easier to work with
    hsv = cv2.cvtColor(frameCrop, cv2.COLOR_BGR2HSV)

    #filter out finger shadow (modify for your use case depending on color temperature of light source)
    low_shd = np.array([100, 77, 66])
    high_shd = np.array([120, 142, 153])
    shd_maskraw = cv2.inRange(hsv, low_shd, high_shd)

    #remove outside of paper boundary
    paper_bound = np.zeros(shd_maskraw.shape).astype(shd_maskraw.dtype)

    #draw shape to hide paper shadow and other outside shadows
    contours = np.array([[drawCorner1x,drawCorner1y],[drawCorner2x,drawCorner2y],[drawCorner3x,drawCorner3y],[drawCorner4x,drawCorner4y]])
    cv2.fillPoly(paper_bound, pts = [contours],color=(255))

    #mask is only where there is finger shadow and gets rid of paper shadow
    shd_mask = cv2.bitwise_and(shd_maskraw, paper_bound, mask = None)

    #find finger

    #get coordinates where the mask shows your fingies
    pixels = np.where(shd_mask == 255)

    #initialize finger location things
    fingerx = -1
    fingery = -1

    #average locations of where mask found pixels corresponding to finger shadow
    try:
        fingerx = int(np.mean(pixels[1]))
        fingery = int(np.mean(pixels[0]))
    except:
        print("no finger")

    #get rid of when finger location is negative for whatever reason
    if fingerx >=0 and fingery >=0:
        #draw circles where it thinks your finger shadow is
        cv2.circle(frameCrop, (fingerx, fingery), 4, (0,255,0), -1)
        cv2.circle(shd_mask, (fingerx, fingery), 4, (255), -1)

    #Uncomment to show center
    #cv2.circle(frameCrop, (int(center[0]),int(center[1])), 4, (0,0,255), -1)
    
    count=count+1

    #Uncomment to show mid lines in image for troubleshooting purposes
    """edge1 =(int(np.mean((drawCorner1x,drawCorner2x))),int(np.mean((drawCorner1y,drawCorner2y))))
    edge2 =(int(np.mean((drawCorner2x,drawCorner3x))),int(np.mean((drawCorner2y,drawCorner3y))))
    edge3 =(int(np.mean((drawCorner3x,drawCorner4x))),int(np.mean((drawCorner3y,drawCorner4y))))
    edge4 =(int(np.mean((drawCorner4x,drawCorner1x))),int(np.mean((drawCorner4y,drawCorner1y)))) 
    
    #get mean point between corners and draw line to opposide side mean point between corners
    cv2.line(frameCrop, edge1,edge3, (255,0,0),2)
    cv2.line(frameCrop, edge2,edge4, (255,0,0),2)"""

    #show original image with fancy lines
    cv2.imshow('original', frameCrop)

    #call scaler function
    scaledx, scaledy= revisedScaler(fingerx, fingery, drawCorner1x, drawCorner1y, drawCorner2x, drawCorner2y, drawCorner3x, drawCorner3y, drawCorner4x, drawCorner4y)
    
    #print input and output to look for strange occurences
    print("rawinput: "+str(fingerx)+", "+str(fingery))
    print("scaledoutput: "+str(scaledx)+", "+str(scaledy))

    #show where the point is on grid
    papershow[scaledy, scaledx]=1
    cv2.imshow('paper', papershow)

    #useful buttons
    if cv2.waitKey(1) & 0xFF == ord('r'):
        count=0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        status = cv2.imwrite('/home/lee/Pictures/save.png',frameCrop)
        print(status)
        break
    
    #outputting
    sendData(scaledx, scaledy, 64, 64)

    #show performance
    time2 = time.time()
    fps = 1/(time2-time1)
    print(fps)



# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()