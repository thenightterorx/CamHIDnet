
from contextlib import nullcontext
import pyautogui
import socket       
import sys
 
# Create a socket object
s = socket.socket()     

ip = str(sys.argv[0])
hostname = str(sys.argv[1])
 
# Define the port on which you want to connect
port = 12354
port1 = port+1

sizex,sizey = pyautogui.size()

pyautogui.moveTo(sizex/2,sizey/2,duration=0)
 
# connect to the server on local computer

try:
    try:
        s.connect((socket.gethostbyname(hostname), port))
    except:
        s.connect((socket.gethostbyname(hostname), port1))
except:
    try:
        s.connect((ip, port))
    except:
        s.connect((ip, port1))

previousx=-1
previousy=-1

recievedx=10
recievedy=10
 
# receive data from the server and decoding to get the string.
while 1==1:
    try:
        recieved = s.recv(1024).decode()
        print(recieved)
        print("test")
        
        if (recieved[0]=="x"):
            recievedx=int(recieved[1:])

        if (recieved[0]=="y"):
            recievedy=int(recieved[1:])

        """if (xpos>0):
            if(ypos>xpos):
                    recievedx = int(recieved[xpos:(ypos-2)])
            else:
                    recievedx = int(recieved[xpos:])
        else:
            recievedx = 0


        if (ypos>0):
            if(xpos>ypos):
                recievedy = int(recieved[ypos:(xpos-2)])
            else:
                recievedy = int(recieved[ypos:])
        else:
            recievedy = 0  """

        if(previousx!=-1 and previousy!=-1 and recievedx!=-1 and recievedy!=-1):
            xmove= recievedx-previousx
            ymove= recievedy-previousy
            pyautogui.moveRel(xmove, ymove, duration=0)

        

        
    except:
        print("error in parsing coords")


# close the connection

"""x = 100
y = 100
pyautogui.moveRel(x, y)
pyautogui.moveRel(x, y)"""


    