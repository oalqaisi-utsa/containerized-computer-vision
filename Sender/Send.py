#
#
#
#
#
#
#
#
#

import time
import random
import cv2 as cv

import socket
HOST = '192.168.10.103'
PORT = 8889
send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # AF_INET = IP, SOCK_STREAM = TCP
send.connect(( HOST , PORT ))
send.settimeout(1800)


from os import path
path = path.dirname(__file__) + '\\'

image = None
countimage = 1
print("start")
for i in range (1,5):
    for j in range (400, 1400) :

            start_time = cv.getTickCount()

            image = str(j) + ".jpg"
            file = open(image, 'rb')
            image_data = file.read()
            image_size = len(image_data)

            header = str(image) + ":" + str(image_size)
            send.sendall( header.encode() )
            answer = send.recv(16)

            send.sendall(image_data)
            answer = send.recv(16)

            print("%s - Sent: %s - %.9s s" % (str(countimage), image, (cv.getTickCount()-start_time)/cv.getTickFrequency() ) )
            countimage = countimage + 1

            file.close()
            time.sleep(1)

send.close()
