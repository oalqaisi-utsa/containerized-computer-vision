#
# Osamah Alqaisi
# Osamah.Alqaisi@my.UTSA.edu
# 
# This code use Histogram of Oriented Gradients (HOG)  
# to detect Body from images
# Images recived by network on port# 8887
# 
#

import socket, cv2 as cv
import numpy as np
import time

# Read Haar Cascade file
haar_file = "cars.xml"
haar_cascade = cv.CascadeClassifier(haar_file)

port_recive = 8888         # Port to listen on 
recive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # AF_INET = IP, SOCK_STREAM = TCP


recive.bind( ('', port_recive) )
recive.listen()
recive_socket, recive_address = recive.accept()  
print("Connect - %s" %str(recive_address))

while True: 
    
    #Calculating waiting time
    start_time = cv.getTickCount()
    data = recive_socket.recv(512)
    waiting_time = (cv.getTickCount()-start_time)/cv.getTickFrequency()
    
    # Calculating Rec image
    start_time = cv.getTickCount()
    header = data.decode('utf-8')
    header = str(header).split(":")
    image_name = header[0]
    image_size = int(header[1])
    recive_socket.sendall(b'Done')
    
    data = b''
    while len(data) < image_size:
        data += recive_socket.recv(4096) 
    recive_socket.sendall(b'Done')

    rec_time = (cv.getTickCount()-start_time)/cv.getTickFrequency()

    # Start Timer
    start_time = cv.getTickCount()
    img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
    output = img
    output = cv.resize(output, (600,400) )
    # Convert image to GRAY scale
    gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    # Call Haar Cascade recognition
    obj_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    num_detect = 0
    if len(obj_rect) > 0:
        # Drow rectangle on object
        for (x,y,w,h) in obj_rect:
            cv.rectangle(output, (x,y), (x+w,y+h), (0,255,0), thickness=2)
            num_detect = num_detect + 1
        # Save output image 
        cv.imwrite(image_name, output)
    # Print the excution time
    print("Wait %.9s s - Rec %s - %.9s -Process %.9s - %s" %(waiting_time , image_name, rec_time , ( (cv.getTickCount()-start_time)/cv.getTickFrequency() ), str(num_detect) ) )
    


recive_socket.close()
