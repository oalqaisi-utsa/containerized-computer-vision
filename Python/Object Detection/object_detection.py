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

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Initialize the parameters
conf = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

# Load names of classes
classesFile = "yolov3.txt";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg";
modelWeights = "yolov3.weights";
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


port_recive = 8880         # Port to listen on 
recive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # AF_INET = IP, SOCK_STREAM = TCP
recive.bind( ('', port_recive) )
recive.listen()
recive_socket, recive_address = recive.accept()  
print("Connect - %s" %str(recive_address))
while True:
    #Calculating waiting time
    start_time = cv.getTickCount()
    data = recive_socket.recv(512)
    waiting_time = (cv.getTickCount() - start_time) / cv.getTickFrequency()
    
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

    rec_time = (cv.getTickCount() - start_time) / cv.getTickFrequency()

    # Start Timer
    start_time = cv.getTickCount()
    img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
    output = img
    output = cv.resize(output, (600,400) )


    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(output, 1/255, (416, 416),[0,0,0], swapRB=True, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outputs = net.forward(getOutputsNames(net))
    
    boxes = []
    confidences = []
    classIds = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf:
                center_x = int(detection[0] * output.shape[1])
                center_y = int(detection[1] * output.shape[0])
                width = int(detection[2] * output.shape[1])
                height = int(detection[3] * output.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, nmsThreshold)

    num_detect = 0
    if len(indices) > 0:
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv.rectangle(output, (left, top), ((left+width), (top+height)), (255, 178, 50), 2)
            num_detect = num_detect + 1

        # Save output image
        cv.imwrite(image_name, output)
    
    # Print the excution time
    print("Wait %.9s s - Rec %s - %.9s -Process %.9s - %s" %(waiting_time , image_name, rec_time , ((cv.getTickCount() - start_time) / cv.getTickFrequency()), str(num_detect) ))    
