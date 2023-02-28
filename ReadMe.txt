PROPOSED SCHEME:
We build an edge system that receives images from IoT sensors. The edge device analyzes images and takes the necessary decision. The edge system is a Raspberry Pi 4 device where a virtual container will be installed. The container has the primary library for computer vision, which is the OpenCV library. The container will be reduced as much as possible to accommodate the Raspberry Pi 4 device. Applications of computer vision will be programmed in C++ and Python.

Containers:
C++ with OpenCV container:
docker pull oalqaisi/cv_cpp
Python with OpenCV container:
docker pull oalqaisi/cv_python

All containers don't have the codes files. You have to upload the codes files to the containers and run them manually.

Files:
CPP and Python folders have four different applications. 
1- Face detection. (Haar Cascades algorithm)
2- Car detection. (Haar Cascades algorithm) 
3- Body detection. (HOG algorithm)
4- Object detection. (CNN by YOLO)


Running C++ Files on Docker Images:
g++ XXX.cpp -o YYY.out -std=c++11 `pkg-config --cflags --libs opencv`
./YYY.out

Running Python Files on Docker Images:
python3 XXX.py
