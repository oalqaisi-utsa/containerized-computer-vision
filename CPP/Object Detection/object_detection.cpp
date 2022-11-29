/*
 * Osamah Alqaisi
 * UTSA
 * 
 * This code use pre-trained YOLO: Real-Time Object Detection files  
 * to detect many objects from images
 * Images recived by network on port# 8880
 * 
*/

// Server side C/C++ program to demonstrate Socket programming
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <time.h>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <fstream>

using namespace std;
using namespace cv;
#define PORT 8880

using namespace dnn;

void detect(Net net, vector<String> ln, uchar sockData[], cv::Mat image, string out_image, int imgSize)
{
    double start_time = cv::getTickCount();
    int num_detect = 0;
    image = cv::imdecode(cv::Mat(1, imgSize, CV_8UC1, sockData),  IMREAD_UNCHANGED);
    resize(image, image, Size(600, 400));

    Mat blob;
    // Create a 4D blob from a frame.
    blobFromImage(image, blob, 1/255.0, cv::Size(416, 416), Scalar(0,0,0), true, false);
    //Sets the input to the network
    net.setInput(blob);     
    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    net.forward(outs, ln);

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > 0.5)
            {
                int centerX = (int)(data[0] * image.cols);
                int centerY = (int)(data[1] * image.rows);
                int width = (int)(data[2] * image.cols);
                int height = (int)(data[3] * image.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                 
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
     
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
    if(indices.size() > 0)
    {
        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            Rect box = boxes[idx];
            rectangle(image, Point(box.x, box.y), Point((box.x + box.width), (box.y + box.height)), Scalar(0, 255, 0), 2);
            num_detect++;
		}
		imwrite( out_image, image );
	}
	cout << "Process " << ((cv::getTickCount()-start_time)/cv::getTickFrequency()) << " - " << num_detect << endl;
}


// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();      
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

int main(int argc, char const* argv[])
{
	

	// Socket Var
	int server_fd, new_socket, valread ;
	struct sockaddr_in address;
	int opt = 1;
	int addrlen = sizeof(address);

	// Creating socket file descriptor
	if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
	perror("socket failed");
		return 0 ;
	}
	// Forcefully attaching socket to the port 8080
	if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
	perror("setsockopt");
	return 0 ;
	}

	/* setup the host_addr structure for use in bind call */
	// server byte order
	address.sin_family = AF_INET;
	// automatically be filled with current host's IP address
	address.sin_addr.s_addr = INADDR_ANY;
	// convert short integer value for port must be converted into network byte order
	address.sin_port = htons(PORT);

	// Forcefully attaching socket to the port
	if (bind(server_fd, (struct sockaddr*)&address,sizeof(address)) < 0) {
	perror("bind failed");
	return 0 ;
	}
	if (listen(server_fd, 3) < 0) {
	perror("listen");
	return 0 ;
	}
	if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
	perror("accept");
	return 0 ;
	}
	char ok []= "OK";
	char buffer[1024]  ;
	string arrNameHW[2] ;
	string strBuffer;
    
    // OpenCV Var
    // Load names of classes
    string classesFile = "yolov3.txt";
    ifstream ifs(classesFile.c_str());
    string line;
    vector<String> classes;
    while (getline(ifs, line)) classes.push_back(line);
    // Give the configuration and weight files for the model
    String modelConfiguration = "yolov3.cfg";
    String modelWeights = "yolov3.weights";
    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    vector<String> ln = getOutputsNames(net);

    Mat image;

	double start_time, wait_time;
	int  imgSize;
    
  while(true)
	{

	    memset(buffer, '*', 16);
	    start_time = cv::getTickCount();
	    valread = recv(new_socket, buffer , 16 , 0);
	    cout<<"Image name: " << buffer << endl;
	    wait_time = (cv::getTickCount()-start_time)/cv::getTickFrequency();
	    start_time = cv::getTickCount();
	    strBuffer = buffer;
	    int start = 0;
	    int end = strBuffer.find(":");
	    int ii = 0;
	    while (end != -1) {
		    arrNameHW[ii] =  strBuffer.substr(start, end - start);
		    start = end + 1;
		    end = strBuffer.find(":", start);
		    ii++;
	    }
	    arrNameHW[ii] = strBuffer.substr(start, end - start);

	    valread = send(new_socket, ok, strlen(ok), 0);
	    
	    imgSize = std::stoi(arrNameHW[1]); 
	    uchar sockData[imgSize];
	    valread = 0; 

	    //Receive data here
	    for (int i = 0; i < imgSize; i += valread) {
		    if ((valread = recv(new_socket, sockData +i, imgSize  - i, 0)) == -1)  {
			    return 0;
		    }
	    }

	    valread = send(new_socket, ok, strlen(ok), 0);
	    
	    cout << "Wait " << wait_time << " s - Rec " << arrNameHW[0] << " - " << ((cv::getTickCount()-start_time)/cv::getTickFrequency()) << " s - " ;
	    detect(net, ln ,  sockData, image, arrNameHW[0], imgSize);

	}
    return 0;
}
