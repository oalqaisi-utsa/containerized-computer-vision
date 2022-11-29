/*
 * Osamah Alqaisi
 * UTSA
 * 
 * This code use pre-trained HaarCascade file  
 * to detect cars from images
 * Images recived by network on port# 8888
 * 
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <time.h>
#include <sstream>

using namespace std;
using namespace cv;
#define PORT 8888

void detect (cv::CascadeClassifier face_cascade, uchar sockData[], cv::Mat image, string out_image , int imgSize )
{
	double start_time = cv::getTickCount();
	cv::Mat greyimg;
	int num_detect = 0;
	image = cv::imdecode(cv::Mat(1, imgSize, CV_8UC1, sockData),  IMREAD_UNCHANGED);

	
	resize(image, image, Size(600, 400));
	cv::cvtColor(image, greyimg, COLOR_BGR2GRAY);
	
	// Detect faces
	vector<Rect> faces;
	//face_cascade.detectMultiScale( greyimg, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
	face_cascade.detectMultiScale( greyimg, faces, 1.1, 2);
	if(faces.size() > 0)
	{
		// Draw circles on the detected faces
		for( int i = 0; i < faces.size(); i++ )
		{
			rectangle( image, Point(faces[i].x, faces[i].y) ,Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ),2,8,0 );
			num_detect++;
		}
		imwrite( out_image, image );
	}
	
	cout << "Process " << ((cv::getTickCount()-start_time)/cv::getTickFrequency()) << " - " << num_detect << endl;
}

int main(int argc, char const* argv[])
{
	// OpenCV Var
	// Load Face cascade (.xml file)
	cv::CascadeClassifier face_cascade;
	face_cascade.load("cars.xml");
	if(face_cascade.empty())
	{
	    cerr<<"Error Loading XML file"<<endl;
	    return 0;
	}

	cv::Mat image;
	double start_time, wait_time;
	int  imgSize;

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

	// Forcefully attaching socket to the port 8888
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
	char buffer[1024];
	string arrNameHW[2] ;
	string strBuffer;
    while(true)
	{
		memset(buffer, '*', 16);
		start_time = cv::getTickCount();
		//cout << "new: " << buffer <<endl;
		valread = recv(new_socket, buffer , 1024 , 0);
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
		detect( face_cascade,  sockData, image, arrNameHW[0] , imgSize);
	}
    return 0;
}

