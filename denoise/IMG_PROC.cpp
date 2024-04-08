/*
 *  @(#)$Id: IMG_PROC.cpp $
 *  @(#) Implementation file of class file
 *
 *  Copyright @ 2015 Shanghai University
 *  All Rights Reserved.
 */
/*  @file   
 *  Implementation of IMG_PROC
 * 
 *  TO DESC : FILE DETAIL DESCRIPTION, BUT DON'T DESCRIBE CLASS DETAIL HERE.
 */
#include <iostream>
#include <string>
#include <vector>
#include <stack>  
#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include "IMG_PROC.h"

using namespace cv; 
using namespace std;
/*************************************************************************

    show_src

    [Description]

        TO DESC
//读取图像
//0:读取图像成功
//1：读取图像失败（不属于图像格式范畴内）

//Windows位图文件 - BMP, DIB；
//JPEG文件 - JPEG, JPG, JPE；
//便携式网络图片 - PNG；
//便携式图像格式 - PBM，PGM，PPM；
//Sun rasters - SR，RAS；
//TIFF文件 - TIFF，TIF;
//OpenEXR HDR 图片 - EXR;
//JPEG 2000 图片- jp2。
*************************************************************************/
int IMG_PROC::show_src(Mat image)
{
    src = image.clone();
	if (!src.data) {
		return 1;
	}
	return 0;
}

/**
 * @brief removeIcon 
 * 消除白色的图例
 * @param {Mat} src
 */
Mat IMG_PROC::removeIcon(Mat src)
{
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<Vec3b>(j, i) == Vec3b(255, 255, 255)) {
				;
			}
		}
	}
	return Mat();
}


/*************************************************************************

    avgRGB

    [Description]

        TO DESC

*************************************************************************/
void IMG_PROC::avgRGB()
{
	int cPointR,cPointG,cPointB;
	avg_R=0,avg_G=0,avg_B=0;                                 
	for(int i=0;i<src.rows;i++) 
	{
		for(int j=0;j<src.cols;j++)  
		{
			cPointB=src.at<Vec3b>(i,j)[0];  
            cPointG=src.at<Vec3b>(i,j)[1];  
            cPointR=src.at<Vec3b>(i,j)[2]; 
			avg_R=cPointR+avg_R;
			avg_G=cPointG+avg_G;
			avg_B=cPointB+avg_B;
		}
	}
	avg_R=avg_R/(src.rows*src.cols);
	avg_G=avg_G/(src.rows*src.cols);
	avg_B=avg_B/(src.rows*src.cols);
}

/*************************************************************************

    remove_areanoise

    [Description]

        TO DESC
//去白斑
//temp1:存储临时图像
//temp:存储去白斑之后的图像
*************************************************************************/

void IMG_PROC::remove_areanoise()                            
{  
	temp1=src.clone();
	std::stack<std::pair<int,int> > neighborPixels;  
	int cPointR,cPointG,cPointB;
	int t_buf = 100;
	//int count = 1;
	for (int i=1; i < src.rows-1; i++)   
	{  
		for (int j = 1; j < src.cols-1; j++)  
		{  	
			cPointB = src.at<Vec3b>(i,j)[0];  
			cPointG = src.at<Vec3b>(i,j)[1];  
			cPointR = src.at<Vec3b>(i,j)[2];   
			//该像素点的R\G\B值       
			if(src.at<Vec3b>(i, j-1)[0] == cPointB&&src.at<Vec3b>(i, j-1)[1] == cPointG&&src.at<Vec3b>(i, j-1)[2] == cPointR&&
			   src.at<Vec3b>(i, j+1)[0] == cPointB&&src.at<Vec3b>(i, j+1)[1] == cPointG&&src.at<Vec3b>(i, j+1)[2] == cPointR&&
			   src.at<Vec3b>(i-1, j)[0] == cPointB&&src.at<Vec3b>(i-1, j)[1] == cPointG&&src.at<Vec3b>(i-1, j)[2] == cPointR&&
			   src.at<Vec3b>(i+1, j)[0] == cPointB&&src.at<Vec3b>(i+1, j)[1] == cPointG&&src.at<Vec3b>(i+1, j)[2] == cPointR&&
			   src.at<Vec3b>(i-1, j-1)[0] == cPointB&&src.at<Vec3b>(i-1, j-1)[1] == cPointG&&src.at<Vec3b>(i-1, j-1)[2] == cPointR&&
			   src.at<Vec3b>(i-1, j+1)[0] == cPointB&&src.at<Vec3b>(i-1, j+1)[1] == cPointG&&src.at<Vec3b>(i-1, j+1)[2] == cPointR&&
			   src.at<Vec3b>(i+1, j-1)[0] == cPointB&&src.at<Vec3b>(i+1, j-1)[1] == cPointG&&src.at<Vec3b>(i+1, j-1)[2] == cPointR&&
			   src.at<Vec3b>(i+1, j+1)[0] == cPointB&&src.at<Vec3b>(i+1, j+1)[1] == cPointG&&src.at<Vec3b>(i+1, j+1)[2] == cPointR &&
			   cPointB > t_buf && cPointG > t_buf && cPointR > t_buf) {
					//cout<<count<<endl;
					//count++;
					//如果3*3范围像素点的RGB值相同，压入栈内
					neighborPixels.push(std::pair<int,int>(i,j)) ;
					while (!neighborPixels.empty())  
					{ 
						//如果栈不为空
						std::pair<int,int> curPixel = neighborPixels.top() ;  
						//取栈顶元素
						int curX = curPixel.first ;  
						//该点x坐标
						int curY = curPixel.second ;
						//该点y坐标  
						temp1.at<Vec3b>(curX,curY)[0] = 0;
						temp1.at<Vec3b>(curX,curY)[1] = 0;
						temp1.at<Vec3b>(curX,curY)[2] = 0;
						//该店RGB值变为空
						neighborPixels.pop();  
						//弹出
						if (curY-1>=0)
						if (!(temp1.at<Vec3b>(curX,curY-1)[0]==0&&
									  temp1.at<Vec3b>(curX,curY-1)[1]==0&&
									  temp1.at<Vec3b>(curX,curY-1)[2]==0))
						if (src.at<Vec3b>(curX, curY-1)[0] == cPointB
									&&src.at<Vec3b>(curX, curY-1)[1] == cPointG
									&&src.at<Vec3b>(curX, curY-1)[2] == cPointR
						   )  {
							neighborPixels.push(std::pair<int,int>(curX, curY-1)) ;  
						}
						if (curY+1<src.cols-1)
						  if (!(temp1.at<Vec3b>(curX,curY+1)[0]==0&&
										  temp1.at<Vec3b>(curX,curY+1)[1]==0&&
										  temp1.at<Vec3b>(curX,curY+1)[2]==0))
							if (src.at<Vec3b>(curX, curY+1)[0] == cPointB
										&&src.at<Vec3b>(curX, curY+1)[1] == cPointG
										&&src.at<Vec3b>(curX, curY+1)[2] == cPointR)  
							{  
								neighborPixels.push(std::pair<int,int>(curX, curY+1)) ;  
							}  
						if (curX-1>=0)
						  if (!(temp1.at<Vec3b>(curX-1,curY)[0]==0&&
										  temp1.at<Vec3b>(curX-1,curY)[1]==0&&
										  temp1.at<Vec3b>(curX-1,curY)[2]==0))
							if (src.at<Vec3b>(curX-1, curY)[0] == cPointB
										&&src.at<Vec3b>(curX-1, curY)[1] == cPointG
										&&src.at<Vec3b>(curX-1, curY)[2] == cPointR)  
							{
								neighborPixels.push(std::pair<int,int>(curX-1, curY)) ;
							}  
						if (curX+1<src.rows-1)
						  if (!(temp1.at<Vec3b>(curX+1,curY)[0]==0&&
										  temp1.at<Vec3b>(curX+1,curY)[1]==0&&
										  temp1.at<Vec3b>(curX+1,curY)[2]==0))
							if (src.at<Vec3b>(curX+1, curY)[0] == cPointB
										&&src.at<Vec3b>(curX+1, curY)[1] == cPointG
										&&src.at<Vec3b>(curX+1, curY)[2] == cPointR) 
							{
								neighborPixels.push(std::pair<int,int>(curX+1, curY)) ;  
							} 
					}  
			}
		}  
	} 
	//imshow("temp1", temp1);
	temp=temp1.clone();
	//填充背景均值
	for (int i=1; i < temp1.rows-1; i++)   
	{  
		for (int j=1; j < temp1.cols-1; j++)  
		{
			if (temp1.at<Vec3b>(i,j)[0]==0 && temp1.at<Vec3b>(i,j)[1]==0 && temp1.at<Vec3b>(i,j)[2]==0)
			{
				for (int k = i - 6; k <= i + 6; k++)
				  for (int l = j - 6; l <= j + 6; l++)
				  {
					  if (k>=0 && k<temp.rows && l>=0 && l<temp.cols)
					  {
						  temp.at<Vec3b>(k,l)[0]=(uchar)avg_B;
						  temp.at<Vec3b>(k,l)[1]=(uchar)avg_G;
						  temp.at<Vec3b>(k,l)[2]=(uchar)avg_R;
					  }
				  }

			}
		}
	}
	//file_in其实是输出路径
	imwrite("removed_area.png", temp);
}


//void IMG_PROC::remove_areanoise(string filename,string file_in)                            
//{  
	//temp1=src.clone();
	//std::stack<std::pair<int,int> > neighborPixels ;  
	//int cPointR,cPointG,cPointB;
	//int aB, aG, aR;
	//int num_window = 9;
	//int area_thread = 0.001;
	//int variance;
	//for (int i=1; i < src.rows-1; i++)   
	//{  
		
		//for (int j = 1; j < src.cols-1; j++)  
		//{  	
			
			//cPointB=src.at<Vec3b>(i,j)[0];  
			//cPointG=src.at<Vec3b>(i,j)[1];  
			//cPointR=src.at<Vec3b>(i,j)[2];

			//aB =(src.at<Vec3b>(i  , j-1)[0] + src.at<Vec3b>(i  , j+1)[0] +  src.at<Vec3b>(i  , j)[0] +
				 //src.at<Vec3b>(i-1, j-1)[0] + src.at<Vec3b>(i-1, j+1)[0] +  src.at<Vec3b>(i-1, j)[0] +
				 //src.at<Vec3b>(i+1, j-1)[0] + src.at<Vec3b>(i+1, j+1)[0] +  src.at<Vec3b>(i+1, j)[0]) / num_window;
			//aG =(src.at<Vec3b>(i  , j-1)[1] + src.at<Vec3b>(i  , j+1)[1] +  src.at<Vec3b>(i  , j)[1] +
				 //src.at<Vec3b>(i-1, j-1)[1] + src.at<Vec3b>(i-1, j+1)[1] +  src.at<Vec3b>(i-1, j)[1] +
				 //src.at<Vec3b>(i+1, j-1)[1] + src.at<Vec3b>(i+1, j+1)[1] +  src.at<Vec3b>(i+1, j)[1]) / num_window;
			//aR =(src.at<Vec3b>(i  , j-1)[2] + src.at<Vec3b>(i  , j+1)[2] +  src.at<Vec3b>(i  , j)[2] +
				 //src.at<Vec3b>(i-1, j-1)[2] + src.at<Vec3b>(i-1, j+1)[2] +  src.at<Vec3b>(i-1, j)[2] +
				 //src.at<Vec3b>(i+1, j-1)[2] + src.at<Vec3b>(i+1, j+1)[2] +  src.at<Vec3b>(i+1, j)[2]) / num_window;

			//variance  =  abs(src.at<Vec3b>(i  , j-1)[0] - aB) + abs(src.at<Vec3b>(i  , j+1)[0] - aB) +  abs(src.at<Vec3b>(i  , j)[0] - aB) +
						 //abs(src.at<Vec3b>(i+1, j-1)[0] - aB) + abs(src.at<Vec3b>(i+1, j+1)[0] - aB) +  abs(src.at<Vec3b>(i+1, j)[0] - aB) +
						 //abs(src.at<Vec3b>(i-1, j-1)[0] - aB) + abs(src.at<Vec3b>(i-1, j+1)[0] - aB) +  abs(src.at<Vec3b>(i-1, j)[0] - aB) +
						 //abs(src.at<Vec3b>(i  , j-1)[1] - aG) + abs(src.at<Vec3b>(i  , j+1)[1] - aG) +  abs(src.at<Vec3b>(i  , j)[1] - aG) +
						 //abs(src.at<Vec3b>(i+1, j-1)[1] - aG) + abs(src.at<Vec3b>(i+1, j+1)[1] - aG) +  abs(src.at<Vec3b>(i+1, j)[1] - aG) +
						 //abs(src.at<Vec3b>(i-1, j-1)[1] - aG) + abs(src.at<Vec3b>(i-1, j+1)[1] - aG) +  abs(src.at<Vec3b>(i-1, j)[1] - aG) +
						 //abs(src.at<Vec3b>(i  , j-1)[2] - aR) + abs(src.at<Vec3b>(i  , j+1)[2] - aR) +  abs(src.at<Vec3b>(i  , j)[2] - aR) +
						 //abs(src.at<Vec3b>(i+1, j-1)[2] - aR) + abs(src.at<Vec3b>(i+1, j+1)[2] - aR) +  abs(src.at<Vec3b>(i+1, j)[2] - aR) +
						 //abs(src.at<Vec3b>(i-1, j-1)[2] - aR) + abs(src.at<Vec3b>(i-1, j+1)[2] - aR) +  abs(src.at<Vec3b>(i-1, j)[2] - aR);

			////cout<<"i:"<<i<<" j:"<<j<<"variance:"<<variance<<endl;
			////该像素点的R\G\B值       
			////cout<<"variance:"<<variance<<endl;
			//if(variance < area_thread * num_window)
			//{
			////如果3*3范围像素点的RGB值相同，压入栈内
				//neighborPixels.push(std::pair<int,int>(i,j)) ;
				//while (!neighborPixels.empty())  
				//{ 
			////如果栈不为空
					//std::pair<int,int> curPixel = neighborPixels.top() ;  
			////取栈顶元素
					//int curX = curPixel.first ;  
			////该点x坐标
					//int curY = curPixel.second ;
			////该点y坐标  
					//temp1.at<Vec3b>(curX,curY)[0]=0;
					//temp1.at<Vec3b>(curX,curY)[1]=0;
					//temp1.at<Vec3b>(curX,curY)[2]=0;
			////该店RGB值变为空
					//neighborPixels.pop();  
			////弹出
					//if(curY-1>=0)
						//if(!(temp1.at<Vec3b>(curX,curY-1)[0]==0&&
							 //temp1.at<Vec3b>(curX,curY-1)[1]==0&&
							 //temp1.at<Vec3b>(curX,curY-1)[2]==0))
							 //if (src.at<Vec3b>(curX, curY-1)[0] == cPointB
							   //&&src.at<Vec3b>(curX, curY-1)[1] == cPointG
							   //&&src.at<Vec3b>(curX, curY-1)[2] == cPointR
							   //)  
							 //{
								 //neighborPixels.push(std::pair<int,int>(curX, curY-1)) ;  
							 //}
					//if(curY+1<src.cols-1)
						//if(!(temp1.at<Vec3b>(curX,curY+1)[0]==0&&
							 //temp1.at<Vec3b>(curX,curY+1)[1]==0&&
							 //temp1.at<Vec3b>(curX,curY+1)[2]==0))
							 //if (src.at<Vec3b>(curX, curY+1)[0] == cPointB
							   //&&src.at<Vec3b>(curX, curY+1)[1] == cPointG
							   //&&src.at<Vec3b>(curX, curY+1)[2] == cPointR)  
							  //{  
								  //neighborPixels.push(std::pair<int,int>(curX, curY+1)) ;  
							  //}  
					//if(curX-1>=0)
						//if(!(temp1.at<Vec3b>(curX-1,curY)[0]==0&&
							 //temp1.at<Vec3b>(curX-1,curY)[1]==0&&
							   //temp1.at<Vec3b>(curX-1,curY)[2]==0))
							 //if (src.at<Vec3b>(curX-1, curY)[0] == cPointB
							   //&&src.at<Vec3b>(curX-1, curY)[1] == cPointG
							   //&&src.at<Vec3b>(curX-1, curY)[2] == cPointR)  
							 //{
								 //neighborPixels.push(std::pair<int,int>(curX-1, curY)) ;
							 //}  
				   //if(curX+1<src.rows-1)
					   //if(!(temp1.at<Vec3b>(curX+1,curY)[0]==0&&
							//temp1.at<Vec3b>(curX+1,curY)[1]==0&&
							//temp1.at<Vec3b>(curX+1,curY)[2]==0))
							//if (src.at<Vec3b>(curX+1, curY)[0] == cPointB
							  //&&src.at<Vec3b>(curX+1, curY)[1] == cPointG
							  //&&src.at<Vec3b>(curX+1, curY)[2] == cPointR) 
							//{
								//neighborPixels.push(std::pair<int,int>(curX+1, curY)) ;  
							//} 
				//}  
			//}
		 //}  
	 //}  
	//imshow("temp1", temp1);
	//temp=temp1.clone();
	////填充背景均值
	 //for (int i=1; i < temp1.rows-1; i++)   
	 //{  
		 //for (int j=1; j < temp1.cols-1; j++)  
		 //{
			 //if(temp1.at<Vec3b>(i,j)[0]==0&&
				//temp1.at<Vec3b>(i,j)[1]==0&&
				//temp1.at<Vec3b>(i,j)[2]==0)
			 //{
				 //for(int k=i-6;k<=i+6;k++)
					 //for(int l=j-6;l<=j+6;l++)
					 //{
						 //if(k>=0&&k<temp.rows&&l>=0&&l<temp.cols)
						 //{
							 //temp.at<Vec3b>(k,l)[0]=(uchar)avg_B;
							 //temp.at<Vec3b>(k,l)[1]=(uchar)avg_G;
							 //temp.at<Vec3b>(k,l)[2]=(uchar)avg_R;
						 //}
					 //}

			 //}
		 //}
	 //}
	 ////file_in其实是输出路径
	 //imwrite(file_in+"\\area"+filename,temp);
//}

void IMG_PROC::remove_areanoise_Tong(string outfile)
{
    //src = imread("image_5.png");
    src = imread(outfile.c_str());
	temp = src.clone();
	for (int i=1; i < src.rows-1; i++)   
	{  
		for (int j=1; j < src.cols-1; j++)  
		{
			if (src.at<Vec3b>(i,j)[0]==0 && src.at<Vec3b>(i,j)[1]==0 && src.at<Vec3b>(i,j)[2]==0)
			{
				for (int k = i - 4; k <= i + 4; k++)
				  for (int l = j - 4; l <= j + 4; l++)
				  {
					  if (k>=0 && k<temp.rows && l>=0 && l<temp.cols)
					  {
						  temp.at<Vec3b>(k,l)[0]=(uchar)avg_B;
						  temp.at<Vec3b>(k,l)[1]=(uchar)avg_G;
						  temp.at<Vec3b>(k,l)[2]=(uchar)avg_R;
					  }
				  }
			}
		}
	}
	//file_in其实是输出路径
    //imshow("removed_area", temp);
	imwrite("removed_area.png",temp);
}

/*************************************************************************

    remove_watermark

    [Description]

        TO DESC

*************************************************************************/
void IMG_PROC::remove_watermark()                           
{                                                                
	int cPointR,cPointG,cPointB;
	for(int i=0;i<src.rows;i++)  
	{
        for(int j=0;j<src.cols;j++)  
        {  
			cPointB=temp.at<Vec3b>(i,j)[0];  
            cPointG=temp.at<Vec3b>(i,j)[1];  
            cPointR=temp.at<Vec3b>(i,j)[2];  
			if(cPointB+cPointR+cPointG>600)  
			{  			
				temp.at<Vec3b>(i,j)[0]=(uchar)avg_B;  
				temp.at<Vec3b>(i,j)[1]=(uchar)avg_G;  
				temp.at<Vec3b>(i,j)[2]=(uchar)avg_R;		
            }
		}  
	}
	imwrite("removed_watered.png", temp);
}

/*************************************************************************

    get_binarysrc

    [Description]

        TO DESC
//OTSU:设使用某一个阈值将灰度图像根据灰度大小，分成目标部分和背景部分两类，
//在这两类的类内方差最小和类间方差最大的时候，得到的阈值是最优的二值化阈值。
*************************************************************************/
void IMG_PROC::get_binarysrc()
{
	cvtColor(temp, temp1, CV_BGR2GRAY);                           
	threshold(temp1, binarysrc, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU); 
	imwrite("binary.png", binarysrc);
}

/*************************************************************************

    new_erode

    [Description]

        TO DESC

*************************************************************************/
void IMG_PROC::new_erode()
{ 
    //使用默认的方形结构元素(等价于以下注释掉的代码)  
    //erode(binarysrc, temp, Mat());  
    Mat element(5, 5, CV_8U, Scalar(1));  
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	erode(binarysrc, temp, element);
	erode(binarysrc, temp, element);
	erode(binarysrc, temp, element);
	erode(binarysrc, temp, element);
	erode(binarysrc, temp, element);
	erode(binarysrc, temp, element);
	//imshow("binarysrc", temp);
	imwrite("eroded_image.png", temp);
}

/*************************************************************************

    new_dilate

    [Description]

        TO DESC

*************************************************************************/
void IMG_PROC::new_dilate()
{
	cvtColor(binarysrc, binarysrc,  CV_GRAY2RGB); 
	cvtColor(temp, temp,  CV_GRAY2RGB); 
	std::stack<std::pair<int,int> > neighborPixels2 ;  
	int cPointR,cPointG,cPointB;
	for (int i=1; i < src.rows-1; i++)   
	{  
		for (int j = 1; j < src.cols-1; j++)  
		{  	
			cPointB=temp.at<Vec3b>(i,j)[0];  
			cPointG=temp.at<Vec3b>(i,j)[1];  
			cPointR=temp.at<Vec3b>(i,j)[2];          
			if(cPointB==255&&cPointG==255&&cPointR==255)
			{
				neighborPixels2.push(std::pair<int,int>(i,j)) ;
				while (!neighborPixels2.empty())  
				{ 
                    std::pair<int,int> curPixel = neighborPixels2.top() ;  
                    int curX = curPixel.first ;  
                    int curY = curPixel.second ;  
					temp.at<Vec3b>(curX,curY)[0]=255;
					temp.at<Vec3b>(curX,curY)[1]=255;
					temp.at<Vec3b>(curX,curY)[2]=255;
                    neighborPixels2.pop();  
					binarysrc.at<Vec3b>(curX,curY)[0]=255;
					binarysrc.at<Vec3b>(curX,curY)[1]=0;
					binarysrc.at<Vec3b>(curX,curY)[2]=0;
					if(curY-1>=0)
						if(binarysrc.at<Vec3b>(curX,curY-1)[0]==255&&
							 binarysrc.at<Vec3b>(curX,curY-1)[1]==255&&
							 binarysrc.at<Vec3b>(curX,curY-1)[2]==255)
							 {
								 neighborPixels2.push(std::pair<int,int>(curX, curY-1)) ;  
							 }
					if(curY+1<src.cols-1)
						if(binarysrc.at<Vec3b>(curX,curY+1)[0]==255&&
							 binarysrc.at<Vec3b>(curX,curY+1)[1]==255&&
							 binarysrc.at<Vec3b>(curX,curY+1)[2]==255)
							 {  
							     neighborPixels2.push(std::pair<int,int>(curX, curY+1)) ;  
						      }  
					if(curX-1>=0)
						if(binarysrc.at<Vec3b>(curX-1,curY)[0]==255&&
							 binarysrc.at<Vec3b>(curX-1,curY)[1]==255&&
							 binarysrc.at<Vec3b>(curX-1,curY)[2]==255)
							 {
								 neighborPixels2.push(std::pair<int,int>(curX-1, curY)) ;  
						      }  
				    if(curX+1<src.rows-1)
					    if(binarysrc.at<Vec3b>(curX+1,curY)[0]==255&&
							 binarysrc.at<Vec3b>(curX+1,curY)[1]==255&&
							 binarysrc.at<Vec3b>(curX+1,curY)[2]==255)								
							{
							     neighborPixels2.push(std::pair<int,int>(curX+1, curY)) ;  
							}  
				}  
			}
		 }  
	 }  
     //imwrite("new_dilate.png",temp);
    imwrite("binary.png",temp);
}

/*************************************************************************

    filling_pic

    [Description]

        TO DESC

*************************************************************************/
Mat IMG_PROC::filling_pic()
{
    temp=imread("binary.png");
	dst=src.clone();
	for(int i=0;i<src.rows;i++) 
	{
		for(int j=0;j<src.cols;j++)  
		{
			if(temp.at<Vec3b>(i,j)[0]==0&&temp.at<Vec3b>(i,j)[1]==0&&temp.at<Vec3b>(i,j)[2]==0)
			{
				dst.at<Vec3b>(i,j)[0]=(uchar)avg_B;
				dst.at<Vec3b>(i,j)[1]=(uchar)avg_G;
				dst.at<Vec3b>(i,j)[2]=(uchar)avg_R;
			}
		}
	}
    imwrite("filled_img.png",dst);
    return dst;
}

/* EOF */
