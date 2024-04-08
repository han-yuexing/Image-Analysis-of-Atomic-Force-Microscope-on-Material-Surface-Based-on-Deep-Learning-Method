/*
 *  @(#)$Id: IMG_PROC.h $
 *  @(#) Declaration file of class IMG_PROC
 *
 *  Copyright @ 2015 Shanghai University
 *  All Rights Reserved.
 */
/*  @file   
 *  Declaration of IMG_PROC
 * 
 *  TO DESC : FILE DETAIL DESCRIPTION, BUT DON'T DESCRIBE CLASS DETAIL HERE.
 */
#ifndef IMG_PROC_H_
#define IMG_PROC_H_
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
// Class declaration
class IMG_PROC;
/*-----------------------------------------------------------------------*/
/**
 *  class IMG_PROC
 *
 * TO DESC : CLASS DETAIL DESCRIPTION
 * 
 * @author $Author$
 * @version $Revision$
 * @date     $Date::                      #$
 */
/*-----------------------------------------------------------------------*/
class IMG_PROC
{
public:
	/*-----------------------------------------------------------------------*/
	/**
	 * show_src//读取原图
	 * 
	 * TO DESC : FUNCTION DESCRIPTION
	 * 
	 * @param string filename:picture name
	 *		  string file_in:picture source folder
	 * 
	 * @return int
	 */
	/*-----------------------------------------------------------------------*/
    int show_src(Mat image);

	Mat removeIcon(Mat src);

	/*-----------------------------------------------------------------------*/
	/**
	 * avgRGB//计算平均像素
	 * 
	 * TO DESC : FUNCTION DESCRIPTION
	 * 
	 * @param NULL
	 * 
	 * @return void
	 */
	/*-----------------------------------------------------------------------*/
	void avgRGB();

	/*-----------------------------------------------------------------------*/
	/**
	 * remove_areanoise//去白斑
	 * 
	 * TO DESC : FUNCTION DESCRIPTION
	 * 
	 * @param NULL
	 * 
	 * @return void
	 */
	/*-----------------------------------------------------------------------*/
	void remove_areanoise();

    void remove_areanoise_Tong(string outfile);

	/*-----------------------------------------------------------------------*/
	/**
	 * remove_watermark//去水印
	 * 
	 * TO DESC : FUNCTION DESCRIPTION
	 * 
	 * @param NULL
	 * 
	 * @return void
	 */
	/*-----------------------------------------------------------------------*/
	void remove_watermark();

	/*-----------------------------------------------------------------------*/
	/**
	 * get_binarysrc//二值化
	 * 
	 * TO DESC : FUNCTION DESCRIPTION
	 * 
	 * @param NULL
	 * 
	 * @return void
	 */
	/*-----------------------------------------------------------------------*/
    void get_binarysrc();

	/*-----------------------------------------------------------------------*/
	/**
	 * new_erode//腐蚀
	 * 
	 * TO DESC : FUNCTION DESCRIPTION
	 * 
	 * @param NULL
	 * 
	 * @return void
	 */
	/*-----------------------------------------------------------------------*/
	void new_erode();

	/*-----------------------------------------------------------------------*/
	/**
	 * new_dilate//膨胀：填充连通域
	 * 
	 * TO DESC : FUNCTION DESCRIPTION
	 * 
	 * @param string filename:picture name
	 *		  string file_in:Binarypicture destination folder
	 * 
	 * @return void
	 */
	/*-----------------------------------------------------------------------*/
	void new_dilate();

	/*-----------------------------------------------------------------------*/
	/**
	 * filling_pic//图像背景填充
	 * 
	 * TO DESC : FUNCTION DESCRIPTION
	 * 
	 * @param string filename:picture name
	 *		  string file_in:Binarypicture source folder
	 *	      string file_out:picture destination folder
	 * 
	 * @return void
	 */
	/*-----------------------------------------------------------------------*/
    Mat filling_pic();

public:
	Mat src,temp,temp1,binarysrc,dst;			//!<src:原始图像； temp：过程临时图像； temp1：临时图像1； binarysrc：二值化图像； dst：目标图像;
	float avg_R,avg_G,avg_B;                    //图像平均像素值
	float runtime;                              //运行时间
};
#endif //IMG_PROC_H_
/* EOF */
