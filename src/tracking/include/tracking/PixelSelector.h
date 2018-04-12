#ifndef INCLUDES_PIXELSELECTOR_H_
#define INCLUDES_PIXELSELECTOR_H_

//#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

#include "ImageProcessing.h"
#include "LieAlgebra.h"

//const int SSD_size = 5;
//const int line_check_dist = 100;

class PixelSelector {

public:

	PixelSelector();
	Eigen::Matrix<float,2,Eigen::Dynamic> selectPixels(cv::Mat img, cv::Mat grad_mag);
	//Eigen::Matrix<float,2,Eigen::Dynamic> epipolar_line_search(Eigen::Matrix4f T, 
  	//	Eigen::Matrix<float,2,Eigen::Dynamic> prev_pixel_coord, cv::Mat curr, cv::Mat ref, int& features_tracked);

private:

	int region_min_thresh;
	int g_th; // from DSO paper
	int desired_number_of_points;
	int prev_number_of_points;
	int prev_region_size;

};

#endif /* INCLUDES_PIXELSELECTOR_ */