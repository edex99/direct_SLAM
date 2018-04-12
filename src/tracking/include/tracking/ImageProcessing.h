#ifndef INCLUDES_IMAGEPROCESSING_H_
#define INCLUDES_IMAGEPROCESSING_H_

//#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>
#include "LieAlgebra.h"

const int blur_kernel_size = 15;

// Freigburg1
//	static constexpr float fx = 517.3; ///< focal length in x direction in pixel
//	static constexpr float fy = 516.5; ///< focal length in y direction in pixel
//	static constexpr float cx = 318.6; ///< center point in x direction in pixel
//	static constexpr float cy = 255.3; ///< center point in y direction in pixel

//	Freiburg3
static constexpr float fx = 535.4; ///< focal length in x direction in pixel
static constexpr float fy = 539.2; ///< focal length in y direction in pixel
static constexpr float cx = 320.1; ///< center point in x direction in pixel
static constexpr float cy = 247.6; ///< center point in y direction in pixel

const Eigen::Matrix3f K((Eigen::Matrix3f() << fx, 0, cx, 0, fy, cy, 0, 0, 1).finished());
const Eigen::Matrix3f invK((Eigen::Matrix3f() << 1/fx, 0, -cx/fx, 0, 1/fy, -cy/fy, 0, 0, 1).finished());


static constexpr float ifx = 1.0/fx;
static constexpr float ify = 1.0/fy;
static constexpr float depthScale = 5000.0; ///< Depth scale factor for the asus xtion pro live sensor
static constexpr float idepthScale = 1.0/depthScale;

std::vector<cv::Mat> createImagePyramid(cv::Mat img, uint num_levels);

std::vector<cv::Mat> createDepthPyramid(cv::Mat depth, uint num_levels);

void calcImageDerivatives(cv::Mat gray, cv::Mat& grad_x, cv::Mat& grad_y, cv::Mat& grad_mag);

Eigen::VectorXf getGradMag(cv::Mat grad_mag, Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coord);

float bilinearInterpolation(uint8_t q11, uint8_t q12, uint8_t q21, uint8_t q22, 
	float x1, float x2, float y1, float y2, float x, float y);

Eigen::Matrix<float,3,Eigen::Dynamic> calc3DPoints(cv::Mat depth, 
	Eigen::MatrixXf pixel_coord, Eigen::Matrix<float,2,Eigen::Dynamic>& pixel_coord_new, Eigen::Matrix3f K_multiplier);

Eigen::Matrix<float,3,Eigen::Dynamic> calc3DPoints2(cv::Mat depth, 
	Eigen::MatrixXf pixel_coord, Eigen::Matrix3f K_multiplier);

Eigen::VectorXf refValues(cv::Mat img, Eigen::MatrixXf pixel_coord);

Eigen::Matrix<float,3,Eigen::Dynamic> transform3Dpoints(Eigen::Matrix<float,3,Eigen::Dynamic> Pts, 
	Eigen::Matrix4f T);

Eigen::Matrix<float,2,Eigen::Dynamic> project3Dpoints(Eigen::Matrix<float,3,Eigen::Dynamic> Pts, 
	Eigen::Matrix3f K_multiplier, Eigen::VectorXf& w, int row, int col);

Eigen::VectorXf warpedValues(cv::Mat img, Eigen::MatrixXf pixel_coord_float, Eigen::VectorXf valid);

#endif /* INCLUDES_IMAGEPROCESSING_ */