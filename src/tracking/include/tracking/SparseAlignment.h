#ifndef INCLUDES_SPARSEALIGNMENT_H_
#define INCLUDES_SPARSEALIGNMENT_H_

#include "LieAlgebra.h"
#include "ImageProcessing.h"
#include "PixelSelector.h"

#include <cmath>
#include <Eigen/Dense>
#include "opencv2/core.hpp"
//#include "opencv2/opencv.hpp"

#include "PointCloudMap.h"

const float Huber_constant = 1.345;
const float Tukey_constant = 4.6851;

/*const Eigen::Matrix<char,2,8> pattern(((Eigen::Matrix<char,2,8>) << -2,-1,-1,0,0,0,1,2,
																	0,1,-1,2,0,-2,1,0).finished());*/
//const short pattern_data[] = {0,0, 1,-1, 2,0, 1,1, 0,2, -1,1, -2,0, -1,-1};
//Eigen::Matrix<short,2,8> pattern(pattern_data);

inline float huber_weight(float residual);
inline float tukey_weight(float residual);

class ImageAlignment {

public:

	ImageAlignment(int num_levels, int gradient_thresh);
	void setKeyframe(std::vector<cv::Mat>& image_pyr, std::vector<cv::Mat>& depth_pyr);

	Eigen::Matrix<float,8,1> directTransformationEstimation(
		std::vector<cv::Mat> pyr_img_curr,
		cv::Mat& out, Eigen::Matrix<float,8,1> theta_init);

private:


	Eigen::VectorXf robust_residuals(Eigen::VectorXf residuals, Eigen::VectorXf valid_pts);

	Eigen::Matrix<float,2,6> dw_dtheta(Eigen::Vector3f P, Eigen::Matrix3f K_multiplier);

	Eigen::Matrix<float,Eigen::Dynamic,6> calc_Jacobian(Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coord, 
		Eigen::Matrix<float,3,Eigen::Dynamic> Pts, cv::Mat& grad_x, cv::Mat& grad_y, Eigen::Matrix3f K_multiplier);

	Eigen::Matrix<float,8,1> optimize(Eigen::Matrix<float,8,1> theta_init, cv::Mat& ref, cv::Mat& curr, cv::Mat& depth_ref,
 		cv::Mat& ref_grad_x, cv::Mat& ref_grad_y, Eigen::MatrixXf pixel_coord, Eigen::Matrix3f K_multiplier, 
 		cv::Mat& ref_grad_mag);

	Eigen::Matrix<float,8,1> optimize2(Eigen::Matrix<float,8,1> theta_init, cv::Mat& curr,int scale);

	const int desired_num_points = 1000;
	const int num_points_thresh = 100;
	int gradient_threshold;

	std::vector<cv::Mat> keyframe_images;
	std::vector<cv::Mat> keyframe_depths;

	std::vector<boost::shared_ptr<Eigen::Matrix<float,3,Eigen::Dynamic> > > keyframe_pts;
	std::vector<boost::shared_ptr<Eigen::Matrix<float,Eigen::Dynamic,8> > > keyframe_jacobians;
	std::vector<boost::shared_ptr<Eigen::VectorXf> > keyframe_ref_vals;
	std::vector<Eigen::Matrix3f> K_scales;

};

#endif /* INCLUDES_SPARSEALIGNMENT_ */