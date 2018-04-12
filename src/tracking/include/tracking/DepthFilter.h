#ifndef INCLUDES_DEPTHFILTER_H_
#define INCLUDES_DEPTHFILTER_H_

//#include "opencv2/opencv.hpp"

#include "ImageProcessing.h"

#include <boost/shared_ptr.hpp>
//#include <boost/make_shared.hpp>

/*#include <cmath>
#include <Eigen/Dense>
#include "opencv2/core.hpp"
#include "PointCloudMap.h"
#include "PixelSelector.h"
#include "LieAlgebra.h"*/

static constexpr float inv_sqrt_2pi = 0.3989422804014327;
static constexpr float depth_std_dev_factor = 0.00258;
static constexpr float min_depth = 0.5;
static constexpr float depth_range = 1.0/min_depth;

const int SSD_size = 5;
const int line_check_dist = 100;

class DepthFilter {
public:

	DepthFilter();
	
	void initializeFilter(cv::Mat depth, Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coord);

	void updateFilter(Eigen::Matrix4f T, 
		Eigen::Matrix<float,2,Eigen::Dynamic> p1, cv::Mat curr_depth, cv::Mat ref, cv::Mat curr);

	int getFeaturesTracked();

private:

	Eigen::Vector2f epipolarMatch(Eigen::Vector3f p_h, Eigen::Matrix3f F, 
		cv::Mat curr, cv::Mat ref_patch, bool& match);

	Eigen::Matrix<float,2,Eigen::Dynamic> epipolarLineSearch(Eigen::Matrix4f T, 
  		Eigen::Matrix<float,2,Eigen::Dynamic> prev_pixel_coord, cv::Mat curr, cv::Mat ref, Eigen::VectorXi& matched);

	Eigen::Matrix<float,3,Eigen::Dynamic> triangulatePoints(Eigen::Matrix4f T1, Eigen::Matrix4f T2, 
		Eigen::Matrix<float,2,Eigen::Dynamic> p1, Eigen::Matrix<float,2,Eigen::Dynamic> p2, cv::Mat curr_depth);

	float computeTau(Eigen::Matrix4f T, Eigen::Vector3f p);

	//Eigen::Matrix<float,4,Eigen::Dynamic> filter;
	boost::shared_ptr<Eigen::Matrix<float,4,Eigen::Dynamic> > filter;
	int features_tracked;

};


#endif /* INCLUDES_DEPTHFILTER_ */