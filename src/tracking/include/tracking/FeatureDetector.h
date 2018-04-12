#ifndef INCLUDES_FEATUREDETECTOR_H_
#define INCLUDES_FEATUREDETECTOR_H_

//#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"


class FeatureDetector {

public:

	FeatureDetector(int max_features);
	void detectFeatures(std::vector<cv::Mat> pyr);


private:

	const int grid_size_row = 40;
	const int grid_size_col = 40;

	const int patch_size = 40;
	cv::Ptr<cv::ORB> orb;

	std::vector<cv::KeyPoint> keypoints;

};

#endif /* INCLUDES_FEATUREDETECTOR_ */