#include "FeatureDetector.h"

FeatureDetector::FeatureDetector(int max_features) {
	// arguments: max features, scale factor, num levels, 
	//    edge threshold, first level, descriptor values, score type, patch size
	orb = cv::ORB::create(max_features,1.0f,1,patch_size,0,2,cv::ORB::FAST_SCORE,patch_size);

}

void FeatureDetector::detectFeatures(std::vector<cv::Mat> pyr) {

	int grid_rows = (pyr[0].rows)/grid_size_row;
	int grid_cols = pyr[0].cols/grid_size_col;
	for (int i=0; i<pyr.size(); i++) {
		cv::Mat img = pyr[i];
		if (img.cols < 480) {
			continue;
		}
		for (int j=1; j<grid_rows-1; j++) {
			for (int k=1; k<grid_cols-1; k++) {

				std::cout << k*grid_size_col-patch_size << " " << j*grid_size_row-patch_size << std::endl;
				std::cout << grid_size_col+2*patch_size << " " << grid_size_row+2*patch_size << std::endl;
				cv::Mat patch = img(cv::Rect(k*grid_size_col-patch_size,j*grid_size_row-patch_size,
					grid_size_col+2*patch_size, grid_size_row+2*patch_size));

				orb->detect(patch,keypoints);
				cv::Mat temp;
				cv::drawKeypoints(patch, keypoints, temp);
				cv::imshow("w", temp);
				cv::waitKey();
				
			}
		}

		//cv::Mat img2 = img(cv::Rect(350,250,120,120));
		orb->detect(img,keypoints);
		std::cout << keypoints.size() << std::endl;

		for (int i=0; i<keypoints.size();i++) {
			std::cout << keypoints[i].pt << " " << keypoints[i].response << " " << keypoints[i].size << std::endl;
		}

	}
}