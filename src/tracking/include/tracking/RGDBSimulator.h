 /**
 * @file AsusProLive.h
 * @author Gian Danuser & Michael Eugster
 * @brief This file contains the declaration of the RGDBSimulator class
 *
 */

#ifndef KINECTV1SIM_H_
#define KINECTV1SIM_H_

#include <vector>
#include <fstream>

#include "Eigen/Core"

#include "IRGBDSensor.h"

/**
 * @class RGDBSimulator RGDBSimulator.h "RGDBSimulator.h"
 * @brief The RGDBSimulator class to run datasets
 */
class RGDBSimulator : public IRGBDSensor
{
public:
	/**
	 * @brief Constructor
	 * @brief folder needs to contain a rgbd.txt with the used image path and names
	 *
	 * @param aSimDataPath path where the data are stored (in)
	 */
	RGDBSimulator(std::string aSimDataPath);

	/**
	 * @brief Destructor
	 */
	virtual ~RGDBSimulator();

	//
	// see IRGBDSensor.h for descriptions
	virtual bool grab(cv::Ptr<cv::Mat>& rgbImage, cv::Ptr<cv::Mat>& grayImage, cv::Ptr<cv::Mat>& depthImage, cv::Ptr<double>& timeStamp);
	virtual bool start();
	virtual void stop();

private:
	const std::string simDataPath;
	const std::string imgList;
	std::ifstream list;
};

#endif /* KINECTV1SIM_H_ */
