#ifndef INCLUDES_MATTOPCCONVERTER_H_
#define INCLUDES_MATTOPCCONVERTER_H_

#include "pcl/point_types.h"
#include "pcl/point_cloud.h"
#include "ImageProcessing.h"

void getColorPC(cv::Mat depthIm, cv::Mat rgbIm, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcOut);

#endif /* INCLUDES_MATTOPCCONVERTER_ */