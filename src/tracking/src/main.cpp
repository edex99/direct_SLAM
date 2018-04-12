// #include <iostream>
// #include <stdio.h>
#include <ctime>
 
// //Include OpenCV
// //#include "opencv2/opencv_modules.hpp"
// //#include <opencv2/highgui.hpp>
// //#include <opencv/cv.h>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
//#include "opencv2/xfeatures2d.hpp"
// #include "opencv2/highgui.hpp"
//#include "opencv2/imgcodecs.hpp"

//#include "LieAlgebra.h"
#include "RGDBSimulator.h"
//#include "PixelSelector.h"
#include "ImageProcessing.h"
#include "SparseAlignment.h"
#include "MatToPcConverter.h"
#include "PointCloudMap.h"
#include "DepthFilter.h"
#include "FeatureDetector.h"

#include "DBoW2.h"

int main(void)
{

  const int num_levels = 6;
  const int gradient_threshold = 10;


  ImageAlignment imageAlign(num_levels,gradient_threshold);
  //DepthFilter depthFilter;
  //FeatureDetector fd(5);

  std::clock_t start;
  double duration;

  std::string folder = "/home/dexheimere/MAV-Project/HSR/02_Libs/HAL/simData/rgbd_dataset_freiburg1_desk/";
  //std::string folder = "/home/dexheimere/MAV-Project/HSR/02_Libs/HAL/simData/rgbd_dataset_freiburg3_structure_notexture_far/";
  //std::string folder = "/home/dexheimere/MAV-Project/HSR/02_Libs/HAL/simData/rgbd_dataset_freiburg3_nostructure_texture_far/";
  RGDBSimulator rgbdSensor(folder);
  IRGBDSensor& sensor = rgbdSensor;

  cv::Ptr<cv::Mat> rgbImage;   
  cv::Ptr<cv::Mat> grayImage; 
  cv::Ptr<cv::Mat> depthImage;
  cv::Mat depthImage_ref, rgbImage_ref; 
  cv::Ptr<double> timeStamp;
  std::vector<cv::Mat> img_pyramid_ref, depth_pyramid_ref;
  Eigen::Matrix4f T_cumulative = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_curr = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_keyframe = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_prev = Eigen::Matrix4f::Identity();
  Eigen::Matrix<float,8,1> theta;
  theta << 0,0,0,0,0,0,0,0;
  Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coord_ref;


  PointCloudMap map3d(0.02f);
  map3d.startMapViewer();

  while (!sensor.start())
  {
    sleep(5);
  }

  int iter = 0;
  bool stop = false;
  while(!stop) {
    stop = !sensor.grab(rgbImage, grayImage, depthImage, timeStamp); // 640 x 480 images

    if (iter==0) {
      iter++;
      /*for (int i=0; i<86; i++) {
        stop = !sensor.grab(rgbImage, grayImage, depthImage, timeStamp);
      }*/

      img_pyramid_ref = createImagePyramid(*grayImage, num_levels);
      depthImage_ref = *depthImage;
      depth_pyramid_ref = createDepthPyramid(depthImage_ref, num_levels);
      rgbImage_ref = *rgbImage;
      imageAlign.setKeyframe(img_pyramid_ref, depth_pyramid_ref);
      cv::Mat grad_x, grad_y, grad_mag;
      calcImageDerivatives(img_pyramid_ref[0], grad_x, grad_y, grad_mag);
      //fd.detectFeatures(img_pyramid_ref);
      //pixel_coord_ref = pixelSel.selectPixels(img_pyramid_ref[0],grad_mag);
      //depthFilter.initializeFilter(depthImage_ref, pixel_coord_ref);

      continue;
    }


    // Image Pyramid and Derivatives
    std::vector<cv::Mat> img_pyramid_curr, depth_pyramid_ref;
    img_pyramid_curr = createImagePyramid(*grayImage, num_levels);
    depth_pyramid_ref = createDepthPyramid(depthImage_ref, num_levels);

    //std::cout << "rgb type: " << rgbImage->type() << std::endl; // uint_8t w/ 3 channels
    //std::cout << "gray type: " << grayImage->type() << std::endl; // uint_8t
    //std::cout << "depth type: " << depthImage->type() << std::endl; // uint16_t

    //cv::imshow("curr", *rgbImage);
    //cv::waitKey(3);

    Eigen::Matrix4f T;
    cv::Mat out;
    start = std::clock();
    theta = imageAlign.directTransformationEstimation(img_pyramid_curr, out, theta);
    T = se3_to_SE3(theta.topRows(6));

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<< "odometry time: "<< duration <<'\n';

    // Find Interest Points
    cv::Mat grad_x, grad_y, grad_mag;
    calcImageDerivatives(img_pyramid_ref[0], grad_x, grad_y, grad_mag);

    //pixel_coord_ref = pixelSel.selectPixels(img_pyramid_ref[0],grad_mag);
    cout << "interest points: " << pixel_coord_ref.cols() << endl;
    //depthFilter.updateFilter(T,pixel_coord_ref, *depthImage, img_pyramid_ref[0], out);
    /*cv::Mat temp_ref = img_pyramid_ref[0].clone();
    cv::Mat temp_curr = out.clone();
    for(uint i=0; i<pixel_coord_curr.cols(); i++) {
        cv::Point pt;
        pt.x = pixel_coord_curr(0,i);
        pt.y = pixel_coord_curr(1,i);
        cv::circle(temp_curr, pt ,3,cv::Scalar(0,0,255));
        pt.x = pixel_coord_ref(0,i);
        pt.y = pixel_coord_ref(1,i);
        cv::circle(temp_ref, pt ,3,cv::Scalar(0,0,255));
    }
    std::cout << "tracked points: " << features_tracked << std::endl;
    cv::imshow("img_ref",temp_ref);
    cv::imshow("img_curr",temp_curr);
    cv::waitKey(0);*/

    std::cout << T_curr << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    getColorPC(depthImage_ref, rgbImage_ref, cloud);
    //if (iter%20 == 0)
      map3d.addMapPart(*cloud, T_cumulative);
    map3d.updateMapViewer();


    T_prev = T_cumulative;
    int features_tracked = 0;
    //if (depthFilter.getFeaturesTracked() < 500) {
    if (features_tracked < 500) {
      /*rgbImage_ref = *rgbImage;
      depthImage_ref = *depthImage;
      img_pyramid_ref = img_pyramid_curr;*/
      start = std::clock();
      imageAlign.setKeyframe(img_pyramid_ref, depth_pyramid_ref);
      duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
      std::cout<<"keyframe setup time: "<< duration <<'\n';
      /*cv::Mat grad_x, grad_y, grad_mag;
      calcImageDerivatives(img_pyramid_ref[0], grad_x, grad_y, grad_mag);
      pixel_coord_ref = pixelSel.selectPixels(img_pyramid_ref[0],grad_mag);
      depthFilter.initializeFilter(depthImage_ref, pixel_coord_ref);*/
      T_curr = T.inverse();
      T_cumulative = T_keyframe*T_curr;
      T_keyframe = T_prev;
      theta << 0,0,0,0,0,0,0,0;
      std::cout << "new keyframe" << std::endl;
    }
    else {
      T_curr = T.inverse();
      T_cumulative = T_keyframe*T_curr;
    }

    rgbImage_ref = *rgbImage;
    depthImage_ref = *depthImage;
    img_pyramid_ref = img_pyramid_curr;

    std::cout << "iter: " << iter << std::endl;
    iter++;

  }


}