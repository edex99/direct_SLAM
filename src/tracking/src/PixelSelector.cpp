#include "PixelSelector.h"

double median_mat( cv::Mat channel ) {
  double m = (channel.rows*channel.cols) / 2;
  int bin = 0;
  double med = -1.0;

  int histSize = 256;
  float range[] = { 0, 256 };
  const float* histRange = { range };
  bool uniform = true;
  bool accumulate = false;
  cv::Mat hist;
  cv::calcHist( &channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

  for ( int i = 0; i < histSize && med < 0.0; ++i ) {
    bin += cvRound( hist.at< float >( i ) );
    if ( bin > m && med < 0.0 )
      med = i;
  }
  return med;
}

PixelSelector::PixelSelector() 
    {
      region_min_thresh = 1;
      g_th = 7; // from DSO paper
      desired_number_of_points = 1000;
      prev_number_of_points = 1000;
      prev_region_size = 15;
    }

Eigen::Matrix<float,2,Eigen::Dynamic> PixelSelector::selectPixels(cv::Mat gray, cv::Mat grad_mag) {
	
  Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coords(2,gray.cols*gray.rows);
  //std::cout << prev_number_of_points << " " << desired_number_of_points << std::endl;
  float ratio = sqrt((float) prev_number_of_points/( (float) desired_number_of_points));
  //std::cout << ratio << std::endl;
  //std::cout << prev_region_size << std::endl;
  int d = (int) ((float) prev_region_size*ratio);
  prev_region_size = d;
  //std::cout << d << std::endl;
  int thresh = g_th;
  int index = 0;

  cv::Mat pixel_selected(gray.rows,gray.cols,CV_8UC1,cv::Scalar(255));

  for (uint i=0; i<3; i++) {
    std::vector<std::vector<int>> keypoint_grid_count(gray.rows/d,std::vector<int>(gray.cols/d,0));
    for(uint i=0; i<keypoint_grid_count.size(); i++) {
      for(uint j=0; j<keypoint_grid_count[0].size(); j++) {
        if (keypoint_grid_count[i][j] < region_min_thresh) { // fix this check, check for duplicate points
          cv::Mat grad_block = grad_mag(cv::Rect(j*d,i*d,d,d));
          cv::Mat mask = pixel_selected(cv::Rect(j*d,i*d,d,d));
          double median = median_mat(grad_block);
          double minVal; 
          double maxVal; 
          cv::Point minLoc; 
          cv::Point maxLoc;
          cv::minMaxLoc(grad_block, &minVal, &maxVal, &minLoc, &maxLoc, mask);
          //std::cout << maxVal << " " << median << std::endl;
          if (maxVal > (thresh + median)) {
            pixel_selected.at<uint8_t>(maxLoc.y,maxLoc.x) = 0;
            maxLoc.x += j*d;
            maxLoc.y += i*d;
            pixel_coords(0,index) = maxLoc.x;
            pixel_coords(1,index) = maxLoc.y;
            index++;
          }
        }
      }
    }
    d*=2;
    thresh -= 1;
  }
  prev_number_of_points = index;

  return pixel_coords.leftCols(index);
}

/*Eigen::Vector2f epipolar_match(Eigen::Vector3f p_h, Eigen::Matrix3f F, cv::Mat curr, cv::Mat ref_patch, bool& match) {
  match = true;
  Eigen::Vector2f out;
  Eigen::Vector3f l = F*p_h;
  float min_ssd = 10000;
  float second_min_ssd = 10000;
  if (std::abs(l(0))<std::abs(l(1))) {
    int start = p_h(0)-line_check_dist;
    int end = p_h(0)+line_check_dist;
    if (start < (SSD_size/2+1))
      start = SSD_size/2+1;
    if (end > (curr.cols-SSD_size/2-1))
      end = curr.cols-SSD_size/2-1;
    for (int u=start; u<end; u++) {
      int v = -(l(2)+l(0)*u)/l(1);
      if ((v>(curr.rows-SSD_size/2-1)) || (v<SSD_size/2+1)) {
        continue;
      }
      //std::cout << "u: " << u << " v: " << v << std::endl;
      //cv::circle(curr, cv::Point(u,v) ,3,cv::Scalar(0,0,255));
      cv::Mat curr_block = curr(cv::Rect(u-SSD_size/2,v-SSD_size/2,SSD_size,SSD_size));
      //cv::Mat diff = ref_patch - curr_block;
      cv::Mat diff;
      cv::absdiff(ref_patch,curr_block,diff);
      float ssd = norm(diff, cv::NORM_L2);

      if (ssd < min_ssd) {
        //std::cout << "ssd: " << ssd << std::endl;
        out(0) = u; out(1) = v;
        second_min_ssd = min_ssd;
        min_ssd = ssd;
      }
    }
  }
  else {
    int start = p_h(1)-line_check_dist;
    int end = p_h(1)+line_check_dist;
    if (start < (SSD_size/2+1))
      start = SSD_size/2+1;
    if (end > (curr.rows-SSD_size/2-1))
      end = curr.rows-SSD_size/2-1;
    for (int v=start; v<end; v++) {
      int u = -(l(2)+l(1)*v)/l(0);
      if ((u>(curr.cols-SSD_size/2-1)) || (u<SSD_size/2+1)) {
        continue;
      } // maybe should find when to break to make faster?
      //std::cout << "u: " << u << " v: " << v << std::endl;
      //cv::circle(curr, cv::Point(u,v) ,3,cv::Scalar(0,0,255));
      cv::Mat curr_block = curr(cv::Rect(u-SSD_size/2,v-SSD_size/2,SSD_size,SSD_size));
      //cv::Mat diff = ref_patch - curr_block;
      cv::Mat diff;
      cv::absdiff(ref_patch,curr_block,diff);
      float ssd = norm(diff, cv::NORM_L2);

      if (ssd < min_ssd) {
        out(0) = u; out(1) = v;
        second_min_ssd = min_ssd;
        min_ssd = ssd;
      }
    }
  }
  //std::cout << min_ssd/(SSD_size*SSD_size) << std::endl;
  if  (1.1*min_ssd > second_min_ssd) { // ||(min_ssd/(SSD_size*SSD_size) > 3.0)
    //out(0) = 0; out(1) = 0;
    match = false;
  }
  //cv::imshow("c",curr);
  //cv::waitKey(0);
  return out;

}*/

/*Eigen::Matrix<float,2,Eigen::Dynamic> PixelSelector::epipolar_line_search(Eigen::Matrix4f T, 
  Eigen::Matrix<float,2,Eigen::Dynamic> prev_pixel_coord, cv::Mat curr, cv::Mat ref, int& features_tracked) {

  features_tracked = 0;
  Eigen::Matrix<float,2,Eigen::Dynamic> curr_pixel_coord = prev_pixel_coord;
  Eigen::Matrix3f Tx = vec3d_to_skew_symmetric(T.topRightCorner(3,1));
  Eigen::Matrix3f E = Tx*T.topLeftCorner(3,3);
  Eigen::Matrix3f F = invK.transpose()*E*invK;
  for (int i=0; i<prev_pixel_coord.cols(); i++) {
    if ((prev_pixel_coord(0,i) >= (curr.cols-SSD_size/2-1)) || (prev_pixel_coord(1,i) >= (curr.rows-SSD_size/2-1)) 
        || (prev_pixel_coord(0,i)<(SSD_size/2+1)) || (prev_pixel_coord(1,i)<(SSD_size/2+1))) {
      curr_pixel_coord.col(i) = Eigen::Vector2f::Zero();
      continue;
    }
    Eigen::Vector3f p_h;
    p_h.topRows(2) = prev_pixel_coord.col(i); p_h(2) = 1;
    //std::cout << p_h.transpose() << std::endl;
    //std::cout << l.transpose() << std::endl;
    cv::Mat ref_patch = ref(cv::Rect(p_h(0)-SSD_size/2, p_h(1)-SSD_size/2,SSD_size,SSD_size));
    bool match = true;
    curr_pixel_coord.col(i) = epipolar_match(p_h, F, curr, ref_patch, match);
    if (match) features_tracked++;
  }
  return curr_pixel_coord;
}*/