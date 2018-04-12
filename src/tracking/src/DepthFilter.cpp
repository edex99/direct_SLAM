#include "DepthFilter.h"

float normal_pdf(float x, float m, float s)
{
    float a = (x - m) / s;
    return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}

DepthFilter::DepthFilter() {
	features_tracked = 0;
}

int DepthFilter::getFeaturesTracked() {
	return features_tracked;
}

float DepthFilter::computeTau(Eigen::Matrix4f T, Eigen::Vector3f p)
{
	// adapted from REMODE/SVO by Scaramuzza lab
	// T is from reference to kth frame
	// parallelize as many operations as possible, input matrix of points
	float p_norm = p.norm();
	Eigen::Vector3f f = p/p_norm;
	Eigen::Vector3f t = T.topRightCorner(3,1);
	Eigen::Vector3f a = p-t;
	float t_norm = t.norm();
	float a_norm = a.norm();
	float alpha = acos(f.dot(t)/t_norm); // dot product
	float beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
	float beta_plus = beta + 2*atan(1/(fx+fy));
	float gamma = 3.14-alpha-beta_plus; // triangle angles sum to PI
	float p_plus_norm = t_norm*sin(beta_plus)/sin(gamma); // law of sines
	return (p_plus_norm - p_norm); // tau
}

Eigen::Vector2f DepthFilter::epipolarMatch(Eigen::Vector3f p_h, Eigen::Matrix3f F, 
	cv::Mat curr, cv::Mat ref_patch, bool& match) {
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

      /*cv::imshow("curr_block",curr_block);
      cv::imshow("ref_patch",ref_patch);
      cv::imshow("c",curr);
      cv::waitKey(0);*/

      /*ref_patch.convertTo(ref_patch, CV_32FC1);
      curr_block.convertTo(curr_block, CV_32FC1);
      float sum_num = cv::sum(curr_block.mul(ref_patch))[0];
      float sum_denom1 = sqrt(cv::sum(curr_block.mul(curr_block))[0]);
      float sum_denom2 = sqrt(cv::sum(ref_patch.mul(ref_patch))[0]);
      float ssd = sum_num/(sum_denom1*sum_denom2);*/

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

      /*cv::imshow("curr_block",curr_block);
      cv::imshow("ref_patch",ref_patch);
      cv::imshow("c",curr);
      cv::waitKey(0);*/

      /*ref_patch.convertTo(ref_patch, CV_32FC1);
      curr_block.convertTo(curr_block, CV_32FC1);
      float sum_num = cv::sum(curr_block.mul(ref_patch))[0];
      float sum_denom1 = sqrt(cv::sum(curr_block.mul(curr_block))[0]);
      float sum_denom2 = sqrt(cv::sum(ref_patch.mul(ref_patch))[0]);
      float ssd = sum_num/(sum_denom1*sum_denom2);*/

      if (ssd < min_ssd) {
        //cv::imshow("r",ref_patch);
        //cv::imshow("curr_patch",curr_block);
        //cv::waitKey();
        //std::cout << "ssd: " << ssd << std::endl;
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

}

Eigen::Matrix<float,2,Eigen::Dynamic> DepthFilter::epipolarLineSearch(Eigen::Matrix4f T, 
  Eigen::Matrix<float,2,Eigen::Dynamic> prev_pixel_coord, cv::Mat curr, cv::Mat ref, Eigen::VectorXi& matched) {

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
    curr_pixel_coord.col(i) = epipolarMatch(p_h, F, curr, ref_patch, match);
    matched(i) = match;
  }
  return curr_pixel_coord;
}

Eigen::Matrix<float,3,Eigen::Dynamic> DepthFilter::triangulatePoints(Eigen::Matrix4f T1, Eigen::Matrix4f T2, 
	Eigen::Matrix<float,2,Eigen::Dynamic> p1, Eigen::Matrix<float,2,Eigen::Dynamic> p2, cv::Mat curr_depth) {

	Eigen::Matrix<float,4,Eigen::Dynamic> out_homogeneous(4, p1.cols());
	Eigen::Matrix<float,3,4> temp;
	temp.leftCols(3) = Eigen::Matrix3f::Identity();
	Eigen::Matrix<float,3,4> M1 = K*temp*T1;
	Eigen::Matrix<float,3,4> M2 = K*temp*T2;

	for (int i=0; i<p1.cols(); i++) {
		uint16_t d = curr_depth.at<uint16_t>(p2(1,i),p2(0,i));
		if (d != 0) {
			Eigen::Vector3f uv1;
			Eigen::Vector4f pt_h;
			pt_h(3) = 1;
			uv1 << p2(0,i),p2(1,i),1;
		    float z = d*idepthScale;
		    pt_h.topRows(3) = z*(invK*uv1);
		    out_homogeneous.col(i) = T2.inverse()*pt_h; 
		    // NEED to TRANSFORM THIS POINT INTO REFERENCE FRAME TO GET DEPTH
		}
		else {
			Eigen::Matrix4f A;
			A.row(0) = p1(0,i)*M1.row(2) - M1.row(0);
			A.row(1) = p1(1,i)*M1.row(2) - M1.row(1);
			A.row(2) = p2(0,i)*M2.row(2) - M2.row(0);
			A.row(3) = p2(1,i)*M2.row(2) - M2.row(1);

			Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
			Eigen::Matrix4f V = svd.matrixV();

			out_homogeneous.col(i) = (V.row(3)).transpose();
		}
	}
	//std::cout << ((out_homogeneous.array().rowwise()/(out_homogeneous.row(3).array())).topRows(3)).col(p1.cols()-1).transpose() << std::endl;
	Eigen::Matrix<float,3,Eigen::Dynamic> out;
	out = (out_homogeneous.array().rowwise()/(out_homogeneous.row(3).array())).matrix().topRows(3);

	return out;
}

void DepthFilter::initializeFilter(cv::Mat depth, Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coord) {

	int size = pixel_coord.cols();
	//filter.resize(4, size);
	Eigen::Matrix<float,4,Eigen::Dynamic> filter_init(4,size);
	for (int i=0; i<size; i++) { // parallelize this, get depth mat in Eigen::vector
		uint16_t depth_uint16 = depth.at<uint16_t>(pixel_coord(1,i),pixel_coord(0,i));
		if (depth_uint16 != 0) {
			filter_init(0,i) = 10;
			filter_init(1,i) = 10;
			filter_init(2,i) = depthScale/depth_uint16;
			filter_init(3,i) = 3.0*depth_std_dev_factor*(filter_init(2,i)*filter_init(2,i)); // fix this? start more uncertain?
		}
		else {
			filter_init(0,i) = 1;
			filter_init(1,i) = 1;
			filter_init(2,i) = 1.0/10.0; // set mean to large number? or for indoor environment assume 10 meters?
			filter_init(3,i) = 1.0/(min_depth*min_depth*4); // need to figure this out, what is the minimum depth?  
		}
	}
	//filter = boost::make_shared<Eigen::Matrix<float,4,Eigen::Dynamic>(filter_init);
	filter.reset(new Eigen::Matrix<float,4,Eigen::Dynamic>(filter_init));
	//std::cout << filter.transpose() << std::endl;
}


void DepthFilter::updateFilter(Eigen::Matrix4f T, 
	Eigen::Matrix<float,2,Eigen::Dynamic> p1, cv::Mat curr_depth, cv::Mat ref, cv::Mat curr) {
	// parameters for Vector4f: inliers a, outliers b, mean u, variance var

	Eigen::VectorXi matched = Eigen::VectorXi::Constant(p1.cols(),0);
	Eigen::Matrix<float,2,Eigen::Dynamic> p2 = epipolarLineSearch(T, 
  p1, curr, ref, matched);

	Eigen::Matrix<float,3,Eigen::Dynamic> pts_triangulated = triangulatePoints(Eigen::Matrix4f::Identity(), T, p1, p2, curr_depth);

	for (int i=0; i<pts_triangulated.cols(); i++) {
		if (!matched(i)) {
			(*filter)(1,i) += 1.0;
			continue;
		}

		float x = 1.0f/pts_triangulated(2,i);
		//std::cout << i << " " << x << std::endl;
		//if (i==(filter.cols()-1))
		//	std::cout << pts_triangulated(2,i) << std::endl;
		//std::cout << i << " " << pts_triangulated(2,i) << std::endl;
		float tau = computeTau(T, pts_triangulated.col(i));
		float tau_inv  =  0.5 * (1.0/std::max((float) 0.0000001, x-tau) - 1.0/(x+tau));
		float tau2 = tau_inv*tau_inv;
		//std::cout << i << " " << pts_triangulated(2,i) << std::endl;//<< " " << tau << " " << tau2 << std::endl;

		// use these for epipolar matching
		/*float z_min = std::max(0.000001f, filter(2,i) - (float) sqrt(filter(3,i)));
		float z_max = filter(2,i) + sqrt(filter(3,i));*/

		float s2 = 1.0/(1.0/(*filter)(3,i) + 1.0/(tau2));
		float m = s2*((*filter)(2,i)/(*filter)(3,i) + x/(tau2));

		int num_obs = (*filter)(0,i)+(*filter)(1,i);
		float gaussian_scale = sqrt(tau2+(*filter)(3,i));
		float C1 = (*filter)(0,i)/num_obs*normal_pdf(x,(*filter)(2,i),gaussian_scale);
		float C2 = (*filter)(1,i)/(num_obs*depth_range);

		float normalization_constant = C1 + C2;
		C1 /= normalization_constant;
		C2 /= normalization_constant;
		float f = C1*((*filter)(0,i)+1.)/(num_obs+1.) + C2*(*filter)(0,i)/(num_obs+1.);
		float e = C1*((*filter)(0,i)+1.)*((*filter)(0,i)+2.)/((num_obs+1.)*(num_obs+2.))
		  + C2*(*filter)(0,i)*((*filter)(0,i)+1.0f)/((num_obs+1.0f)*(num_obs+2.0f));
		 float u_new = C1*m+C2*(*filter)(2,i);
		 float var_new = C1*(s2 + m*m) + C2*((*filter)(3,i)+(*filter)(2,i)*(*filter)(2,i)) - u_new*u_new;

		(*filter)(0,i) = (e-f)/(f-e/f);;
		(*filter)(1,i) = (*filter)(0,i)*(1.0f-f)/f;
		(*filter)(2,i) = u_new;
		(*filter)(3,i) = var_new;

	}

	/*for (int i=0; i<pts_triangulated.cols(); i++) {
		if (matched(i))
			std::cout << pts_triangulated.col(i).transpose() << std::endl;
	}*/
	//std::cout << pts_triangulated.transpose() << std::endl;
	//std::cout << filter->transpose() << std::endl;

	features_tracked = matched.sum();
	cv::Mat temp_ref = ref.clone();
    cv::Mat temp_curr = curr.clone();
    for(uint i=0; i<p1.cols(); i++) {
        cv::Point pt;
        pt.x = p2(0,i);
        pt.y = p2(1,i);
        cv::circle(temp_curr, pt ,3,cv::Scalar(0));
        pt.x = p1(0,i);
        pt.y = p1(1,i);
        cv::circle(temp_ref, pt ,3,cv::Scalar(0));
    }
    std::cout << "tracked points: " << features_tracked << std::endl;
    /*cv::imshow("img_ref",temp_ref);
    cv::imshow("img_curr",temp_curr);
    cv::waitKey(0);*/
}