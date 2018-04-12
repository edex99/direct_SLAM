#include "SparseAlignment.h"

inline float huber_weight(float x) {
	if (std::abs(x) <= Huber_constant)
		return 1.0;
	else
		return Huber_constant/std::abs(x);
}
inline float tukey_weight(float x) {
	if (std::abs(x) <= Tukey_constant) 
		return (1-((x*x)/(Tukey_constant*Tukey_constant)))*(1-((x*x)/(Tukey_constant*Tukey_constant)));
	else
		return 0;
}

ImageAlignment::ImageAlignment(int num_levels, int gradient_thresh){
	gradient_threshold = gradient_thresh;
	boost::shared_ptr<Eigen::Matrix<float,3,Eigen::Dynamic> > temp1;
	boost::shared_ptr<Eigen::Matrix<float,Eigen::Dynamic,8> > temp2;
	boost::shared_ptr<Eigen::VectorXf> temp3;
	for (int i=0; i<num_levels; i++) {
		keyframe_pts.push_back(temp1);
		keyframe_jacobians.push_back(temp2);
		keyframe_ref_vals.push_back(temp3);
		float a = std::pow(2, (int) -i); //float b = std::pow(2.0,-i-1);
		Eigen::Matrix3f K_multiplier;
		K_multiplier << a,0,0, 0,a,0, 0,0,1;
		K_scales.push_back(K_multiplier);
	}
}

void ImageAlignment::setKeyframe(std::vector<cv::Mat>& image_pyr, std::vector<cv::Mat>& depth_pyr){
	/*keyframe_depths = depth_pyr;
	keyframe_images = image_pyr;*/
	for (int i=image_pyr.size()-1; i>=0; i--) {
		cv::Mat ref = image_pyr[i];
		cv::Mat ref_grad_x, ref_grad_y, ref_grad_mag;
    	calcImageDerivatives(ref,ref_grad_x,ref_grad_y,ref_grad_mag);
    	Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coord;//(2,ref.cols*ref.rows);

		cv::Mat mask;
		if (ref.rows > 60) {
			cv::threshold(ref_grad_mag,mask,gradient_threshold,255,0);
		}
		else {
			cv::threshold(ref_grad_mag,mask,0,255,0);
		}
		cv::Mat depthLocations;
		cv::Mat depthLocations_f;
		cv::Mat depthLocations_1channel;
		cv::Mat depthImage_8bit;
		depth_pyr[i].convertTo(depthImage_8bit,CV_8UC1);
		cv::bitwise_and(mask, depthImage_8bit,mask);
		cv::findNonZero(mask, depthLocations);
		depthLocations.convertTo(depthLocations_f, CV_32FC1);
		depthLocations_1channel = (depthLocations_f.reshape(1,0)).t();
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> pixel_coord2(depthLocations_1channel.ptr<float>(), depthLocations_1channel.rows, depthLocations_1channel.cols);
		pixel_coord = pixel_coord2;

		if (i==0) {
			std::cout << pixel_coord.cols() << std::endl;
			if (pixel_coord.cols() > desired_num_points+num_points_thresh) {
				gradient_threshold++;
			}
			else if (pixel_coord.cols() < desired_num_points-num_points_thresh) {
				gradient_threshold--;
			}
		}


    	Eigen::Matrix3f K_multiplier = K_scales[i];
		Eigen::Matrix<float,3,Eigen::Dynamic> Pts(3,pixel_coord.cols());
		Pts = calc3DPoints2(depth_pyr[i], pixel_coord, K_multiplier);
		keyframe_pts[i].reset(new Eigen::Matrix<float,3,Eigen::Dynamic>(Pts));
		Eigen::VectorXf ref_val;
		ref_val = refValues(ref, pixel_coord);
		keyframe_ref_vals[i].reset(new Eigen::VectorXf(ref_val));
		Eigen::MatrixXf J(pixel_coord.cols(),8);
		J.leftCols(6) = calc_Jacobian(pixel_coord, Pts, ref_grad_x, ref_grad_y, K_multiplier);
		J.col(6) = ref_val;
		J.col(7) = Eigen::VectorXf::Constant(pixel_coord.cols(),1.0);
		keyframe_jacobians[i].reset(new Eigen::Matrix<float,Eigen::Dynamic,8>(J));
	}
	std::cout << gradient_threshold << std::endl;
}

Eigen::VectorXf ImageAlignment::robust_residuals(Eigen::VectorXf residuals, Eigen::VectorXf valid_pts) {
	int m = residuals.rows()/2;
	int size = residuals.rows();
	Eigen::VectorXf sorted(size);
	sorted = residuals.array().abs();
	Eigen::VectorXf weights = Eigen::VectorXf::Zero(size);
	//std::cout << residuals.data() << " " << residuals.data()+residuals.rows() << std::endl;
	std::nth_element(sorted.data(), sorted.data()+m, sorted.data()+sorted.size());
	//std::sort(sorted.data(), sorted.data()+sorted.size());
	//std::nth_element(sorted.data().begin(),sorted.data().begin()+m,sorted.data().end());

	float median = sorted(m);
	float stdev = 1.4826*(1+(5/(size-6)))*median;
	residuals = residuals.array()/stdev;
	//Eigen::VectorXf weights(size);
	for (int i=0; i<size; i++) {
		//if (valid_pts(i) > 0) {
			weights(i) = huber_weight(residuals(i));//*(25)/(25+grad_mag(i));
		//}
	}
	return weights;

}

Eigen::Matrix<float,2,6> ImageAlignment::dw_dtheta(Eigen::Vector3f P, Eigen::Matrix3f K_multiplier) {
	Eigen::Matrix<float,2,6> out;
	float inv_Z = 1/P(2);
	/*out.topRows(1) << P(0)*P(1)*inv_Z, -(P(0)*P(0)+P(2)*P(2))*inv_Z, P(1), 1, 0, -P(0)*inv_Z;
	out.topRows(1) *= fx*K_multiplier(0,0);
	out.bottomRows(1) << (P(1)*P(1)+P(2)*P(2))*inv_Z, -P(0)*P(1)*inv_Z, -P(0), 0, 1, -P(1)*inv_Z;
	out.bottomRows(1) *= fy*K_multiplier(1,1);
	out *= inv_Z;*/
	Eigen::Matrix<float,2,3> proj_partial;
	proj_partial << fx*K_multiplier(0,0), 0, -P(0)*fx*K_multiplier(0,0)*inv_Z, 
					0, fy*K_multiplier(1,1), -P(1)*fy*K_multiplier(1,1)*inv_Z;
	Eigen::Matrix<float,3,6> warp_partial;
	warp_partial.leftCols(3) = vec3d_to_skew_symmetric(-P);
	warp_partial.rightCols(3) = Eigen::Matrix3f::Identity();
	out = inv_Z*(proj_partial*warp_partial);
	/*std::cout << proj_partial << std::endl;
	std::cout << warp_partial << std::endl;
	std::cin.get();*/

	return out;
}

Eigen::Matrix<float,Eigen::Dynamic,6> ImageAlignment::calc_Jacobian(Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coord, 
	Eigen::Matrix<float,3,Eigen::Dynamic> Pts, cv::Mat& grad_x, cv::Mat& grad_y, Eigen::Matrix3f K_multiplier) {
	
	Eigen::Matrix<float,Eigen::Dynamic,6> J(Pts.cols(),6);
	Eigen::RowVector2f grad_I;
	Eigen::Matrix<float,2,6> warp_derivative;
	for (uint i=0; i<pixel_coord.cols(); i++) {
		grad_I(0) = grad_x.at<int16_t>(pixel_coord(1,i), pixel_coord(0,i)); // u axis first
		grad_I(1) = grad_y.at<int16_t>(pixel_coord(1,i), pixel_coord(0,i)); // v axis second
		warp_derivative = dw_dtheta(Pts.col(i), K_multiplier);
		J.row(i) = grad_I*warp_derivative;

	}																					
	return J;
}																																																																																																											

Eigen::Matrix<float,8,1> ImageAlignment::optimize(Eigen::Matrix<float,8,1> theta_init, cv::Mat& ref, cv::Mat& curr, 
	cv::Mat& depth_ref, cv::Mat& ref_grad_x, cv::Mat& ref_grad_y, Eigen::MatrixXf pixel_coord, 
	Eigen::Matrix3f K_multiplier, cv::Mat& ref_grad_mag) {

	Eigen::Matrix<float,3,Eigen::Dynamic> Pts(3,pixel_coord.cols());
	//Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coord_new(2,pixel_coord.cols());
	//Pts = calc3DPoints(depth_ref, pixel_coord, pixel_coord_new, K_multiplier);
	//pixel_coord = pixel_coord_new;
	Pts = calc3DPoints2(depth_ref, pixel_coord, K_multiplier);

	Eigen::VectorXf warped_val, ref_val, e;
	ref_val = refValues(ref, pixel_coord);
	Eigen::MatrixXf J(pixel_coord.cols(),8);
	J.leftCols(6) = calc_Jacobian(pixel_coord, Pts, ref_grad_x, ref_grad_y, K_multiplier);
	J.col(6) = ref_val;
	J.col(7) = Eigen::VectorXf::Constant(pixel_coord.cols(),1.0);
	//J.rightCols(2) = Eigen::MatrixXf::Zero(pixel_coord.cols(),2);


	Eigen::Matrix<float,8,1> theta, theta_prev;
	//theta << 0,0,0,0,0,0,0,0;
	theta = theta_init;

	//theta.topRows(6) = se3_init;
	theta_prev << 0,0,0,0,0,0,0,0;
	Eigen::Matrix<float,8,1> delta_theta;
	delta_theta << 0,0,0,0,0,0,0,0;
	Eigen::Matrix4f T;
	T = se3_to_SE3(theta.topRows(6));

	float e_init_norm = 0;
	int iter = 0;
	Eigen::MatrixXf rhs(8,1);
	do { 

		Eigen::VectorXf valid_pts = Eigen::VectorXf::Constant(Pts.cols(),1);

		Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coord_curr(2,Pts.cols());
		Eigen::Matrix<float,3,Eigen::Dynamic> transformedPts(3,Pts.cols());
		transformedPts = transform3Dpoints(Pts, T);
		pixel_coord_curr = project3Dpoints(transformedPts, K_multiplier,valid_pts,ref.rows,ref.cols);
		warped_val = warpedValues(curr, pixel_coord_curr, valid_pts);

		warped_val = (1+theta(6))*(warped_val.array())+theta(7);
		//warped_val = (warped_val.array()-theta(7))/(1+theta(6));
		e = warped_val-ref_val;

		int num_valid = valid_pts.sum();
		Eigen::VectorXf e_new(num_valid);
		Eigen::MatrixXf J_new(num_valid,8);
		int count = 0;
		for (int i=0; i<e.rows();i++) {
			if (valid_pts(i)>0) {
				//std::cout << i << std::endl;
				e_new(count) = e(i);
				J_new.row(count) = J.row(i);
				count++; 
			}
		}

		//Eigen::VectorXf w = robust_residuals(e_new, valid_pts);

		/*diff_img2 = cv::Mat(ref.rows, ref.cols, CV_8UC1, cv::Scalar(0));
		for (int i=0; i<pixel_coord.cols();i++) {
			if (valid_pts(i) != 0) {
				diff_img2.at<uint8_t>(pixel_coord(1,i), pixel_coord(0,i)) = abs(e(i));
			}
		}
		cv::imshow("diff2", diff_img2);
		cv::waitKey();*/

		//std::clock_t start;
		//double duration;
		//start = std::clock();
		Eigen::MatrixXf lhs(8,8), temp(8,Pts.cols());
		lhs = (J_new.transpose())*J_new;
		rhs = (J_new.transpose())*e_new;
		delta_theta = (lhs.inverse())*rhs;
		//duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
		//std::cout<<"time: "<< duration <<'\n';

		/*Eigen::MatrixXf lhs(6,6), rhs(6,1), temp(6,Pts.cols());
		lhs = (J.leftCols(6).transpose())*valid_pts.asDiagonal()*(J.leftCols(6));
		rhs = (J.leftCols(6).transpose())*valid_pts.asDiagonal()*e;
		delta_theta.topRows(6) = (lhs.inverse())*(rhs);*/

		Eigen::Matrix4f T_prev = T;
		T = T_prev*se3_to_SE3(-delta_theta.topRows(6));
		theta_prev = theta;
		theta.topRows(6) = SE3_to_se3(T);

		float alpha_prev = theta(6);
		theta(6) = (alpha_prev-delta_theta(6))/(1+delta_theta(6));
		theta(7) = (theta(7)-delta_theta(7))/(1+delta_theta(6));
		//theta(6) = (alpha_prev+1)/(1+delta_theta(6));
		//theta(7) = theta(7)-(alpha_prev+1)*delta_theta(7);


		/*std::cout << "residual norm " << e.norm() << std::endl;
		std::cout << "gradient: " << (J.transpose()*e).lpNorm<Eigen::Infinity>()/valid_pts.sum() << std::endl;
		cin.get();*/
		iter++;

	} while ( (iter < 100) && ((theta-theta_prev).norm() > .000001) && (delta_theta.norm() > .000001)
		&& ( rhs.lpNorm<Eigen::Infinity>() > .00000001) );

	//cv::imshow("diff2", diff_img2);
	//cv::waitKey();

	return theta;
}

Eigen::Matrix<float,8,1> ImageAlignment::optimize2(Eigen::Matrix<float,8,1> theta_init, cv::Mat& curr,int scale) {

	Eigen::Matrix3f K_multiplier = K_scales[scale];
	int num_points = (*(keyframe_pts[scale])).cols();

	Eigen::VectorXf warped_val, e;
	Eigen::Matrix<float,8,1> theta, theta_prev;
	theta = theta_init;

	theta_prev << 0,0,0,0,0,0,0,0;
	Eigen::Matrix<float,8,1> delta_theta;
	delta_theta << 0,0,0,0,0,0,0,0;
	Eigen::Matrix4f T;
	T = se3_to_SE3(theta.topRows(6));

	float e_init_norm = 0;
	int iter = 0;
	Eigen::MatrixXf lhs(8,8), lhs_inv(8,8), rhs(8,1), temp(8,num_points);
	do { 

		Eigen::VectorXf valid_pts = Eigen::VectorXf::Constant(num_points,1);

		Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coord_curr(2,num_points);
		Eigen::Matrix<float,3,Eigen::Dynamic> transformedPts(3,num_points);
		transformedPts = transform3Dpoints(*(keyframe_pts[scale]), T);
		pixel_coord_curr = project3Dpoints(transformedPts, K_multiplier,valid_pts,curr.rows,curr.cols);
		warped_val = warpedValues(curr, pixel_coord_curr, valid_pts);

		warped_val = (1+theta(6))*(warped_val.array())+theta(7);
		e = warped_val-*(keyframe_ref_vals[scale]);

		// Which one of these is faster? First gets rid of unneccesary multiplications but copies entire Jacobian
		// Second one generally has more multiplications due to diagonal matrix but doesn't copy data
		// Second one would be better if robust estimation is used
		// Test average time for multiple runs?

		/*int num_valid = valid_pts.sum(); // DO THIS REDUCTION INSIDE THE WARPEDVALUES Function? Get rid of another loop
		Eigen::VectorXf e_new(num_valid);
		Eigen::MatrixXf J_new(num_valid,8);
		int count = 0;
		for (int i=0; i<e.rows();i++) {
			if (valid_pts(i)>0) {
				e_new(count) = e(i);
				J_new.row(count) = (*(keyframe_jacobians[scale])).row(i);
				count++; 
			}
		}
		Eigen::VectorXf w = robust_residuals(e_new, valid_pts);
		temp = J_new.transpose()*w.asDiagonal();
		lhs = temp*J_new;
		rhs = temp*e_new;*/

		temp = (*(keyframe_jacobians[scale])).transpose()*valid_pts.asDiagonal();
		lhs = temp*(*(keyframe_jacobians[scale]));
		rhs = temp*e;

		lhs_inv = lhs.inverse();
		delta_theta = lhs_inv*rhs;

		Eigen::Matrix4f T_prev = T;
		T = T_prev*se3_to_SE3(-delta_theta.topRows(6));
		theta_prev = theta;
		theta.topRows(6) = SE3_to_se3(T);

		float alpha_prev = theta(6);
		theta(6) = (alpha_prev-delta_theta(6))/(1+delta_theta(6));
		theta(7) = (theta(7)-delta_theta(7))/(1+delta_theta(6));

		//std::cout << iter << " " << delta_theta.norm() << " " << std::endl;

		iter++;

	} while ( (iter < 100) &&  (delta_theta.norm() > .000001)
		&& ( rhs.lpNorm<Eigen::Infinity>() > .00000001) );

	if (scale == 0)
		std::cout << std::log((100000000000*lhs_inv).determinant()) << std::endl;

	return theta;
}

Eigen::Matrix<float,8,1> ImageAlignment::directTransformationEstimation(
	std::vector<cv::Mat> pyr_img_curr, cv::Mat& out, Eigen::Matrix<float,8,1> theta_init) {

	Eigen::Matrix<float,8,1> theta;
	//theta << 0,0,0,0,0,0,0,0;
	theta = theta_init;

	/*for (int i=pyr_img_curr.size()-1; i>=0; i--) {
		cv::Mat ref = keyframe_images[i];
		cv::Mat curr = pyr_img_curr[i];
		cv::Mat ref_grad_x, ref_grad_y, ref_grad_mag;
    	calcImageDerivatives(ref,ref_grad_x,ref_grad_y,ref_grad_mag);
    	Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coord;//(2,ref.cols*ref.rows);

    	if (ref.rows > 120) {
    		continue;
    		//pixel_coord = selectPixels(ref, ref_grad_mag);
    	}
    	else {
    		cv::Mat mask;
    		cv::threshold(ref_grad_mag,mask,0,255,0);
    		cv::Mat depthLocations;
    		cv::Mat depthLocations_f;
    		cv::Mat depthLocations_1channel;
    		cv::Mat depthImage_8bit;
    		keyframe_depths[i].convertTo(depthImage_8bit,CV_8UC1);
    		cv::bitwise_and(mask, depthImage_8bit,mask);
    		cv::findNonZero(mask, depthLocations);
    		depthLocations.convertTo(depthLocations_f, CV_32FC1);
    		depthLocations_1channel = (depthLocations_f.reshape(1,0)).t();
    		depthLocations.release();
    		depthLocations_f.release();
    		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> pixel_coord2(depthLocations_1channel.ptr<float>(), depthLocations_1channel.rows, depthLocations_1channel.cols);
    		//depthLocations_1channel.release();
    		pixel_coord = pixel_coord2;

    	}

    	Eigen::Matrix3f K_multiplier = K_scales[i];
		Eigen::Matrix<float,3,Eigen::Dynamic> Pts(3,pixel_coord.cols());
		Pts = calc3DPoints2(keyframe_depths[i], pixel_coord, K_multiplier);
		keyframe_pts[i].reset(new Eigen::Matrix<float,3,Eigen::Dynamic>(Pts));
		Eigen::VectorXf ref_val;
		ref_val = refValues(ref, pixel_coord);
		keyframe_ref_vals[i].reset(new Eigen::VectorXf(ref_val));
		Eigen::MatrixXf J(pixel_coord.cols(),8);
		J.leftCols(6) = calc_Jacobian(pixel_coord, Pts, ref_grad_x, ref_grad_y, K_multiplier);
		J.col(6) = ref_val;
		J.col(7) = Eigen::VectorXf::Constant(pixel_coord.cols(),1.0);
		keyframe_jacobians[i].reset(new Eigen::Matrix<float,Eigen::Dynamic,8>(J));

		//theta = optimize(theta, ref, curr, keyframe_depths[i], ref_grad_x, ref_grad_y, pixel_coord, K_multiplier, ref_grad_mag);
		theta = optimize2(theta,curr,i);
	}*/
	for (int i=pyr_img_curr.size()-1; i>=0; i--) {
		if ((pyr_img_curr[i]).rows>480)
			continue;
		else
		theta = optimize2(theta,pyr_img_curr[i],i);
	}

	out = (1+theta(6))*(pyr_img_curr[0].clone()) + theta(7);

	return theta;
}


/*Eigen::Matrix<short,2,Eigen::Dynamic> concatenatePatternPixelCoord(Eigen::Matrix<short,2,Eigen::Dynamic> pixel_coord) {
	Eigen::Matrix<short,Eigen::Dynamic,Eigen::Dynamic> out(2,pattern.cols()*pixel_coord.cols());
	for (int i=0; i<pixel_coord.cols();i++) {
		for (int j=0; j<pattern.cols(); j++) {
			out.col(i+j) = pixel_coord.col(i) + pattern.col(j);
		}
	}
	return out;
}

Eigen::Matrix<float,2,Eigen::Dynamic> concatenatePatternSubPixelCoord(Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coord) {
	Eigen::Matrix<float,2,8> pattern_f = pattern.cast<float> ();
	Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> out(2,pattern_f.cols()*pixel_coord.cols());
	for (int i=0; i<pixel_coord.cols();i++) {
		for (int j=0; j<pattern_f.cols(); j++) {
			out.col(i+j) = pixel_coord.col(i) + pattern_f.col(j);
		}
	}
	return out;
}*/