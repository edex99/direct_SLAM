#include "ImageProcessing.h"

std::vector<cv::Mat> createImagePyramid(cv::Mat gray, uint num_levels) {
	std::vector<cv::Mat> pyr_imgs;
	cv::Mat downsampled_img;
	pyr_imgs.push_back(gray);
	for (uint i=1; i<num_levels; i++) {
		cv::pyrDown(gray,downsampled_img);
		pyr_imgs.push_back(downsampled_img);
		gray = downsampled_img;
	}
	return pyr_imgs;
}

std::vector<cv::Mat> createDepthPyramid(cv::Mat depth, uint num_levels) {
	std::vector<cv::Mat> pyr_depth;
	cv::Mat downsampled_depth;
	pyr_depth.push_back(depth);
	for (uint i=1; i<num_levels; i++) {
		cv::resize(depth, downsampled_depth, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST); // CHECK INTERPOLATION TYPES
		pyr_depth.push_back(downsampled_depth);
		depth = downsampled_depth; // USE FULL IMAGE RESIZE INSTEAD? accuracy vs. time to resize from full image
	}
	return pyr_depth;
}

void calcImageDerivatives(cv::Mat gray, cv::Mat& grad_x, cv::Mat& grad_y, cv::Mat& grad_mag) {
	//cv::Mat blur; // DO WE NEED BLURRING? MAKE SURE NOT REDUNDANT WITH FULL CODE 
    //cv::GaussianBlur(gray, blur, cv::Size(blur_kernel_size,blur_kernel_size),0,0);
    cv::Scharr( gray, grad_x, CV_16S,1,0,3);
    cv::Scharr( gray, grad_y, CV_16S,0,1,3);
    grad_x = grad_x/32;
    grad_y = grad_y/32;
    
    cv::Mat abs_grad_x, abs_grad_y;
    convertScaleAbs( grad_x, abs_grad_x);
    convertScaleAbs( grad_y, abs_grad_y);
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_mag ); // should convert this to L2 gradient instead of L1?
    //std::cout << grad_x.type() << std::endl;
    //std::cin.get();
}

Eigen::VectorXf getGradMag(cv::Mat grad_mag, Eigen::Matrix<float,2,Eigen::Dynamic> pixel_coord) {
	Eigen::VectorXf out(pixel_coord.cols());
	for (int i=0; i<pixel_coord.cols();i++) {
		out(i) = grad_mag.at<uchar>(pixel_coord(1,i),pixel_coord(0,i));
	}
	return out;
}

float bilinearInterpolation(uint8_t q11, uint8_t q12, uint8_t q21, uint8_t q22, 
	float x1, float x2, float y1, float y2, float x, float y) 
{
    float x2x1, y2y1, x2x, y2y, yy1, xx1;
    x2x1 = x2 - x1;
    y2y1 = y2 - y1;
    x2x = x2 - x;
    y2y = y2 - y;
    yy1 = y - y1;
    xx1 = x - x1;
    return 1.0 / (x2x1 * y2y1) * (
        q11 * x2x * y2y +
        q12 * xx1 * y2y +
        q21 * x2x * yy1 +
        q22 * xx1 * yy1
    );
}

Eigen::Matrix<float,3,Eigen::Dynamic> calc3DPoints(cv::Mat depth, 
	Eigen::MatrixXf pixel_coord, Eigen::Matrix<float,2,Eigen::Dynamic>& pixel_coord_new, Eigen::Matrix3f K_multiplier) {

	int index=0;
	Eigen::Matrix<float,3,Eigen::Dynamic> Pts(3,pixel_coord.cols());
	Eigen::Matrix3f inv_K_new = (K_multiplier*K).inverse();
	for (int i=0; i<pixel_coord.cols(); i++) { // parallelize this, get depth mat in Eigen::vector
		int x = pixel_coord(0,i);
		int y = pixel_coord(1,i);
		uint16_t d = depth.at<uint16_t>(y,x);
		if (d!=0) {
			pixel_coord_new(0,index) = x;
			pixel_coord_new(1,index) = y;
			Eigen::Vector3f uv1,point3D;
			uv1 << x,y,1;
		    float z = d*idepthScale;
		    point3D = z*(inv_K_new*uv1);
		    Pts.col(index) = point3D;
		    index++;
		}
	}
	pixel_coord_new.conservativeResize(2,index);
	return Pts.leftCols(index);
}

Eigen::Matrix<float,3,Eigen::Dynamic> calc3DPoints2(cv::Mat depth, 
	Eigen::MatrixXf pixel_coord, Eigen::Matrix3f K_multiplier) {

	Eigen::Matrix<float,3,Eigen::Dynamic> Pts(3,pixel_coord.cols());
	Eigen::Matrix<float,3,Eigen::Dynamic> pixel_coord_homogeneous =  Eigen::MatrixXf::Constant(3,pixel_coord.cols(),1);
	pixel_coord_homogeneous.topRows(2) = pixel_coord;
	Eigen::Matrix3f inv_K_new = (K_multiplier*K).inverse();
	Eigen::RowVectorXf depth_val(pixel_coord.cols());
	for (int i=0; i<pixel_coord.cols(); i++) { // parallelize this, get depth mat in Eigen::vector
		depth_val(i) = idepthScale*((float) depth.at<uint16_t>(pixel_coord(1,i),pixel_coord(0,i)));
	}
	Pts = (inv_K_new*pixel_coord_homogeneous).array().rowwise()*(depth_val.array());
	return Pts;
}

Eigen::VectorXf refValues(cv::Mat img, Eigen::MatrixXf pixel_coord) {
	
	Eigen::VectorXf vals(pixel_coord.cols());
	for (int i=0; i<pixel_coord.cols(); i++) {
		//std::cout << "pixel coord: " << (pixel_coord.col(i)).transpose() << std::endl;
		//std::cout << " val: " << img.at<uint16_t>(pixel_coord(1,i), pixel_coord(0,i)) << std::endl;
		vals(i) = (float) img.at<uint8_t>(pixel_coord(1,i), pixel_coord(0,i));
	}  // use ptr code below if all pixels used, but for pixel selection use at
	// fast processing for whole image, reintegrate this later
	/*Eigen::VectorXf vals(img.rows*img.cols);
	for(int i=0; i<img.rows; i++) {
	    uint16_t* p = img.ptr<uint16_t>(i);
	    for (int j=0; j<img.cols; j++) {
	       	vals(i*img.cols + j) = p[j];
	    }
	}*/
	return vals;
}

Eigen::Matrix<float,3,Eigen::Dynamic> transform3Dpoints(Eigen::Matrix<float,3,Eigen::Dynamic> Pts, 
	Eigen::Matrix4f T) {

	//std::cout << T << std::endl;
	//std::cout << T.inverse() << std::endl;
	//std::cout << transformationInverse(T) << std::endl;

	Eigen::Matrix<float,4,Eigen::Dynamic> Pts_homogenous = Eigen::MatrixXf::Constant(4,Pts.cols(),1);
	Pts_homogenous.topRows(3) = Pts;
	Eigen::Matrix<float,4,Eigen::Dynamic> transformedPts_homogenous = T*Pts_homogenous;
	return transformedPts_homogenous.topRows(3);

	/*w = transformedPts.row(2).transpose();

	Eigen::Matrix<float,3,Eigen::Dynamic> image_coords = K_multiplier*K*transformedPts;
	image_coords = image_coords.array().rowwise()/(image_coords.row(2).array());
	return image_coords.topRows(2);*/
}

Eigen::Matrix<float,2,Eigen::Dynamic> project3Dpoints(Eigen::Matrix<float,3,Eigen::Dynamic> Pts, 
	Eigen::Matrix3f K_multiplier, Eigen::VectorXf& valid, int row, int col) {

	Eigen::Matrix<float,3,Eigen::Dynamic> image_coords = K_multiplier*K*Pts;
	Eigen::ArrayXf z = image_coords.row(2).array();
	valid = (z>0).select(valid,Eigen::VectorXf::Zero(valid.rows())); // check in front of camera, is this correct?
	image_coords = image_coords.array().rowwise()/(image_coords.row(2).array());
	Eigen::ArrayXf x = image_coords.row(0).array();
	Eigen::ArrayXf y = image_coords.row(1).array();
	valid = ((x > 0) && (x<col) && (y>0) && (y<row)).select(valid,Eigen::VectorXf::Zero(valid.rows())); // check within image bounds

	return image_coords.topRows(2);
}

Eigen::VectorXf warpedValues(cv::Mat img, Eigen::MatrixXf pixel_coord_float, Eigen::VectorXf valid) {

	float epsilon = 0.001;
	Eigen::VectorXf newVals(pixel_coord_float.cols());

	for (uint i=0; i<pixel_coord_float.cols();i++) {

		if (valid(i)<1) {
			newVals(i) = 0;
			continue;
		}

		float x = pixel_coord_float(0,i);
		float y = pixel_coord_float(1,i);
		int x_floor = std::floor(x);
		int x_ceil = std::ceil(x);
		int y_floor = std::floor(y);
		int y_ceil = std::ceil(y);


		if ((y-y_floor) < epsilon)
			y_ceil = y_floor+1;
		if ((y_ceil-y) < epsilon)
			y_floor = y_ceil-1;
		if ((x-x_floor) < epsilon)
			x_ceil = x_floor+1;
		if ((x_ceil-x) < epsilon)
			x_floor = x_ceil-1;

		newVals(i) = (float) bilinearInterpolation(img.at<uint8_t>(y_floor,x_floor),img.at<uint8_t>(y_floor,x_ceil),
								img.at<uint8_t>(y_ceil,x_floor),img.at<uint8_t>(y_ceil,x_ceil),
								x_floor, x_ceil, y_floor, y_ceil, x, y);

	}
	return newVals;
}

/*Eigen::VectorXf warpedValues2(cv::Mat img, Eigen::MatrixXf pixel_coord_ref, 
	Eigen::MatrixXf pixel_coord_float, Eigen::VectorXf valid) {

	Eigen::VectorXf newVals(pixel_coord_float.cols());

	cv::Mat map_x = cv::Mat(img.rows,img.cols, CV_32F, cvScalar(0.));
	cv::Mat map_y = cv::Mat(img.rows,img.cols, CV_32F, cvScalar(0.));
	for (int i=0; i<pixel_coord_ref.cols(); i++) {
			float x = pixel_coord_ref(0,i);
			float y = pixel_coord_ref(1,i);
			map_x.at<float>(y,x) = pixel_coord_float(0,i);
			map_y.at<float>(y,x) = pixel_coord_float(1,i);
	}
	cv::Mat dst;
	cv::remap( img, dst, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0) );
	for (int i=0; i<pixel_coord_ref.cols(); i++) {
		newVals(i) = dst.at<uint8_t>(pixel_coord_ref(1,i),pixel_coord_ref(0,i));
	}
	return newVals;
}*/