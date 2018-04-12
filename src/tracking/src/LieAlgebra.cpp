#include "LieAlgebra.h"

Eigen::Matrix3f vec3d_to_skew_symmetric(Eigen::Vector3f vec) {
	Eigen::Matrix3f vec_skew;
	vec_skew << 	0, -vec(2), vec(1),
				vec(2), 0, -vec(0),
				-vec(1), vec(0), 0;
	return vec_skew;
}

/*Eigen::Matrix4f hat_SE3(Eigen::Matrix<float, 6, 1> vec) {
	Eigen::Matrix4f mat;
	mat.topLeftCorner(3,3) = vec3d_to_skew_symmetric(vec.head(3));
	mat.topRightCorner(3,1) = vec.tail(3).transpose();
	mat.bottomRows(1) = Eigen::RowVector4f::Zero();
	return mat;
}*/

Eigen::Matrix4f se3_to_SE3(Eigen::Matrix<float,6,1> vec) {
	Eigen::Vector3f w = vec.head(3);
	Eigen::Vector3f v = vec.tail(3);
	Eigen::Matrix4f T;
	float phi = w.norm();
	
	float epsilon = 0.000000001;
	if (phi < epsilon) {
		T.topLeftCorner(4,4) = Eigen::Matrix4f::Identity();
		T.topRightCorner(3,1) = v;
	}
	else {
		Eigen::Matrix3f skew = vec3d_to_skew_symmetric(w);
		Eigen::Matrix3f topLeft = Eigen::Matrix3f::Identity();
		topLeft += (std::sin(phi)/phi)*skew + ((1-std::cos(phi))/(phi*phi))*skew*skew;
		T.topLeftCorner(3,3) = topLeft;

		Eigen::Matrix3f V = Eigen::Matrix3f::Identity();
		V += ((1-std::cos(phi))/(phi*phi))*skew + ((phi-std::sin(phi))/(phi*phi*phi))*skew*skew;
		T.topRightCorner(3,1) = V*v;

		T.bottomLeftCorner(1,3) = Eigen::RowVector3f::Zero();
		T(3,3) = 1;
	}

	return T;
}

Eigen::Matrix<float,6,1> SE3_to_se3(Eigen::Matrix4f T) {
	Eigen::Matrix<float,6,1> out;
	Eigen::Matrix3f R = T.topLeftCorner(3,3);
	float trace = R.trace();
	float theta = std::acos((trace-1)/2);

	float epsilon = 0.000000001;
	if (theta < epsilon) {
		out.head(3) = Eigen::Vector3f::Zero();
		out.tail(3) = T.topRightCorner(3,1);
		return out;
	}

	Eigen::Matrix3f logR;
	logR = (theta/(2*std::sin(theta)))*(R-R.transpose());

	float a = std::sin(theta)/theta;
	float b = 2*(1-std::cos(theta))/(theta*theta);
	Eigen::Matrix3f invV = Eigen::Matrix3f::Identity();
	invV -= logR/2;
	invV += ((1-(a/b))/(theta*theta))*logR*logR;

	out.tail(3) = invV*T.topRightCorner(3,1);
	out(0) = logR(2,1);
	out(1) = logR(0,2);
	out(2) = logR(1,0);
	return out;

}

Eigen::Matrix4f inverseT(Eigen::Matrix4f T) {
	Eigen::Matrix4f T_inv = Eigen::Matrix4f::Zero();
	Eigen::Matrix3f R = T.topLeftCorner(3,3);
	T_inv.topLeftCorner(3,3) = R.transpose();
	T_inv.topRightCorner(3,1) = -(R.transpose()*T.topRightCorner(3,1));
	T_inv(3,3) = 1;
	/*std::cout << T << std::endl;
	std::cout << T_inv << std::endl;
	std::cout << T*T_inv << std::endl;
	std::cout << T*T.inverse() << std::endl;
	std::cin.get();*/

	return T_inv;
}

/*Eigen::Matrix4f SE3_to_se32(Eigen::Matrix4f T) {
	Eigen::Matrix3f R = T.topLeftCorner(3,3);
	float trace = R.trace();
	float theta = acos((trace-1)/2);

	Eigen::Matrix3f logR;
	logR = (theta/(2*sin(theta)))*(R-R.transpose());

	float a = sin(theta)/theta;
	float b = 2*(1-cos(theta))/(theta*theta);
	Eigen::Matrix3f invV = Eigen::Matrix3f::Identity();
	invV -= logR/2;
	invV += ((1-a/b)/(theta*theta))*logR*logR;

	Eigen::Matrix4f out;
	out.topRightCorner(3,1) = invV*T.topRightCorner(3,1);
	out.topLeftCorner(3,3) = logR;
	out.bottomRows(1) = Eigen::Vector4f::Zero().transpose();
	return out;
}*/