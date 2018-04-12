#ifndef INCLUDES_LIEALGEBRA_H_
#define INCLUDES_LIEALGEBRA_H_

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

Eigen::Matrix3f vec3d_to_skew_symmetric(Eigen::Vector3f vec);

//Eigen::Matrix4f hat_SE3(Eigen::Matrix<float, 6, 1> vec);

Eigen::Matrix4f se3_to_SE3(Eigen::Matrix<float, 6, 1> vec);

Eigen::Matrix<float,6,1> SE3_to_se3(Eigen::Matrix4f T);

Eigen::Matrix4f inverseT(Eigen::Matrix4f T);

//Eigen::Matrix4f SE3_to_se32(Eigen::Matrix4f T);

#endif /* INCLUDES_LIEALGEBRA_ */