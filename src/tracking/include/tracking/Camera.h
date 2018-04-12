#ifndef INCLUDES_CAMERA_H_
#define INCLUDES_CAMERA_H_

#include <Eigen/Dense>

const float Fx = 200;
const float Fy = 200;
const float Cx = 400;
const float Cy = 400;

const Eigen::Matrix3f K((Eigen::Matrix3f() << Fx, 0, Cx, 0, Fy, Cy, 0, 0, 1).finished());

const Eigen::Matrix3f invK((Eigen::Matrix3f() << 1/Fx, 0, -Cx/Fx, 0, 1/Fy, -Cy/Fy, 0, 0, 1).finished());

Eigen::Vector2f projectPoint(Eigen::Vector3f P);

Eigen::Vector3f inverseProject(Eigen::Vector2f p, float depth);

#endif /* INCLUDES_CAMERA_ */