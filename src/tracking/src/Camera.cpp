#include "Camera.h"

Eigen::Vector2f projectPoint(Eigen::Vector3f P) {
	Eigen::Vector3f temp = K*P;
	temp /= temp(2);
	return temp.topRows(2);
}

Eigen::Vector3f inverseProject(Eigen::Vector2f p, float depth) {
	Eigen::Vector3f u;
	u(0) = p(0); u(1) = p(1); u(2) = 1;
	return depth*(invK*u);
}