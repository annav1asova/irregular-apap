#include <opencv2/opencv.hpp>
#include "MathUtils.h"
#include <random>


using namespace cv;
using namespace std;
using namespace Eigen;

#ifndef THRESHOLD
    #define THRESHOLD 0.1
#endif

class MatchDetector {

    public:
        MatrixXf match;
        MatrixXf inlier;
        vector<KeyPoint> left_keypoint, right_keypoint;


        void detectSiftMatch(const char* img1_path, const char* img2_path);

        void normalizeMatch(Matrix3f &T1, Matrix3f &T2);

        bool singleModelRANSAC(int M);

        void getKeyPoints(const Matrix3f &inv_T1, const Matrix3f &inv_T2);

};

