#pragma once
#include "GridBox.h"
#include "segmentation.h"

using namespace cv;
using namespace std;
using namespace Eigen;

class APAP {

    public:

        int x_offset = 0, y_offset = 100;

        vector<Point2d> points;

        virtual void calculate_Wi_Matrices(Mat img, vector<Point2d> &obj) {}; //cor

        virtual void calculate_CellHomography(MatrixXd &A) {}; //cor

        virtual void ConvertImage(const Mat &img, Mat &target) {}; // cor

        bool isBlack(const Mat &img, int x, int y, uchar &b, uchar &g, uchar &r);


        uchar getWarpValue(uchar val1, uchar val2, int weight1, int weight2);

        void warpImage(const Mat &image_1, const Mat &img_2, Mat &target);

        MatrixXd calculate_Wi_forPoint(double x, double y);

        APAP() {};
};