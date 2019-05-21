#pragma once
#include "GridBox.h"
#include "segmentation.h"
#include "APAP.h"


//void calculate_Wi_Matrices(Mat img, vector<Point2d>& obj, vector<MatrixXd>& vec, region &r); //for tree
class APAP_irregular: public APAP {
//    void calculate_Wi_Matrices(Mat img, vector<Point2d> &obj, vector<MatrixXd> &vec);
public:
    APAP_irregular(vector<region> allRegions) : APAP() {
        this->allRegions = allRegions;
    }

    void calculate_Wi_Matrices(Mat img, vector<Point2d> &obj); //cor

//    vector<Matrix3d> calculate_CellHomography(vector<MatrixXd> &matrices, MatrixXd &A);
//void calculate_CellHomography(region r, MatrixXd& A, int offset_x, int offset_y);

    void calculate_CellHomography(MatrixXd &A); //cor

//    void ConvertImage(const Mat &img, Mat &target, vector<Matrix3d> H_vec, int C1, int C2);

    void ConvertImage(const Mat &img, Mat &target); // cor

//    void warpImage(const Mat &img_1, const Mat &img_2, Mat &target);

    void findGrid(int &gridIndex, double x, double y, GridBox *grids, int sizeGB);
    GridBox *getIndex(int offset_x, int offset_y) ;


//private:
    vector<region> allRegions;
};