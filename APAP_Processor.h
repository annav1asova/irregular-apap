#pragma once
#include "GridBox.h"
#include "segmentation.h"


//void calculate_Wi_Matrices(Mat img, vector<Point2d>& obj, vector<MatrixXd>& vec, region &r); //for tree

void calculate_Wi_Matrices(Mat img, vector<Point2d>& obj, vector<MatrixXd>& vec);


void calculate_Wi_Matrices(Mat img, vector<Point2d>& obj, vector<region> &allRegions); //cor

vector<Matrix3d> calculate_CellHomography(vector<MatrixXd>& matrices, MatrixXd& A);
//void calculate_CellHomography(region r, MatrixXd& A, int offset_x, int offset_y);

void calculate_CellHomography(vector<region> &allRegions, MatrixXd& A, int offset_x, int offset_y); //cor

void ConvertImage(const Mat& img, Mat& target, vector<Matrix3d> H_vec, int C1, int C2);
void ConvertImage(const Mat& img, Mat& target, vector<region> allReg, int x_offset, int y_offset); // cor

void warpImage(const Mat& img_1, const Mat& img_2, Mat& target);