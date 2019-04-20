#include <iostream>
#include "Homography.h"
#include "APAP_Processor.h"
#include "MathUtils.h"
#include "segmentation.h"
#include <chrono>
#include <ctime>

using namespace cv;
using namespace std;
using namespace Eigen;
//const char* surface_path = "C:\\Users\\atsst\\Pictures\\Saved Pictures\\";
//const char* pc_path = "C:\\Users\\Administrator\\Pictures\\Saved Pictures\\opencv\\";

const char* pc_path = "/Users/annavlasova/Desktop/Новая папка/rgb";

string current_path = pc_path;
vector<Point2d> Left, Right;
vector<KeyPoint> left_keypoint, right_keypoint;

int main() {
	auto start = std::chrono::system_clock::now();

	region r = segment_region();

	vector<region> allRegions;
	getRegionList(r, allRegions);

	cout << allRegions.size();

//	string imagename1 = current_path + "2.png",
//		   imagename2 = current_path + "1.png";
//	Mat img_1 = imread(imagename1), img_2 = imread(imagename2);

	const char* img1_path = "/Users/annavlasova/Desktop/Новая папка/rgb/1.png";
	const char* img2_path = "/Users/annavlasova/Desktop/Новая папка/rgb/2.png";

	Mat img_1, img_2;

	img_1 = imread(img1_path);
	img_2 = imread(img2_path);

//	cout << "img1 channels " << img1.channels() << endl;
	cout << "img_1 channels " << img_1.channels() << endl;

	vector<DMatch> matches;
	//vector<KeyPoint> keypoints_1, keypoints_2;
	//vector<KeyPoint> keyPoints_obj, keyPoints_scene;
	vector<Point2d> obj, scene;//������


	cout << "SIFT Feature points..." << endl;

	/*
	findFeaturePointsWithSIFT(img_1, img_2, keypoints_1, keypoints_2, matches);
	cout << "Using RANSAC filtering matches..." << endl;
	RANSAC_filter(img_1, img_2, keypoints_1, keypoints_2, keyPoints_obj, keyPoints_scene, matches);*/
	ifstream fin("/Users/annavlasova/Downloads/APAP-Processor-master/features2.txt");
	double x1, y1, x2, y2;
	for (int i = 0; i < 468; i++)
	{
		fin >> x1 >> y1 >> x2 >> y2;
//		cout << x1 << "," << y1 << "\t" << x2 << "," << y2 << endl;
		left_keypoint.emplace_back(KeyPoint(Point2f(x1, y1), 1.f));
		right_keypoint.emplace_back(KeyPoint(Point2f(x2, y2), 1.f));
	}

	MatrixXd A;
	Matrix3d H;
	vector<MatrixXd> Wi_Vector;
	vector<Matrix3d> H_vectors;
	cout << "Calculating homography..." << endl;
	//getHomography(keyPoints_scene, keyPoints_obj, scene, obj, H, A);
	getHomography(right_keypoint, left_keypoint, scene, obj, H, A);

	cout << "H : " << H << endl;

//	cout << "Calculating Weighted matrices..." << endl;
//	calculate_Wi_Matrices(img_2, scene, Wi_Vector);
//	cout << "testing..." << endl;
//	H_vectors = calculate_CellHomography(Wi_Vector, A);
//	Mat homography_target, display;
//	cout << "Converting image 2..." << endl;
//	ConvertImage(img_2, homography_target, H_vectors, C1, C2);
//	cout << "Converting image 3..." << endl;
//
//	warpImage(img_1, homography_target, display);


	cout << "Calculating Weighted matrices..." << endl;
	calculate_Wi_Matrices(img_2, scene, allRegions);
	cout << "testing..." << endl;
    int x_offset = 0, y_offset = 100;

    calculate_CellHomography(allRegions, A, x_offset, y_offset);
	Mat homography_target, display;
	cout << "Converting image 2..." << endl;
	ConvertImage(img_2, homography_target, allRegions, x_offset, y_offset);
	cout << "Converting image 3..." << endl;

	warpImage(img_1, homography_target, display);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	cout << elapsed_seconds.count() << "s\n";

    waitKey(0);

	return 0;
}
