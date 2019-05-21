#include <iostream>
#include "Homography.h"
#include "APAP.h"
#include "APAP_irregular.h"
#include "APAP_regular.h"
#include "MathUtils.h"
#include "segmentation.h"
#include <chrono>
#include <ctime>

using namespace cv;
using namespace std;
using namespace Eigen;

const char* pc_path = "/Users/annavlasova/Desktop/Новая папка/rgb";

string current_path = pc_path;
vector<Point2d> Left, Right;
vector<KeyPoint> left_keypoint, right_keypoint;

int main() {
	auto start = std::chrono::system_clock::now();

	const char* img1_path = "/Users/annavlasova/Desktop/Новая папка/rgb/1.png";
	const char* img2_path = "/Users/annavlasova/Desktop/Новая папка/rgb/2.png";

	region r = segment_region(img1_path);

	vector<region> allRegions;
	getRegionList(r, allRegions);

	cout << allRegions.size();

//	string imagename1 = current_path + "2.png",
//		   imagename2 = current_path + "1.png";
//	Mat img_1 = imread(imagename1), img_2 = imread(imagename2);

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
	getHomography(right_keypoint, left_keypoint, scene, obj, H, A);

	cout << "H : " << H << endl;


//	APAP *apap = new APAP_regular(C1, C2);
	APAP *apap = new APAP_irregular(allRegions);


	cout << "Calculating Weighted matrices..." << endl;
	apap->calculate_Wi_Matrices(img_2, scene);
	cout << "testing..." << endl;
	apap->calculate_CellHomography(A);
	Mat homography_target, display;
	cout << "Converting image 2..." << endl;
	apap->ConvertImage(img_2, homography_target);
	cout << "Converting image 3..." << endl;

	apap->warpImage(img_1, homography_target, display);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	cout << elapsed_seconds.count() << "s\n";

    waitKey(0);

	return 0;
}
