#include <iostream>
#include "Homography.h"
#include "APAP.h"
#include "APAP_irregular.h"
#include "APAP_regular.h"
#include "MatchDetector.h"
#include "MathUtils.h"
#include "segmentation.h"
#include <chrono>
#include <ctime>

using namespace cv;
using namespace std;
using namespace Eigen;

const char* pc_path = "/Users/annavlasova/Desktop/Новая папка/rgb";

//string current_path = pc_path;
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

	Mat img_1, img_2;

	img_1 = imread(img1_path);
	img_2 = imread(img2_path);

	cout << "img_1 channels " << img_1.channels() << endl;

	vector<DMatch> matches;
	vector<Point2d> obj, scene;

	cout << "SIFT Feature points..." << endl;

	MatchDetector *md = new MatchDetector();
    md->detectSiftMatch(img1_path, img2_path);
    Matrix3f T1, T2;
    md->normalizeMatch(T1, T2);
    const int RANSAC_M = 500;
    md->singleModelRANSAC(RANSAC_M);
    md->getKeyPoints(T1.inverse(), T2.inverse());

	MatrixXd A;
	Matrix3d H;
	vector<MatrixXd> Wi_Vector;
	vector<Matrix3d> H_vectors;
	cout << "Calculating homography..." << endl;
	getHomography(md->right_keypoint, md->left_keypoint, scene, obj, H, A);

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
