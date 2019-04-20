#include "APAP_Processor.h"
#include "MathUtils.h"
#include "segmentation.h"
/*
 * \param points �����㼯
 * \param x y x*������
 */
using namespace cv;
using namespace std;
using namespace Eigen;

vector<Point2d> points;
MatrixXd calculate_Wi_forPoint(double x, double y) // оставить
{
	const double sigma_squared = sigma * sigma;
	MatrixXd Wi(2 * points.size(), 2 * points.size());
	Wi.setZero();
	for (size_t i = 0; i < points.size(); i++)
	{
		double u = (double)points[i].x, v = (double)points[i].y;
		double sqr_dist = getSqrDist(x, y, u, v);
		double candidate = exp(-sqr_dist / sigma_squared);
		double omega_i = max(candidate, gamma2);
		Wi(i * 2, i * 2) = omega_i;
		Wi(i * 2 + 1, i * 2 + 1) = omega_i;
	}
	return Wi;
}


void calculate_Wi_Matrices(Mat img, vector<Point2d>& obj, vector<region> &allRegions)
{
	points = obj;
	for (int i = 0; i < allRegions.size(); i++) {
		Rect cell = allRegions[i].roi;
		int centerX = cell.x + cell.width / 2;
		int centerY = cell.y + cell.height / 2;
		MatrixXd Wi = calculate_Wi_forPoint(centerX, centerY);
		allRegions[i].Wi = Wi;
	}
}


void calculate_CellHomography(vector<region> &allRegions, MatrixXd& A, int offset_x, int offset_y) {
	for (size_t i = 0; i < allRegions.size(); i++) {
		MatrixXd WA = allRegions[i].Wi * A;
		Matrix3d H;
		JacobiSVD<MatrixXd> svd(WA, ComputeThinU | ComputeThinV);
		MatrixXd V = svd.matrixV();
		VectorXd h = V.col(V.cols() - 1);
		H << h[0], h[1], h[2],
			h[3], h[4], h[5],
			h[6], h[7], h[8];
		allRegions[i].Hmg = H;
	}
}

GridBox* getIndex(int offset_x, int offset_y, vector<region> allRegions) {
//													 здесь сетка проектируется и
//													 спроектированная начинает отдельно хранится в виде гридбокса -
//													 структуры у которой есть функция проверки на принадлженость точки многоугольинку
	GridBox* a = new GridBox[allRegions.size()];
	for (int H_ind = 0; H_ind < allRegions.size(); H_ind++) {
		double topleftx, toplefty,
			toprightx, toprighty,
			bottomleftx, bottomlefty,
			bottomrightx, bottomrighty;


		region r = allRegions[H_ind];
		Matrix3d H = allRegions[H_ind].Hmg;
		ConvertCoordinates(r.roi.x, r.roi.y, topleftx, toplefty, H);
		ConvertCoordinates(r.roi.x + r.roi.width, r.roi.y, toprightx, toprighty, H);
		ConvertCoordinates(r.roi.x, r.roi.y + r.roi.height, bottomleftx, bottomlefty, H);
		ConvertCoordinates(r.roi.x + r.roi.width, r.roi.y + r.roi.height, bottomrightx, bottomrighty, H);
		GridBox gbox = GridBox(Point2d(topleftx + offset_x, toplefty + offset_y),
							   Point2d(toprightx + offset_x, toprighty + offset_y),
							   Point2d(bottomleftx + offset_x, bottomlefty + offset_y),
							   Point2d(bottomrightx + offset_x, bottomrighty + offset_y));

		a[H_ind] = gbox;
	}
	return a;

}

void findGrid(int &gridIndex, double x, double y, GridBox* grids, int sizeGB) {
	for (int i = 0; i < sizeGB; i++) {
		if (grids[i].contains(x, y)) {
			gridIndex = i;
			return;
		}
	}
}


void ConvertImage(const Mat& img, Mat& target, vector<region> allRegions, int x_offset, int y_offset)
{
	int Width = img.size().width, Height = img.size().height;
//	int x_offset = 0, y_offset = 100;
	GridBox* grids = getIndex(x_offset, y_offset, allRegions);
	target = Mat::zeros(height, width, img.type());
	uchar b, g, r;

	cout  << " convert image starting" << endl;

	int gridIndex = -1;

	for (int y = y_offset; y < height; y++) {
		cout << y << " y" << endl;

		for (int x = x_offset; x < width; x++)
		{
			if (gridIndex < 0 || !allRegions[gridIndex].gbox.contains(x, y)) {
				gridIndex = -1;
				findGrid(gridIndex, x, y, grids, allRegions.size());
			}

			if (gridIndex >= 0) {
				double t_nx, t_ny;
				ConvertCoordinates(x - x_offset, y - y_offset, t_nx, t_ny, allRegions[gridIndex].Hmg.inverse());

				if (t_nx >= 0 && t_nx <= Width && t_ny >= 0 && t_ny <= Height)
				{
					ConvertPoint(img, Width, Height, t_nx, t_ny, b, g, r);
					target.at<Vec3b>(y, x) = Vec3b(b, g, r);
				}
			}

		}
	}
	cout << 3 << endl;

	Mat gridMat = Mat::zeros(height, width, img.type());
	for (int ind = 0; ind < allRegions.size(); ind++) {
		GridBox grid = grids[ind];
		double* verty = grid.verty, *vertx = grid.vertx;
		int i, j;
			Point2d p1, p2;

			for (i = 0, j = 3; i < 4; j = i++)
			{
				p1 = Point2d(vertx[i], verty[i]);
				p2 = Point2d(vertx[j], verty[j]);
				line(gridMat, p1, p2, Scalar(255, 0, 0), 1, LINE_AA);
			}
	}
//	imshow("warp img 2", target);
	imshow("grids", gridMat);
}

bool isBlack(const Mat& img, int x, int y, uchar& b, uchar& g, uchar& r)
{
	if (x >= img.size().width || y >= img.size().height)
		return true;
	Vec3b color = img.at<Vec3b>(y, x);
	b = color[0];
	g = color[1];
	r = color[2];
	if (b == 0 && g == 0 && r == 0)
		return true;
	return false;
}


uchar getWarpValue(uchar val1, uchar val2, int weight1, int weight2)
{
	return (val1 * weight1 + val2 * weight2) / (weight1 + weight2);
}


void warpImage(const Mat& image_1, const Mat& img_2, Mat& target)
{
	uchar b, g, r;
	uchar b1, g1, r1, b2, g2, r2;
	target = Mat::zeros(height, width, CV_8UC3);
	Mat img_1 = Mat::zeros(width, height, CV_8UC3);

	Rect aaa = Rect(0, 100, image_1.size().width, image_1.size().height);

    image_1.copyTo(img_1(aaa));

    for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			int weight_left = isBlack(img_1, x, y, b1, g1, r1) ? 0 : 1,
				weight_right = isBlack(img_2, x, y, b2, g2, r2) ? 0 : 1;
			if (weight_left + weight_right > 0)
			{
				b = getWarpValue(b1, b2, weight_left, weight_right);
				g = getWarpValue(g1, g2, weight_left, weight_right);
				r = getWarpValue(r1, r2, weight_left, weight_right);
				target.at<Vec3b>(y, x) = Vec3b(b, g, r);
			}
		}
	imshow("APAP target", target);
//	waitKey(0);
}

//--------- APAP version (without irregular grid)

void calculate_Wi_Matrices(Mat img, vector<Point2d>& obj, vector<MatrixXd>& vec)
{
	points = obj;
	int Width = img.size().width, Height = img.size().height;
	ArrayXd heightArray = ArrayXd::LinSpaced(C1 + 1, 0, Height - 1),
		widthArray = ArrayXd::LinSpaced(C2 + 1, 0, Width - 1);
	ofstream fout("Wi.txt");
	int count = 0;
	for (int i = 0; i < C1; i++) {
		double y = (heightArray(i) + heightArray(i + 1)) / 2;
		for (int j = 0; j < C2; j++) {
		//	cout << "i = " << i << ", j = " << j << endl;
			double x = (widthArray(j) + widthArray(j + 1)) / 2;
			MatrixXd Wi = calculate_Wi_forPoint(x, y);
			vec.push_back(Wi);
		}
	}

}

vector<Matrix3d> calculate_CellHomography(vector<MatrixXd>& matrices, MatrixXd& A)
{
	vector<Matrix3d> H_vec;
	for (size_t i = 0; i < matrices.size(); i++)
	{
		MatrixXd WA = matrices[i] * A;
		Matrix3d H;
		JacobiSVD<MatrixXd> svd(WA, ComputeThinU | ComputeThinV);
		MatrixXd V = svd.matrixV();
		VectorXd h = V.col(V.cols() - 1);
		H << h[0], h[1], h[2],
			h[3], h[4], h[5],
			h[6], h[7], h[8];
		H_vec.push_back(H);
	}
	return H_vec;
}

GridBox** getIndex(const Mat& img, int C1, int C2, int offset_x, int offset_y, vector<Matrix3d> H_vec) {
													// здесь сетка проектируется и
													// спроектированная начинает отдельно хранится в виде гридбокса -
													// структуры у которой есть функция проверки на принадлженость точки многоугольинку
	GridBox** a = new GridBox*[C2 + 1];
	for (int i = 0; i < C2 + 1; i++)
		a[i] = new GridBox[C1 + 1];

	ArrayXf widthArray = ArrayXf::LinSpaced(C2 + 1, 0, img.size().width),
		heightArray = ArrayXf::LinSpaced(C1 + 1, 0, img.size().height); // 0 ~ C1 - 1, 0 ~ C2 - 1
	double min_x, min_y;
	for (int gy = 0; gy < C1; gy++)
		for (int gx = 0; gx < C2; gx++)
		{
			int H_index = gy * C2 + gx;
			double topleftx, toplefty,
				toprightx, toprighty,
				bottomleftx, bottomlefty,
				bottomrightx, bottomrighty;
			ConvertCoordinates(widthArray[gx], heightArray[gy], topleftx, toplefty, H_vec[H_index]);
			ConvertCoordinates(widthArray[gx + 1], heightArray[gy], toprightx, toprighty, H_vec[H_index]);
			ConvertCoordinates(widthArray[gx], heightArray[gy + 1], bottomleftx, bottomlefty, H_vec[H_index]);
			ConvertCoordinates(widthArray[gx + 1], heightArray[gy + 1], bottomrightx, bottomrighty, H_vec[H_index]);
			GridBox gbox = GridBox(Point2d(topleftx + offset_x, toplefty + offset_y),
									Point2d(toprightx + offset_x, toprighty + offset_y),
									Point2d(bottomleftx + offset_x, bottomlefty + offset_y),
									Point2d(bottomrightx + offset_x, bottomrighty + offset_y));
			a[gy][gx] = gbox;
		}
	return a;
}



void findGrid(int &gx, int &gy, double x, double y, GridBox** grids) {
	for (int grid_x = 0; grid_x < C2; grid_x++)
		for (int grid_y = 0; grid_y < C1; grid_y++)
		{
			if (grids[grid_y][grid_x].contains(x, y))
			{
				gx = grid_x;
				gy = grid_y;
				return;
			}
		}
}


void ConvertImage(const Mat& img, Mat& target, vector<Matrix3d> H_vec, int C1, int C2)
{
	int Width = img.size().width, Height = img.size().height;
	int x_offset = 0, y_offset = 100;
	GridBox** grids = getIndex(img, C1, C2, x_offset, y_offset, H_vec);
	target = Mat::zeros(height, width, img.type());
	uchar b, g, r;

	cout << y_offset << " yf" << height << endl;
	cout << x_offset << " xf" << width << endl;

	for (int y = y_offset; y < height; y++) {
		cout << y << " " << endl;

		for (int x = x_offset; x < width; x++)
		{
			int gx = -1, gy = -1;
			findGrid(gx, gy, x, y, grids);
			if (gx >= 0 && gy >= 0) {
				int H_index = gx + gy * C2;
				double t_nx, t_ny;
				ConvertCoordinates(x - x_offset, y - y_offset, t_nx, t_ny, H_vec[H_index].inverse());

				if (t_nx >= 0 && t_nx <= Width && t_ny >= 0 && t_ny <= Height)
				{
					ConvertPoint(img, Width, Height, t_nx, t_ny, b, g, r);
					target.at<Vec3b>(y, x) = Vec3b(b, g, r);
				}
			}
		}
	}
	cout << 3 << endl;

	Mat gridMat = Mat::zeros(height, width, img.type());
	for (int gy=0;gy<C1;gy++)
		for (int gx = 0; gx < C2; gx++)
		{
			GridBox grid = grids[gy][gx];
			double* verty = grid.verty, *vertx = grid.vertx;
			int i, j;
			Point2d p1, p2;

			for (i = 0, j = 3; i < 4; j = i++)
			{
				p1 = Point2d(vertx[i], verty[i]);
				p2 = Point2d(vertx[j], verty[j]);
				line(gridMat, p1, p2, Scalar(255, 0, 0), 1, LINE_AA);
			}
		}
	imshow("warp img 2", target);
	imshow("grids", gridMat);
}