#include "APAP.h"

bool APAP::isBlack(const Mat &img, int x, int y, uchar &b, uchar &g, uchar &r) {
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


uchar APAP::getWarpValue(uchar val1, uchar val2, int weight1, int weight2) {
    return (val1 * weight1 + val2 * weight2) / (weight1 + weight2);
}

void APAP::warpImage(const Mat &image_1, const Mat &img_2, Mat &target) {
    uchar b, g, r;
    uchar b1, g1, r1, b2, g2, r2;
    target = Mat::zeros(height, width, CV_8UC3);
    Mat img_1 = Mat::zeros(width, height, CV_8UC3);

    Rect aaa = Rect(0, 100, image_1.size().width, image_1.size().height);

    image_1.copyTo(img_1(aaa));

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            int weight_left = isBlack(img_1, x, y, b1, g1, r1) ? 0 : 1,
                    weight_right = isBlack(img_2, x, y, b2, g2, r2) ? 0 : 1;
            if (weight_left + weight_right > 0) {
                b = getWarpValue(b1, b2, weight_left, weight_right);
                g = getWarpValue(g1, g2, weight_left, weight_right);
                r = getWarpValue(r1, r2, weight_left, weight_right);
                target.at<Vec3b>(y, x) = Vec3b(b, g, r);
            }
        }
    imshow("APAP target", target);
}

MatrixXd APAP::calculate_Wi_forPoint(double x, double y) {
    const double sigma_squared = sigma * sigma;
    MatrixXd Wi(2 * points.size(), 2 * points.size());
    Wi.setZero();
    for (size_t i = 0; i < points.size(); i++) {
        double u = (double) points[i].x, v = (double) points[i].y;
        double sqr_dist = getSqrDist(x, y, u, v);
        double candidate = exp(-sqr_dist / sigma_squared);
        double omega_i = max(candidate, gamma2);
        Wi(i * 2, i * 2) = omega_i;
        Wi(i * 2 + 1, i * 2 + 1) = omega_i;
    }
    return Wi;
}