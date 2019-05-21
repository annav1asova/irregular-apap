#include "APAP_irregular.h"

void APAP_irregular::calculate_Wi_Matrices(Mat img, vector<Point2d> &obj) {
    points = obj;
    cout << "points size" << points.size();
    for (int i = 0; i < allRegions.size(); i++) {
        Rect cell = allRegions[i].roi;
        int centerX = cell.x + cell.width / 2;
        int centerY = cell.y + cell.height / 2;
        MatrixXd Wi = APAP::calculate_Wi_forPoint(centerX, centerY);
        allRegions[i].Wi = Wi;
    }
    cout << "allreg size" << allRegions.size() << endl;

}


void APAP_irregular::calculate_CellHomography(MatrixXd &A) {
    cout << "calculate cell homography" << endl;

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

GridBox * APAP_irregular::getIndex(int offset_x, int offset_y) {
//													 здесь сетка проектируется и
//													 спроектированная начинает отдельно хранится в виде гридбокса -
//													 структуры у которой есть функция проверки на принадлженость точки многоугольинку

    GridBox *a = new GridBox[allRegions.size()];
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

void APAP_irregular::findGrid(int &gridIndex, double x, double y, GridBox *grids, int sizeGB) {
    for (int i = 0; i < sizeGB; i++) {
        if (grids[i].contains(x, y)) {
            gridIndex = i;
            return;
        }
    }
}


void APAP_irregular::ConvertImage(const Mat &img, Mat &target) {
    cout << "convert image" << endl;

    int Width = img.size().width, Height = img.size().height;
    GridBox *grids = getIndex(x_offset, y_offset);
    target = Mat::zeros(height, width, img.type());
    uchar b, g, r;

    cout << " convert image starting" << endl;

    int gridIndex = -1;

    for (int y = y_offset; y < height; y++) {
        cout << y << " y" << endl;

        for (int x = x_offset; x < width; x++) {
            if (gridIndex < 0 || !allRegions[gridIndex].gbox.contains(x, y)) {
                gridIndex = -1;
                findGrid(gridIndex, x, y, grids, allRegions.size());
            }

            if (gridIndex >= 0) {
                double t_nx, t_ny;
                ConvertCoordinates(x - x_offset, y - y_offset, t_nx, t_ny, allRegions[gridIndex].Hmg.inverse());

                if (t_nx >= 0 && t_nx <= Width && t_ny >= 0 && t_ny <= Height) {
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
        double *verty = grid.verty, *vertx = grid.vertx;
        int i, j;
        Point2d p1, p2;

        for (i = 0, j = 3; i < 4; j = i++) {
            p1 = Point2d(vertx[i], verty[i]);
            p2 = Point2d(vertx[j], verty[j]);
            line(gridMat, p1, p2, Scalar(255, 0, 0), 1, LINE_AA);
        }
    }
//	imshow("warp img 2", target);
    imshow("grids", gridMat);
}

