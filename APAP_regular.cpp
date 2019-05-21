#include "APAP_regular.h"

//--------- APAP version (without irregular grid)


void APAP_regular::calculate_Wi_Matrices(Mat img, vector<Point2d> &obj) {
    cout << "calculate wi matrices";
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
            Wi_Vector.push_back(Wi);
        }
    }
}

void APAP_regular::calculate_CellHomography(MatrixXd &A) {
    cout << "calculate cell homo " << Wi_Vector.size()<< endl;

    for (size_t i = 0; i < Wi_Vector.size(); i++) {
        MatrixXd WA = Wi_Vector[i] * A;
        Matrix3d H;
        JacobiSVD<MatrixXd> svd(WA, ComputeThinU | ComputeThinV);
        MatrixXd V = svd.matrixV();
        VectorXd h = V.col(V.cols() - 1);
        H << h[0], h[1], h[2],
             h[3], h[4], h[5],
             h[6], h[7], h[8];
        H_vec.push_back(H);
    }
    cout << "H-vec size" << H_vec.size() << endl;
}

GridBox** APAP_regular::getIndex(const Mat &img, int offset_x, int offset_y) {
        // здесь сетка проектируется и
        // спроектированная начинает отдельно хранится в виде гридбокса -
        // структуры у которой есть функция проверки на принадлженость точки многоугольинку
    cout << "get index start" << endl;

    cout << C1 << " " << C2 << "Hvec size" << H_vec.size() << endl;


    GridBox **a = new GridBox *[C2 + 1];
    for (int i = 0; i < C2 + 1; i++)
        a[i] = new GridBox[C1 + 1];

    ArrayXf widthArray = ArrayXf::LinSpaced(C2 + 1, 0, img.size().width),
            heightArray = ArrayXf::LinSpaced(C1 + 1, 0, img.size().height); // 0 ~ C1 - 1, 0 ~ C2 - 1

    double min_x, min_y;
    for (int gy = 0; gy < C1; gy++) {
        for (int gx = 0; gx < C2; gx++) {
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
   }
    return a;
}


void APAP_regular::findGrid(int &gx, int &gy, double x, double y, GridBox **grids) {
    for (int grid_x = 0; grid_x < C2; grid_x++) {
        for (int grid_y = 0; grid_y < C1; grid_y++) {
            if (grids[grid_y][grid_x].contains(x, y)) {
                gx = grid_x;
                gy = grid_y;
                return;
            }
        }
    }
}

void APAP_regular::ConvertImage(const Mat &img, Mat &target) {
    int Width = img.size().width, Height = img.size().height;
    GridBox **grids = getIndex(img, x_offset, y_offset);
    target = Mat::zeros(height, width, img.type());

    uchar b, g, r;
    cout << y_offset << " yf" << height << endl;
    cout << x_offset << " xf" << width << endl;

    for (int y = y_offset; y < height; y++) {
        cout << y << " " << endl;

        for (int x = x_offset; x < width; x++) {
            int gx = -1, gy = -1;
            findGrid(gx, gy, x, y, grids);
            if (gx >= 0 && gy >= 0) {
                int H_index = gx + gy * C2;
                double t_nx, t_ny;
                ConvertCoordinates(x - x_offset, y - y_offset, t_nx, t_ny, H_vec[H_index].inverse());

                if (t_nx >= 0 && t_nx <= Width && t_ny >= 0 && t_ny <= Height) {
                    ConvertPoint(img, Width, Height, t_nx, t_ny, b, g, r);
                    target.at<Vec3b>(y, x) = Vec3b(b, g, r);
                }
            }
        }
    }
    cout << 3 << endl;

    Mat gridMat = Mat::zeros(height, width, img.type());
    for (int gy = 0; gy < C1; gy++)
        for (int gx = 0; gx < C2; gx++) {
            GridBox grid = grids[gy][gx];
            double *verty = grid.verty, *vertx = grid.vertx;
            int i, j;
            Point2d p1, p2;

            for (i = 0, j = 3; i < 4; j = i++) {
                p1 = Point2d(vertx[i], verty[i]);
                p2 = Point2d(vertx[j], verty[j]);
                line(gridMat, p1, p2, Scalar(255, 0, 0), 1, LINE_AA);
            }
        }
//    imshow("warp img 2", target);
    imshow("grids", gridMat);
}

