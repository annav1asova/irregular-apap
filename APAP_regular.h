#include "APAP.h"

//void calculate_Wi_Matrices(Mat img, vector<Point2d>& obj, vector<MatrixXd>& vec, region &r); //for tree
class APAP_regular: public APAP {
    public:
        APAP_regular(int C1, int C2) : APAP() {
            this->C1 = C1;
            this->C2 = C2;
        }

        void calculate_Wi_Matrices(Mat img, vector<Point2d> &obj);

        void calculate_CellHomography(MatrixXd &A);

        void ConvertImage(const Mat &img, Mat &target);

    private:
        int C1, C2;
        vector<MatrixXd> Wi_Vector;
        vector<Matrix3d> H_vec;

        void findGrid(int &gx, int &gy, double x, double y, GridBox **grids);
        GridBox** getIndex(const Mat &img, int offset_x, int offset_y);
};