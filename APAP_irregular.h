#include "APAP.h"

class APAP_irregular: public APAP {
    public:
        APAP_irregular(vector<region> allRegions) : APAP() {
            this->allRegions = allRegions;
        }

        void calculate_Wi_Matrices(Mat img, vector<Point2d> &obj);

        void calculate_CellHomography(MatrixXd &A);

        void ConvertImage(const Mat &img, Mat &target);

    private:
        void findGrid(int &gridIndex, double x, double y, GridBox *grids, int sizeGB);
        GridBox *getIndex(int offset_x, int offset_y) ;

        vector<region> allRegions;
};