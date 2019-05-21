#include "MatchDetector.h"

void MatchDetector::detectSiftMatch(const char* img1_path, const char* img2_path) {
    Mat img1 = imread(img1_path);
    Mat img2 = imread(img2_path);

    Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();
    Ptr<cv::xfeatures2d::SiftDescriptorExtractor> extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    vector<KeyPoint> key1;
    vector<KeyPoint> key2;
    Mat desc1, desc2;


    detector->detect(img1, key1);
    detector->detect(img2, key2);
    extractor->compute(img1, key1, desc1);
    extractor->compute(img2, key2, desc2);

    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(desc1, desc2, matches);

    match.resize(matches.size(), 6);
    cout << "match count: " << matches.size() << endl;
    for (int i = 0; i < matches.size(); i++) {
        match(i, 0) = key1[matches[i].queryIdx].pt.x;
        match(i, 1) = key1[matches[i].queryIdx].pt.y;
        match(i, 2) = 1;
        match(i, 3) = key2[matches[i].trainIdx].pt.x;
        match(i, 4) = key2[matches[i].trainIdx].pt.y;
        match(i, 5) = 1;
    }
}


// Test if three point are colinear
bool colinearity(const Vector3f &p1, const Vector3f &p2, const Vector3f &p3) {
    if (abs(p1.dot(p2.cross(p3))) < FLT_EPSILON)
        return true;
    else
        return false;
}

Matrix3f rollVector9f(const VectorXf &h) {
    Matrix3f H;
    H << h[0], h[1], h[2],
            h[3], h[4], h[5],
            h[6], h[7], h[8];
    return H;
}

VectorXf unrollMatrix3f(const Matrix3f &H) {
    VectorXf h(9);
    h << H(0, 0), H(0, 1), H(0, 2),
            H(1, 0), H(1, 1), H(1, 2),
            H(2, 0), H(2, 1), H(2, 2);
    return h;
}

void RandomSampling(int m, int N, vector<int> &samples) {
    samples.reserve(m);
    random_device rd;
    mt19937 randomGenerator(rd());
    // too slow
    vector<int> numberBag(N);
    for (int i = 0; i < N; i++)
        numberBag[i] = i;

    int max = static_cast<int>(numberBag.size()-1);
    for (int i = 0; i < m; i++) {
        uniform_int_distribution<> uniformDistribution(0, max);
        int index = uniformDistribution(randomGenerator);
        swap(numberBag[index], numberBag[max]);
        samples[N-1-max] = numberBag[max];
        max--;
    }
}

void fitHomography(MatrixXf pts1, MatrixXf pts2, Matrix3f &H, MatrixXf &A) {
    int psize = pts1.rows();
    A.resize(psize*2, 9);
    for (auto i = 0; i < psize; i++) {
        Vector3f p1 = pts1.row(i);
        Vector3f p2 = pts2.row(i);
        A.row(i*2) << 0, 0, 0, -p1[0], -p1[1], -p1[2], p2[1]*p1[0], p2[1]*p1[1], p2[1]*p1[2];
        A.row(i*2+1) << p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2];
    }

    JacobiSVD<MatrixXf, HouseholderQRPreconditioner> svd(A, ComputeFullV);
    MatrixXf V = svd.matrixV();
    VectorXf h = V.col(V.cols()-1);
    H = rollVector9f(h);
}

bool sampleValidTest(const MatrixXf &pts1, const MatrixXf &pts2) {
    return !(colinearity(pts1.row(1), pts1.row(2), pts1.row(3)) ||
             colinearity(pts1.row(0), pts1.row(1), pts1.row(2)) ||
             colinearity(pts1.row(0), pts1.row(2), pts1.row(3)) ||
             colinearity(pts1.row(0), pts1.row(1), pts1.row(3)) ||
             colinearity(pts2.row(1), pts2.row(2), pts2.row(3)) ||
             colinearity(pts2.row(0), pts2.row(1), pts2.row(2)) ||
             colinearity(pts2.row(0), pts2.row(2), pts2.row(3)) ||
             colinearity(pts2.row(0), pts2.row(1), pts2.row(3)));
}

void filterPointAtInfinity(MatrixXf &pts1, MatrixXf &pts2) {
    int finiteCount = 0;
    for (int i = 0; i < pts1.rows(); i++) {
        if (abs(pts1(i, 2)) > FLT_EPSILON && abs(pts2(i, 2)) > FLT_EPSILON)
            finiteCount++;
    }
    MatrixXf temp_pts1, temp_pts2;
    temp_pts1.resize(finiteCount, pts1.cols());
    temp_pts2.resize(finiteCount, pts2.cols());
    int idx = 0;
    for (int i = 0; i < pts1.rows(); i++) {
        if (abs(pts1(i, 2)) > FLT_EPSILON && abs(pts2(i, 2)) > FLT_EPSILON) {
            temp_pts1.row(idx) = pts1.row(i);
            temp_pts2.row(idx) = pts2.row(i);
            idx++;
        }
    }
    pts1 = temp_pts1;
    pts2 = temp_pts2;
}

void noHomogeneous(MatrixXf &mat) {
    MatrixXf temp;
    if (mat.cols() == 3) {
        temp.resize(mat.rows(), 2);
        temp.col(0).array() = mat.col(0).array()/mat.col(2).array();
        temp.col(1).array() = mat.col(1).array()/mat.col(2).array();
        mat = temp;
    } else
        cout << "toHomogeneous with wrong dimension" << endl;
}

void noHomogeneous(Vector3f &vec) {
    if (abs(vec[2]) < FLT_EPSILON)
        cerr << "Divide by 0" << endl;
    vec[0] = vec[0]/vec[2];
    vec[1] = vec[1]/vec[2];
    vec[2] = 1;
}

void computeHomographyResidue(MatrixXf pts1, MatrixXf pts2, const Matrix3f &H, MatrixXf &residue){
    // cross residue
    filterPointAtInfinity(pts1, pts2);
    residue.resize(pts1.rows(), 1);
    MatrixXf Hx1 = (H*pts1.transpose()).transpose();
    MatrixXf invHx2 = (H.inverse()*pts2.transpose()).transpose();

    noHomogeneous(Hx1);
    noHomogeneous(invHx2);
    noHomogeneous(pts1);
    noHomogeneous(pts2);

    MatrixXf diffHx1pts2 = Hx1 - pts2;
    MatrixXf diffinvHx2pts1 = invHx2 - pts1;
    residue = diffHx1pts2.rowwise().squaredNorm() + diffinvHx2pts1.rowwise().squaredNorm();
}


bool MatchDetector::singleModelRANSAC(int M) {
    int maxdegen = 10;
    int dataSize = match.rows();
    int psize = 4;
    MatrixXf x1 = match.block(0, 0, match.rows(), 3);
    MatrixXf x2 = match.block(0, 3, match.rows(), 3);
    vector<int> sample;
    MatrixXf pts1(4, 3);
    MatrixXf pts2(4, 3);
    int maxInlier = -1;
    MatrixXf bestResidue;
    for (int m = 0; m < M; m++) {
        int degencount = 0;
        int isdegen = 1;
        while (isdegen==1 && degencount < maxdegen) {
            degencount ++;
            RandomSampling(psize, dataSize, sample);
            for (int i = 0; i < psize; i++) {
                pts1.row(i) = x1.row(sample[i]);
                pts2.row(i) = x2.row(sample[i]);
            }
            if (sampleValidTest(pts1, pts2))
                isdegen = 0;
        }
        if (isdegen) {
            cout << "Cannot find valid p-subset" << endl;
            return false;
        }
        Matrix3f local_H;
        MatrixXf local_A;
        fitHomography(pts1, pts2, local_H, local_A);

        MatrixXf residue;
        computeHomographyResidue(x1, x2, local_H, residue);
        int inlierCount = (residue.array() < THRESHOLD).count();
        if (inlierCount > maxInlier) {
            maxInlier = inlierCount;
            bestResidue = residue;
        }
    }
    inlier.resize(maxInlier, match.cols());
    int transferCounter = 0;
    for (int i = 0; i < dataSize; i++) {
        if (bestResidue(i) < THRESHOLD) {
            inlier.row(transferCounter) = match.row(i);
            transferCounter++;
        }
    }
    if (transferCounter != maxInlier) {
        cout << "RANSAC result size does not match!!!" << endl;
        return false;
    }
    return true;
}

// ------

void normalizePts(MatrixXf &mat, Matrix3f &T) {
    float cx = mat.col(0).mean();
    float cy = mat.col(1).mean();
    mat.array().col(0) -= cx;
    mat.array().col(1) -= cy;

    float sqrt_2 = sqrt(2);
    float meandist = (mat.array().col(0)*mat.array().col(0) + mat.array().col(1)*mat.array().col(1)).sqrt().mean();
    float scale = sqrt_2/meandist;
    mat.leftCols<2>().array() *= scale;

    T << scale, 0, -scale*cx, 0, scale, -scale*cy, 0, 0, 1;
}

// normalize match respect to "In defense of eight point algorithm"
void MatchDetector::normalizeMatch(Matrix3f &T1, Matrix3f &T2) {
    MatrixXf pts1 = match.leftCols<3>();
    MatrixXf pts2 = match.block(0, 3, match.rows(), 3);
    normalizePts(pts1, T1);
    normalizePts(pts2, T2);
    match.leftCols<3>() = pts1;
    match.block(0, 3, match.rows(), 3) = pts2;
}

void MatchDetector::getKeyPoints(const Matrix3f &inv_T1, const Matrix3f &inv_T2) {
    MatrixXf pts1 = (inv_T1 * inlier.block(0, 0, inlier.rows(), 3).transpose()).transpose();
    MatrixXf pts2 = (inv_T2 * inlier.block(0, 3, inlier.rows(), 3).transpose()).transpose();

    for (int i = 0; i < inlier.rows(); i++) {
        left_keypoint.emplace_back(KeyPoint(Point2f(pts1(i, 0), pts1(i, 1)), 1.f));
        right_keypoint.emplace_back(KeyPoint(Point2f(pts2(i, 0), pts2(i, 1)), 1.f));
    }
}