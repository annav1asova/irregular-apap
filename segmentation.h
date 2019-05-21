#pragma once

#include "MathUtils.h"
#include "GridBox.h"

#include <opencv2/opencv.hpp>
#include <iostream>

struct region {
    // tree data structure
    std::vector<region> childs;
    bool validity; // TODO: have a method for clear the data structure and remove regions with false validity

    // tree for split&merge procedure
    cv::Rect roi;
    MatrixXd Wi;
    Matrix3d Hmg;
    GridBox gbox;

    cv::Mat m;
    cv::Scalar label;
    cv::Mat mask; // for debug. don't use in real cases because it is computationally too heavy.
};

region segment_region(const String& filename);

void getRegionList(region r, vector<region> &listRegion);
