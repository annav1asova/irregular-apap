cmake_minimum_required(VERSION 3.13)
project(APAP_Processor_master)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)


set(CMAKE_CXX_STANDARD 14)
set(VLFEAT_INCLUDE ${PROJECT_SOURCE_DIR}/thirdparty/vlfeat/include)
set(VLFEAT_LIB ${PROJECT_SOURCE_DIR}/thirdparty/vlfeat/lib/libvl.so)

include_directories(${EIGEN3_INCLUDE_DIR}
                    ${VLFEAT_INCLUDE})

add_executable(APAP_Processor_master
                APAP_Processor.cpp
        APAP_Processor.h
        features.txt
        GridBox.cpp
        GridBox.h
        Homography.cpp
        Homography.h
        Main.cpp
        MathUtils.cpp
        MathUtils.h
        stdafx.h
        Wi.txt
        segmentation.cpp
        segmentation.h)

link_directories(${VLFEAT_LIB})

target_link_libraries(APAP_Processor_master
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${VLFEAT_LIB}
        )
