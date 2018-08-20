// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Objects/ITMRGBDCalib.h"
#include <Eigen/Core>
#include <iostream>

#include <stdio.h>

namespace ITMLib
{
	namespace Objects
	{
		bool readIntrinsics(std::istream & src, ITMIntrinsics & dest);
		bool readIntrinsics(const char *fileName, ITMIntrinsics & dest);
		bool readExtrinsics(std::istream & src, ITMExtrinsics & dest);
		bool readExtrinsics(const char *fileName, ITMExtrinsics & dest);
		bool readDisparityCalib(std::istream & src, ITMDisparityCalib & dest);
		bool readDisparityCalib(const char *fileName, ITMDisparityCalib & dest);
		bool readRGBDCalib(std::istream & src, ITMRGBDCalib & dest);
        bool readRGBDCalib(const char *fileName, ITMRGBDCalib & dest);

        bool readRGBDCalib(const char *rgbIntrinsicsFile, const char *depthIntrinsicsFile, const char *disparityCalibFile, const char *extrinsicsFile, ITMRGBDCalib & dest);

//        Eigen::Matrix<double, 3, 4> ReadProjection(std::string &expected_label, std::istream &in);
//        void ReadKittiOdometryCalibration(const std::string &fpath,
//                                          Eigen::Matrix<double, 3, 4> &left_gray_proj,
//                                          Eigen::Matrix<double, 3, 4> &right_gray_proj,
//                                          Eigen::Matrix<double, 3, 4> &left_color_proj,
//                                          Eigen::Matrix<double, 3, 4> &right_color_proj,
//                                          Eigen::Matrix4d &velo_to_left_cam) ;
//        ITMRGBDCalib *CreateItmCalib(const Eigen::Matrix<double, 3, 4> &left_cam_proj,
//                                                      const Eigen::Vector2i &frame_size);
    }
}


