// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ImageSourceEngine.h"

#include "../Utils/FileUtils.h"

#include <Eigen/Core>

#include <stdio.h>

float focal_length_px = 0.0;
float baseline_m = 0.0;


using namespace InfiniTAM::Engine;

//=================================   Creat Calibration==============================//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 3, 4> ReadProjection(const std::string &expected_label, std::istream &in)
{
  Eigen::Matrix<double, 3, 4> matrix;
  std::string label;
  in >> label;
  assert(expected_label == label && "Unexpected token in calibration file.");

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      in >> matrix(row, col);
    }
  }

  return matrix;
}

void ReadKittiOdometryCalibration(const std::string &fpath,
                                  Eigen::Matrix<double, 3, 4> &left_gray_proj/*,
                                  Eigen::Matrix<double, 3, 4> &right_gray_proj,
                                  Eigen::Matrix<double, 3, 4> &left_color_proj,
                                  Eigen::Matrix<double, 3, 4> &right_color_proj,
                                  Eigen::Matrix4d &velo_to_left_cam*/) {
  static const std::string kLeftGray = "P0:";
//  static const std::string kRightGray = "P1:";
//  static const std::string kLeftColor = "P2:";
//  static const std::string kRightColor = "P3:";
  std::ifstream in(fpath);
  if (! in.is_open()) {
    std::cout<<"Can't open calib.txt. " <<std::endl;
  }

  left_gray_proj = ReadProjection(kLeftGray, in);
//  right_gray_proj = ReadProjection(kRightGray, in);
//  left_color_proj = ReadProjection(kLeftColor, in);
//  right_color_proj = ReadProjection(kRightColor, in);

//  std::string dummy;
//  in >> dummy;
//  if (dummy != "Tr:") {
//    std::getline(in, dummy); // skip to the end of current line
//    in >> dummy;
//    std::cout<<dummy<<std::endl;
//  }

//  for (int row = 0; row < 3; ++row) {
//    for (int col = 0; col < 4; ++col) {
//      in >> velo_to_left_cam(row, col);
//    }
//  }
//  velo_to_left_cam(3, 0) = 0.0;
//  velo_to_left_cam(3, 1) = 0.0;
//  velo_to_left_cam(3, 2) = 0.0;
//  velo_to_left_cam(3, 3) = 1.0;
}

ITMRGBDCalib *CreateItmCalib(const Eigen::Matrix<double, 3, 4> left_cam_proj/*, const Eigen::Vector2i frame_size*/)
{
  ITMRGBDCalib *calib = new ITMRGBDCalib;
  float kMetersToMillimeters = 1.0f / 1000.0f;

  ITMIntrinsics intrinsics;
  float fx = static_cast<float>(left_cam_proj(0, 0));
  float fy = static_cast<float>(left_cam_proj(1, 1));
  float cx = static_cast<float>(left_cam_proj(0, 2));
  float cy = static_cast<float>(left_cam_proj(1, 2));
  float sizeX = 2000;
  float sizeY = 1000;
  intrinsics.SetFrom(fx, fy, cx, cy, sizeX, sizeY);

  calib->intrinsics_rgb = intrinsics;
  calib->intrinsics_d = intrinsics;

  Matrix4f identity; identity.setIdentity();
  calib->trafo_rgb_to_depth.SetFrom(identity);

  calib->disparityCalib.SetFrom(kMetersToMillimeters, 0.0f, ITMDisparityCalib::TRAFO_AFFINE);
  return calib;
}

void GetKittiCalib(const char *fileName, ITMRGBDCalib & calib)
{
    Eigen::Matrix<double, 3, 4> left_gray_proj;
    Eigen::Matrix<double, 3, 4>  right_gray_proj;
    Eigen::Matrix<double, 3, 4>  left_color_proj;
    Eigen::Matrix<double, 3, 4>  right_color_proj;
    Eigen::Matrix4d velo_to_left_gray_cam;
//    Eigen::Vector2i frame_size(imgSize.width, imgSize.height);

    ReadKittiOdometryCalibration(fileName, left_gray_proj/*, right_gray_proj, left_color_proj,             right_color_proj,velo_to_left_gray_cam*/);


    //Stereo camera calibration.  Camera Focal and Baseling
    focal_length_px = left_gray_proj(0, 0);
    baseline_m = 0.537150654273f;

    calib = *CreateItmCalib(left_gray_proj/*, frame_size*/);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

ImageSourceEngine::ImageSourceEngine(const char *calibFilename)
{
    GetKittiCalib(calibFilename, calib);
    //    readRGBDCalib(calibFilename, calib);
}

ImageFileReader::ImageFileReader(const char *calibFilename, const char *rgbImageMask, const char *depthImageMask)
    :ImageSourceEngine(calibFilename)
{    
    strncpy(this->rgbImageMask, rgbImageMask, BUF_SIZE);
	strncpy(this->depthImageMask, depthImageMask, BUF_SIZE);

	currentFrameNo = 0;
	cachedFrameNo = -1;

    cached_rgb = NULL;
	cached_depth = NULL;
}

ImageFileReader::~ImageFileReader()
{
	delete cached_rgb;
	delete cached_depth;
}

void ImageFileReader::loadIntoCache(void)
{
	if (currentFrameNo == cachedFrameNo) return;
	cachedFrameNo = currentFrameNo;

	//TODO> make nicer
	cached_rgb = new ITMUChar4Image(true, false); 
	cached_depth = new ITMShortImage(true, false);

	char str[2048];

    sprintf(str, rgbImageMask, currentFrameNo);
	if (!ReadImageFromFile(cached_rgb, str)) 
	{
		delete cached_rgb; cached_rgb = NULL;
		printf("error reading file '%s'\n", str);
	}

	sprintf(str, depthImageMask, currentFrameNo);
	if (!ReadImageFromFile(cached_depth, str)) 
	{
		delete cached_depth; cached_depth = NULL;
		printf("error reading file '%s'\n", str);
    }
}

bool ImageFileReader::hasMoreImages(void)
{
	loadIntoCache();
	return ((cached_rgb!=NULL)&&(cached_depth!=NULL));
}

void ImageFileReader::getImages(ITMUChar4Image *rgb, ITMShortImage *rawDepth)
{
	bool bUsedCache = false;
    if (cached_rgb != NULL) {
		rgb->SetFrom(cached_rgb, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		delete cached_rgb;
		cached_rgb = NULL;
		bUsedCache = true;
	}
	if (cached_depth != NULL) {
        rawDepth->SetFrom(cached_depth, ORUtils::MemoryBlock<short>::CPU_TO_CPU);
        delete cached_depth;
		cached_depth = NULL;
		bUsedCache = true;
	}

	if (!bUsedCache) {
		char str[2048];

		sprintf(str, rgbImageMask, currentFrameNo);
		if (!ReadImageFromFile(rgb, str)) printf("error reading file '%s'\n", str);

		sprintf(str, depthImageMask, currentFrameNo);
		if (!ReadImageFromFile(rawDepth, str)) printf("error reading file '%s'\n", str);
	}

	++currentFrameNo;
}

Vector2i ImageFileReader::getDepthImageSize(void)
{
	loadIntoCache();
	return cached_depth->noDims;
}

Vector2i ImageFileReader::getRGBImageSize(void)
{
	loadIntoCache();
	if (cached_rgb != NULL) return cached_rgb->noDims;
	return cached_depth->noDims;
}

CalibSource::CalibSource(const char *calibFilename, Vector2i setImageSize, float ratio)
	: ImageSourceEngine(calibFilename)
{
	this->imgSize = setImageSize;
	this->ResizeIntrinsics(calib.intrinsics_d, ratio);
	this->ResizeIntrinsics(calib.intrinsics_rgb, ratio);
}

void CalibSource::ResizeIntrinsics(ITMIntrinsics &intrinsics, float ratio)
{
	intrinsics.projectionParamsSimple.fx *= ratio;
	intrinsics.projectionParamsSimple.fy *= ratio;
	intrinsics.projectionParamsSimple.px *= ratio;
	intrinsics.projectionParamsSimple.py *= ratio;
	intrinsics.projectionParamsSimple.all *= ratio;
}

RawFileReader::RawFileReader(const char *calibFilename, const char *rgbImageMask, const char *depthImageMask, Vector2i setImageSize, float ratio) 
	: ImageSourceEngine(calibFilename)
{
	this->imgSize = setImageSize;
	this->ResizeIntrinsics(calib.intrinsics_d, ratio);
	this->ResizeIntrinsics(calib.intrinsics_rgb, ratio);
	
	strncpy(this->rgbImageMask, rgbImageMask, BUF_SIZE);
	strncpy(this->depthImageMask, depthImageMask, BUF_SIZE);

	currentFrameNo = 0;
	cachedFrameNo = -1;

	cached_rgb = NULL;
	cached_depth = NULL;
}

void RawFileReader::ResizeIntrinsics(ITMIntrinsics &intrinsics, float ratio)
{
	intrinsics.projectionParamsSimple.fx *= ratio;
	intrinsics.projectionParamsSimple.fy *= ratio;
	intrinsics.projectionParamsSimple.px *= ratio;
	intrinsics.projectionParamsSimple.py *= ratio;
	intrinsics.projectionParamsSimple.all *= ratio;
}

void RawFileReader::loadIntoCache(void)
{
	if (currentFrameNo == cachedFrameNo) return;
	cachedFrameNo = currentFrameNo;

	//TODO> make nicer
	cached_rgb = new ITMUChar4Image(imgSize, MEMORYDEVICE_CPU);
	cached_depth = new ITMShortImage(imgSize, MEMORYDEVICE_CPU);

	char str[2048]; FILE *f; bool success = false;

	sprintf(str, rgbImageMask, currentFrameNo);

	f = fopen(str, "rb");
	if (f)
	{
		size_t tmp = fread(cached_rgb->GetData(MEMORYDEVICE_CPU), sizeof(Vector4u), imgSize.x * imgSize.y, f);
		fclose(f);
		if (tmp == (size_t)imgSize.x * imgSize.y) success = true;
	}
	if (!success)
	{
		delete cached_rgb; cached_rgb = NULL;
		printf("error reading file '%s'\n", str);
	}

	sprintf(str, depthImageMask, currentFrameNo); success = false;
	f = fopen(str, "rb");
	if (f)
	{
		size_t tmp = fread(cached_depth->GetData(MEMORYDEVICE_CPU), sizeof(short), imgSize.x * imgSize.y, f);
		fclose(f);
		if (tmp == (size_t)imgSize.x * imgSize.y) success = true;
	}
	if (!success)
	{
		delete cached_depth; cached_depth = NULL;
		printf("error reading file '%s'\n", str);
	}
}


bool RawFileReader::hasMoreImages(void)
{
	loadIntoCache(); 

	return ((cached_rgb != NULL) || (cached_depth != NULL));
}

void RawFileReader::getImages(ITMUChar4Image *rgb, ITMShortImage *rawDepth)
{
	bool bUsedCache = false;

	if (cached_rgb != NULL)
	{
		rgb->SetFrom(cached_rgb, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		delete cached_rgb;
		cached_rgb = NULL;
		bUsedCache = true;
	}

	if (cached_depth != NULL)
    {
		rawDepth->SetFrom(cached_depth, ORUtils::MemoryBlock<short>::CPU_TO_CPU);
        delete cached_depth;
		cached_depth = NULL;
		bUsedCache = true;
	}

	if (!bUsedCache) this->loadIntoCache();

	++currentFrameNo;
}
