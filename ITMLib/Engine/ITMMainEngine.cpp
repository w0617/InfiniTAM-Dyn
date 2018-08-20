// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include <thread>
#include <future>
#include "ITMMainEngine.h"

using namespace ITMLib::Engine;

ITMMainEngine::ITMMainEngine(const ITMLibSettings *settings, const ITMRGBDCalib *calib, Vector2i imgSize_rgb, Vector2i imgSize_d)
{
  	bool createMeshingEngine = settings->createMeshingEngine;

	if ((imgSize_d.x == -1) || (imgSize_d.y == -1)) imgSize_d = imgSize_rgb;

	this->settings = settings;

    printf("sdfLocalBlockNum %d\n",settings->sdfLocalBlockNum);
    this->scene = new ITMScene<ITMVoxel, ITMVoxelIndex>(
            &(settings->sceneParams),
            settings->useSwapping,
            settings->deviceType == ITMLibSettings::DEVICE_CUDA ? MEMORYDEVICE_CUDA : MEMORYDEVICE_CPU,
            settings->sdfLocalBlockNum);

    meshingEngine = NULL;
	switch (settings->deviceType)
	{
	case ITMLibSettings::DEVICE_CPU:
		lowLevelEngine = new ITMLowLevelEngine_CPU();
		viewBuilder = new ITMViewBuilder_CPU(calib);
		visualisationEngine = new ITMVisualisationEngine_CPU<ITMVoxel, ITMVoxelIndex>(scene, settings);
		if (createMeshingEngine) {
			// TODO(andrei): Consider also passing the settings object here.
			meshingEngine = new ITMMeshingEngine_CPU<ITMVoxel, ITMVoxelIndex>();
		}
		break;
    case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
		lowLevelEngine = new ITMLowLevelEngine_CUDA();
		viewBuilder = new ITMViewBuilder_CUDA(calib);
        visualisationEngine = new ITMVisualisationEngine_CUDA<ITMVoxel, ITMVoxelIndex>(scene, settings);
        if (createMeshingEngine)
        {
            meshingEngine = new ITMMeshingEngine_CUDA<ITMVoxel, ITMVoxelIndex>(settings->sdfLocalBlockNum);
		}
#endif
		break;
	case ITMLibSettings::DEVICE_METAL:
#ifdef COMPILE_WITH_METAL
		lowLevelEngine = new ITMLowLevelEngine_Metal();
		viewBuilder = new ITMViewBuilder_Metal(calib);
		visualisationEngine = new ITMVisualisationEngine_Metal<ITMVoxel, ITMVoxelIndex>(scene, settings);
		if (createMeshingEngine) meshingEngine = new ITMMeshingEngine_CPU<ITMVoxel, ITMVoxelIndex>();
#endif
		break;
	}

	mesh = NULL;
	if (createMeshingEngine) {
		MemoryDeviceType deviceType = (settings->deviceType == ITMLibSettings::DEVICE_CUDA
		                               ? MEMORYDEVICE_CUDA
		                               : MEMORYDEVICE_CPU);
		mesh = new ITMMesh(deviceType, settings->sdfLocalBlockNum);
	}

	Vector2i trackedImageSize = ITMTrackingController::GetTrackedImageSize(settings, imgSize_rgb, imgSize_d);

	renderState_live = visualisationEngine->CreateRenderState(trackedImageSize);
	renderState_freeview = NULL; //will be created by the visualisation engine

	denseMapper = new ITMDenseMapper<ITMVoxel, ITMVoxelIndex>(settings);
	denseMapper->ResetScene(scene);

	imuCalibrator = new ITMIMUCalibrator_iPad();
	tracker = ITMTrackerFactory<ITMVoxel, ITMVoxelIndex>::Instance().Make(trackedImageSize, settings, lowLevelEngine, imuCalibrator, scene);
	trackingController = new ITMTrackingController(tracker, visualisationEngine, lowLevelEngine, settings);

	trackingState = trackingController->BuildTrackingState(trackedImageSize);
	tracker->UpdateInitialPose(trackingState);

	view = NULL; // will be allocated by the view builder

	fusionActive = true;
	mainProcessingActive = true;
}

ITMMainEngine::~ITMMainEngine()
{
	delete renderState_live;
	if (renderState_freeview!=NULL) delete renderState_freeview;

	delete scene;

	delete denseMapper;
	delete trackingController;

	delete tracker;
	delete imuCalibrator;

	delete lowLevelEngine;
	delete viewBuilder;

	delete trackingState;
	if (view != NULL) delete view;

	delete visualisationEngine;

	if (meshingEngine != NULL) delete meshingEngine;

	if (mesh != NULL) delete mesh;
}

ITMMesh* ITMMainEngine::UpdateMesh(void)
{
	if (mesh != NULL) meshingEngine->MeshScene(mesh, scene);
	return mesh;
}

void ITMMainEngine::SaveSceneToMesh(const char *objFileName)
{
	if (mesh == NULL) {
		fprintf(stderr,
				"Warning: the mesh is NULL so it can't be saved to the file %s.",
				objFileName);
      return;
    }

	std::string fname(objFileName);

	if (!write_result.valid() ||
		 write_result.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready
	) {
		// TODO(andrei): This seems to be memory-intensive. One could maybe also do this in a
		// streaming manner, instead of running marching cubes on everything at once.
		meshingEngine->MeshScene(mesh, scene);

		// Mesh generation is fast (less than a second), but writing stuff to the disk can take
		// minutes, so we do it asynchronously.
		write_result = std::async(std::launch::async, [=] {
			mesh->WriteOBJ(fname.c_str());
			printf(" >>> Async mesh writing completed OK.\n");
		});
	}
	else {
		printf("Please wait until the previous write is finished...\n");
	}

}

void ITMMainEngine::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage, ITMIMUMeasurement *imuMeasurement)
{
	// prepare image and turn it into a depth image
	if (imuMeasurement==NULL) {
		viewBuilder->UpdateView(&view, rgbImage, rawDepthImage, settings->useBilateralFilter, settings->modelSensorNoise);
	}
	else {
		viewBuilder->UpdateView(&view, rgbImage, rawDepthImage, settings->useBilateralFilter, imuMeasurement);
	}

	if (!mainProcessingActive) return;

	// tracking
	trackingController->Track(trackingState, view);

	// fusion
	if (fusionActive) {

        printf("===================== Draw Mapper ======================\n");
		denseMapper->ProcessFrame(view, trackingState, scene, renderState_live);

        if( settings->decayParams.enabled ) {
            printf("===================== Decay Mapper =====================\n");
            denseMapper->Decay(scene, renderState_live, settings->decayParams.max_decay_weight,
                               settings->decayParams.min_decay_age, false);
        }
	}

	// raycast to renderState_live for tracking and free visualisation
	trackingController->Prepare(trackingState, view, renderState_live);
}

Vector2i ITMMainEngine::GetImageSize(void) const
{
	return renderState_live->raycastImage->noDims;
}

void ITMMainEngine::GetImage(ITMUChar4Image *out, ITMFloatImage *outFloat, GetImageType getImageType,
							 ITMPose *pose, ITMIntrinsics *intrinsics)
{
	if (view == NULL) return;

	if (nullptr != out) {
		out->Clear();
	}
	if (nullptr != outFloat) {
		outFloat->Clear();
	}

	auto noDims = (nullptr != out) ? out->noDims : outFloat->noDims;

	switch (getImageType)
	{
	case ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_RGB:
		out->ChangeDims(view->rgb->noDims);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA) {
			out->SetFrom(view->rgb, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		}
		else {
			out->SetFrom(view->rgb, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		}
		break;

	case ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_DEPTH:
		out->ChangeDims(view->depth->noDims);
		if (settings->trackerType==ITMLib::Objects::ITMLibSettings::TRACKER_WICP)
		{
			if (settings->deviceType == ITMLibSettings::DEVICE_CUDA) view->depthUncertainty->UpdateHostFromDevice();
			ITMVisualisationEngine<ITMVoxel, ITMVoxelIndex>::WeightToUchar4(out, view->depthUncertainty);
		}
		else
		{
			if (settings->deviceType == ITMLibSettings::DEVICE_CUDA) view->depth->UpdateHostFromDevice();
			ITMVisualisationEngine<ITMVoxel, ITMVoxelIndex>::DepthToUchar4(out, view->depth);
		}

		break;

	case ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST:
	{
		ORUtils::Image<Vector4u> *srcImage = renderState_live->raycastImage;
		out->ChangeDims(srcImage->noDims);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA) {
			out->SetFrom(srcImage, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		}
		else {
			out->SetFrom(srcImage, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		}
		break;
	}

	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_SHADED:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_DEPTH_WEIGHT:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_DEPTH:
	{
		IITMVisualisationEngine::RenderImageType type = IITMVisualisationEngine::RENDER_SHADED_GREYSCALE;
		if (getImageType == ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME) {
			type = IITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME;
		}
		else if (getImageType == ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL) {
			type = IITMVisualisationEngine::RENDER_COLOUR_FROM_NORMAL;
		}
		else if (getImageType == ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_DEPTH_WEIGHT) {
			type = IITMVisualisationEngine::RENDER_COLOUR_FROM_DEPTH_WEIGHT;
		}
		else if (getImageType == ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_DEPTH) {
			type = IITMVisualisationEngine::RENDER_DEPTH_MAP;
		}
		if (nullptr == renderState_freeview) {
			renderState_freeview = visualisationEngine->CreateRenderState(noDims);
		}

		// This renders the free camera view. It uses raycasting.
		visualisationEngine->FindVisibleBlocks(pose, intrinsics, renderState_freeview);
		visualisationEngine->CreateExpectedDepths(pose, intrinsics, renderState_freeview);
		visualisationEngine->RenderImage(pose, intrinsics, renderState_freeview,
										 renderState_freeview->raycastImage,
										 renderState_freeview->raycastFloatImage,
										 type);

		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA) {
			// Depth is rendered as float, the rest, as RGBA uchars.
			if (getImageType == ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_DEPTH) {
				outFloat->SetFrom(renderState_freeview->raycastFloatImage,
								   ORUtils::MemoryBlock<float>::CUDA_TO_CPU);
			}
			else {
				out->SetFrom(renderState_freeview->raycastImage,
							 ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
			}
		}
		else {
			// depth rendering is unsupported in CPU mode
			out->SetFrom(renderState_freeview->raycastImage, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		}
		break;
	}

	case ITMMainEngine::InfiniTAM_IMAGE_UNKNOWN:
		break;
	};
}

void ITMMainEngine::turnOnIntegration() { fusionActive = true; }
void ITMMainEngine::turnOffIntegration() { fusionActive = false; }
void ITMMainEngine::turnOnMainProcessing() { mainProcessingActive = true; }
void ITMMainEngine::turnOffMainProcessing() { mainProcessingActive = false; }
