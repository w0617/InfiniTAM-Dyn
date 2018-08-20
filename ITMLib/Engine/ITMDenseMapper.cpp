// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMDenseMapper.h"

#include "../Objects/ITMRenderState_VH.h"

#include "../ITMLib.h"

using namespace ITMLib::Engine;

template<class TVoxel, class TIndex>
ITMDenseMapper<TVoxel, TIndex>::ITMDenseMapper(const ITMLibSettings *settings)
{
	swappingEngine = NULL;

	switch (settings->deviceType)
	{
	case ITMLibSettings::DEVICE_CPU:
		sceneRecoEngine = new ITMSceneReconstructionEngine_CPU<TVoxel,TIndex>();
		if (settings->useSwapping) swappingEngine = new ITMSwappingEngine_CPU<TVoxel,TIndex>();
		break;
	case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
		sceneRecoEngine = new ITMSceneReconstructionEngine_CUDA<TVoxel,TIndex>(settings->sdfLocalBlockNum);
		if (settings->useSwapping) {
			swappingEngine = new ITMSwappingEngine_CUDA<TVoxel,TIndex>(settings->sdfLocalBlockNum);
		}
#endif
		break;
	case ITMLibSettings::DEVICE_METAL:
#ifdef COMPILE_WITH_METAL
		sceneRecoEngine = new ITMSceneReconstructionEngine_Metal<TVoxel, TIndex>();
		if (settings->useSwapping) swappingEngine = new ITMSwappingEngine_CPU<TVoxel, TIndex>();
#endif
		break;
	}
}

template<class TVoxel, class TIndex>
ITMDenseMapper<TVoxel,TIndex>::~ITMDenseMapper()
{
	delete sceneRecoEngine;
	if (swappingEngine!=NULL) delete swappingEngine;
}

template<class TVoxel, class TIndex>
void ITMDenseMapper<TVoxel,TIndex>::ResetScene(ITMScene<TVoxel,TIndex> *scene)
{
	sceneRecoEngine->ResetScene(scene);
}

template<class TVoxel, class TIndex>
void ITMDenseMapper<TVoxel,TIndex>::ProcessFrame(const ITMView *view, const ITMTrackingState *trackingState, ITMScene<TVoxel,TIndex> *scene, ITMRenderState *renderState)
{
	// allocation
//        cout <<"trackingState->pose_d:  "<<endl<< trackingState->pose_d->GetM() << endl;
	sceneRecoEngine->AllocateSceneFromDepth(scene, view, trackingState, renderState);

	// integration
	sceneRecoEngine->IntegrateIntoScene(scene, view, trackingState, renderState);

	if (swappingEngine != NULL) {
		printf("Swap phase.\n");
		// swapping: CPU -> GPU
		swappingEngine->IntegrateGlobalIntoLocal(scene, renderState);
		// swapping: GPU -> CPU
		swappingEngine->SaveToGlobalMemory(scene, renderState);
		printf("Swap phase done.\n");
	}
}

template<class TVoxel, class TIndex>
void ITMDenseMapper<TVoxel,TIndex>::UpdateVisibleList(const ITMView *view, const ITMTrackingState *trackingState, ITMScene<TVoxel,TIndex> *scene, ITMRenderState *renderState)
{
	sceneRecoEngine->AllocateSceneFromDepth(scene, view, trackingState, renderState, true);
}

template<class TVoxel, class TIndex>
void ITMDenseMapper<TVoxel, TIndex>::Decay(
		ITMScene<TVoxel, TIndex> *scene,
		ITMRenderState *renderState,
		int maxWeight,
		int minAge,
		bool forceAllVoxels
) {
	sceneRecoEngine->Decay(scene, renderState, maxWeight, minAge, forceAllVoxels);
}

template<class TVoxel, class TIndex>
size_t ITMDenseMapper<TVoxel, TIndex>::GetDecayedBlockCount() const {
	return sceneRecoEngine->GetDecayedBlockCount();
}

template class ITMLib::Engine::ITMDenseMapper<ITMVoxel, ITMVoxelIndex>;
