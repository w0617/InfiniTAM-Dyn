// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../ITMSceneReconstructionEngine.h"
#include "../../../../ORUtils/MemoryBlock.h"

#include <queue>

namespace ITMLib
{
	namespace Engine
	{
		/// \brief A snapshot of what blocks were visible at some point in time. Needed by the voxel
		///        decay.
		struct VisibleBlockInfo {
			size_t count;
			size_t frameIdx;
			ORUtils::MemoryBlock<Vector3i> *blockCoords;
		};

		template<class TVoxel, class TIndex>
		class ITMSceneReconstructionEngine_CUDA : public ITMSceneReconstructionEngine < TVoxel, TIndex >
		{};

		// Reconstruction engine for voxel hashing.
		template<class TVoxel>
		class ITMSceneReconstructionEngine_CUDA<TVoxel, ITMVoxelBlockHash> : public ITMSceneReconstructionEngine < TVoxel, ITMVoxelBlockHash >
		{
		private:
			void *allocationTempData_device;
			void *allocationTempData_host;
			unsigned char *entriesAllocType_device;
			Vector4s *blockCoords_device;

			// Keeps track of recent lists of visible block IDs. Used by the voxel decay.
			std::queue<VisibleBlockInfo> frameVisibleBlocks;
			int *lastFreeBlockId_device;
			// Used to avoid data races when deleting elements from the hash table.
			int *locks_device;
			// Used by the full-volume decay code.
			Vector4s *allocatedBlockPositions_device;

			long totalDecayedBlockCount = 0L;
			size_t frameIdx = 0;

			/// \brief Runs a voxel decay process on the blocks specified in `visibleBlockInfo`.
			void PartialDecay(
				ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
				const ITMRenderState *renderState,
				const VisibleBlockInfo &visibleBlockInfo,
				int minAge,
				int maxWeight);

			/// \brief Runs a voxel decay process on the entire volume.
			void FullDecay(
				ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
				const ITMRenderState *renderState,
				int minAge,
				int maxWeight);

		public:
//			WeightParams fusionWeightParams;

			void ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene);

			void AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view, const ITMTrackingState *trackingState,
				const ITMRenderState *renderState, bool onlyUpdateVisibleList = false);

			void IntegrateIntoScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view, const ITMTrackingState *trackingState,
				const ITMRenderState *renderState);

			void Decay(ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
					   const ITMRenderState *renderState,
					   int maxWeight,
					   int minAge,
					   bool forceAllVoxels) override;

			size_t GetDecayedBlockCount() override;

			ITMSceneReconstructionEngine_CUDA(long sdfLocalBlockNum);
			~ITMSceneReconstructionEngine_CUDA(void);
		};

		// Reconstruction engine for plain voxel arrays (vanilla Kinectfusion-style).
		template<class TVoxel>
		class ITMSceneReconstructionEngine_CUDA<TVoxel, ITMPlainVoxelArray> : public ITMSceneReconstructionEngine < TVoxel, ITMPlainVoxelArray >
		{
		public:
			void ResetScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene);

			void AllocateSceneFromDepth(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view, const ITMTrackingState *trackingState,
				const ITMRenderState *renderState, bool onlyUpdateVisibleList = false);

			void IntegrateIntoScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view, const ITMTrackingState *trackingState,
				const ITMRenderState *renderState);

		  void Decay(ITMScene<TVoxel, ITMPlainVoxelArray> *scene,
					 const ITMRenderState *renderState,
					 int maxWeight,
					 int minAge,
					 bool forceAllVoxels) override;

		  size_t GetDecayedBlockCount() override;
		};
	}
}
