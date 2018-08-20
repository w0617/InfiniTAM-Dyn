// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../ITMMeshingEngine.h"

namespace ITMLib
{
	namespace Engine
	{
		template<class TVoxel, class TIndex>
		class ITMMeshingEngine_CUDA : public ITMMeshingEngine < TVoxel, TIndex >
		{
		 public:
		  ITMMeshingEngine_CUDA(long sdfLocalBlockNum);
		};

		template<class TVoxel>
		class ITMMeshingEngine_CUDA<TVoxel, ITMVoxelBlockHash> : public ITMMeshingEngine < TVoxel, ITMVoxelBlockHash >
		{
		private:
			uint *noTriangles_device;
			Vector4s *visibleBlockGlobalPos_device;
			long sdfLocalBlockNum;

		public:
		  	ITMMeshingEngine_CUDA(long sdfLocalBlockNum);
			void MeshScene(ITMMesh *mesh, const ITMScene<TVoxel, ITMVoxelBlockHash> *scene);

			~ITMMeshingEngine_CUDA(void);
		};

		template<class TVoxel>
		class ITMMeshingEngine_CUDA<TVoxel, ITMPlainVoxelArray> : public ITMMeshingEngine < TVoxel, ITMPlainVoxelArray >
		{
		private:
		  long sdfLocalBlockNum;

		public:
			ITMMeshingEngine_CUDA(long sdfLocalBlockNum);
			void MeshScene(ITMMesh *mesh, const ITMScene<TVoxel, ITMPlainVoxelArray> *scene);

			~ITMMeshingEngine_CUDA(void);
		};

	/// \brief Populates 'visibleBlockGlobalPos' with the positions of the blocks which are in use.
	// This kernel is run for every bucket of the hash map.
	__global__ void findAllocatedBlocks(Vector4s *visibleBlockGlobalPos,
										const ITMHashEntry *hashTable,
										int noTotalEntries);

	}
}
