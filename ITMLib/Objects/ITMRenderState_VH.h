// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <stdlib.h>

#include "ITMRenderState.h"
#include "../../ORUtils/MemoryBlock.h"

namespace ITMLib
{
	namespace Objects
	{
		/** \brief
		    Stores the render state used by the SceneReconstruction 
			and visualisation engines, as used by voxel hashing.
		*/
		class ITMRenderState_VH : public ITMRenderState
		{
		private:
			MemoryDeviceType memoryType;

			/** A list of "visible entries", that are currently
			being processed by the tracker.
			*/
			ORUtils::MemoryBlock<Vector3i> *visibleBlocks;

			/** A list of "visible entries", that are
			currently being processed by integration
			and tracker. One entry corresponds to a hash table element.
			*/
			ORUtils::MemoryBlock<uchar> *entriesVisibleType;
            
		public:
			/** Number of entries in the live list. */
			int noVisibleBlocks;
            
			ITMRenderState_VH(int noTotalEntries, const Vector2i & imgSize, float vf_min, float vf_max, long sdfLocalBlockNum, MemoryDeviceType memoryType = MEMORYDEVICE_CPU)
				: ITMRenderState(imgSize, vf_min, vf_max, memoryType)
            {
				this->memoryType = memoryType;

				visibleBlocks = new ORUtils::MemoryBlock<Vector3i>(sdfLocalBlockNum, memoryType);
				entriesVisibleType = new ORUtils::MemoryBlock<uchar>(noTotalEntries, memoryType);
				
				noVisibleBlocks = 0;
            }
            
			~ITMRenderState_VH()
            {
				delete visibleBlocks;
				delete entriesVisibleType;
            }

			const Vector3i* GetVisibleBlockPositions() const { return visibleBlocks->GetData(memoryType); }
			Vector3i* GetVisibleBlockPositions() { return visibleBlocks->GetData(memoryType); }

			/** Get the list of "visible entries", that are
			currently processed by integration and tracker.
			*/
			uchar *GetEntriesVisibleType(void) { return entriesVisibleType->GetData(memoryType); }

#ifdef COMPILE_WITH_METAL
			const void* GetVisibleEntryIDs_MB(void) { return visibleEntryIDs->GetMetalBuffer(); }
			const void* GetEntriesVisibleType_MB(void) { return entriesVisibleType->GetMetalBuffer(); }
#endif
		};
	}
} 
