// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include <iostream>
#include "ITMMeshingEngine_CUDA.h"
#include "../../DeviceAgnostic/ITMMeshingEngine.h"
#include "ITMCUDAUtils.h"

#include "../../../../ORUtils/CUDADefines.h"

template<class TVoxel>
__global__ void meshScene_device(ITMMesh::Triangle *triangles, unsigned int *noTriangles_device, float factor, int noTotalEntries,
	int noMaxTriangles, const Vector4s *visibleBlockGlobalPos, const TVoxel *localVBA, const ITMHashEntry *hashTable);

using namespace ITMLib::Engine;

template<class TVoxel>
ITMMeshingEngine_CUDA<TVoxel,ITMVoxelBlockHash>::ITMMeshingEngine_CUDA(long sdfLocalBlockNum)
	: sdfLocalBlockNum(sdfLocalBlockNum)
{
	ITMSafeCall(cudaMalloc((void**)&visibleBlockGlobalPos_device, sdfLocalBlockNum * sizeof(Vector4s)));
	ITMSafeCall(cudaMalloc((void**)&noTriangles_device, sizeof(unsigned int)));
}

template<class TVoxel>
ITMMeshingEngine_CUDA<TVoxel,ITMVoxelBlockHash>::~ITMMeshingEngine_CUDA(void)
{
	ITMSafeCall(cudaFree(visibleBlockGlobalPos_device));
	ITMSafeCall(cudaFree(noTriangles_device));
}

/// \brief Hacky operator for easily displaying CUDA dim3 objects.
std::ostream& operator<<(std::ostream &out, dim3 dim) {
	out << "[" << dim.x << ", " << dim.y << ", " << dim.z << "]";
	return out;
}

template<class TVoxel>
void ITMMeshingEngine_CUDA<TVoxel, ITMVoxelBlockHash>::MeshScene(ITMMesh *mesh, const ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
{
	// TODO-LOW(andrei): This doesn't work if swapping is enabled. That is, it only saves the active
	// mesh, and doesn't attempt to somehow stream all the blocks which have been swapped out to
	// RAM. (It would *not* be trivial to extend the meshing engine to support this IMHO.)

	ITMMesh::Triangle *triangles = mesh->triangles->GetData(MEMORYDEVICE_CUDA);
	const TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	const ITMHashEntry *hashTable = scene->index.GetEntries();

	int noMaxTriangles = mesh->noMaxTriangles, noTotalEntries = scene->index.noTotalEntries;
	float factor = scene->sceneParams->voxelSize;

	ITMSafeCall(cudaMemset(noTriangles_device, 0, sizeof(unsigned int)));
	ITMSafeCall(cudaMemset(visibleBlockGlobalPos_device, 0, sizeof(Vector4s) * sdfLocalBlockNum));

	{ // identify used voxel blocks
		dim3 cudaBlockSize(256); 
		dim3 gridSize((int)ceil((float)noTotalEntries / (float)cudaBlockSize.x));
		findAllocatedBlocks << <gridSize, cudaBlockSize >> >(visibleBlockGlobalPos_device, hashTable, noTotalEntries);
	}

	{ // mesh used voxel blocks
		dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
		dim3 gridSize(sdfLocalBlockNum / 16, 16);

		meshScene_device<TVoxel> << <gridSize, cudaBlockSize >> >(
				triangles,
				noTriangles_device,
				factor,
				noTotalEntries,
				noMaxTriangles,
				visibleBlockGlobalPos_device,
				localVBA,
				hashTable);

		ITMSafeCall(cudaMemcpy(
				&mesh->noTotalTriangles,
				noTriangles_device,
				1 * sizeof(unsigned int),
				cudaMemcpyDeviceToHost));
		printf("Meshing done: %d/%d triangles in mesh.\n", mesh->noTotalTriangles, mesh->noMaxTriangles);
	}
}

template<class TVoxel>
ITMMeshingEngine_CUDA<TVoxel,ITMPlainVoxelArray>::ITMMeshingEngine_CUDA(long sdfLocalBlockNum)
		: sdfLocalBlockNum(sdfLocalBlockNum)
{}

template<class TVoxel>
ITMMeshingEngine_CUDA<TVoxel,ITMPlainVoxelArray>::~ITMMeshingEngine_CUDA(void) 
{}

template<class TVoxel>
void ITMMeshingEngine_CUDA<TVoxel, ITMPlainVoxelArray>::MeshScene(ITMMesh *mesh, const ITMScene<TVoxel, ITMPlainVoxelArray> *scene)
{}

__global__ void ITMLib::Engine::findAllocatedBlocks(
		Vector4s *visibleBlockGlobalPos,
		const ITMHashEntry *hashTable,
		int noTotalEntries
) {
	int entryId = threadIdx.x + blockIdx.x * blockDim.x;
	if (entryId > noTotalEntries - 1) return;

	const ITMHashEntry &currentHashEntry = hashTable[entryId];

	// If this bucket is not unused (ptr < -1), and not swapped out (ptr == -1), we are interested
	// in it in the next stage.
	if (currentHashEntry.ptr >= 0) {
		// visibleBlockGlobalPos maps each VBA entry to the block's position in 3D. If a VBA is not
		// referenced, its 'w' stays 0. Neat!
		visibleBlockGlobalPos[currentHashEntry.ptr] = Vector4s(
				currentHashEntry.pos.x, currentHashEntry.pos.y, currentHashEntry.pos.z, 1);
	}
}

template<class TVoxel>
__global__ void meshScene_device(ITMMesh::Triangle *triangles, unsigned int *noTriangles_device, float factor, int noTotalEntries, 
	int noMaxTriangles, const Vector4s *visibleBlockGlobalPos, const TVoxel *localVBA, const ITMHashEntry *hashTable)
{
	const Vector4s globalPos_4s = visibleBlockGlobalPos[blockIdx.x + gridDim.x * blockIdx.y];

	if (globalPos_4s.w == 0) return;

	Vector3i globalPos = Vector3i(globalPos_4s.x, globalPos_4s.y, globalPos_4s.z) * SDF_BLOCK_SIZE;

	Vector3f vertList[12];

	Vector3i localPos = Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);
	int cubeIndex = buildVertList(vertList, globalPos, localPos, localVBA, hashTable);

	if (cubeIndex < 0) return;

	for (int i = 0; triangleTable[cubeIndex][i] != -1; i += 3)
	{
		int triangleId = atomicAdd(noTriangles_device, 1);

		if (triangleId < noMaxTriangles - 1)
		{
			Vector3f p0 = vertList[triangleTable[cubeIndex][i]];
			Vector3f p1 = vertList[triangleTable[cubeIndex][i + 1]];
			Vector3f p2 = vertList[triangleTable[cubeIndex][i + 2]];
			triangles[triangleId].p0 = p0 * factor;
			triangles[triangleId].p1 = p1 * factor;
			triangles[triangleId].p2 = p2 * factor;

			Vector3f c0 =
					VoxelColorReader<TVoxel::hasColorInformation, TVoxel, ITMVoxelBlockHash>::interpolate3(
							localVBA,
							hashTable,
                            p0);
			Vector3f c1 =
					VoxelColorReader<TVoxel::hasColorInformation, TVoxel, ITMVoxelBlockHash>::interpolate3(
							localVBA,
							hashTable,
                            p1);
			Vector3f c2 =
					VoxelColorReader<TVoxel::hasColorInformation, TVoxel, ITMVoxelBlockHash>::interpolate3(
							localVBA,
							hashTable,
                            p2);

			triangles[triangleId].c0 = c0;
			triangles[triangleId].c1 = c1;
			triangles[triangleId].c2 = c2;
		}
	}
}

template class ITMLib::Engine::ITMMeshingEngine_CUDA<ITMVoxel, ITMVoxelIndex>;
