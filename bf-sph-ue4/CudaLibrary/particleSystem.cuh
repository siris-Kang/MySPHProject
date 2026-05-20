/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

extern "C"
{
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, int size);
    void freeArray(void *devPtr);

    void threadSync();

    void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);
    //void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


    void setParameters(SimParams *hostParams);

    void integrateSystem(float *pos,
                         float *vel,
                         float deltaTime,
                         uint numParticles);

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles);

    cudaError_t reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     uint   numParticles,
                                     uint   numCells);

    /*
    void collide(float *newVel,
                 float *sortedPos,
                 float *sortedVel,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells);
                 */

    void computeDensityAndPressure(float* densities,
                                   float* pressures,
                                   float* sortedPos,
                                   float* sortedVel,
                                   uint*  gridParticleIndex,
                                   uint*  cellStart,
                                   uint*  cellEnd,
                                   uint   numParticles,
                                   uint   numFluidParticles,
                                   uint   numCells); // parameter 9

    void computeForceAndViscosity(float* newVelocities,
                                  float* entireForce,
                                  float  deltaTime,
                                  float* updatedDensities,
                                  float* updatedPressures,
                                  float* sortedPos,
                                  float* sortedVel,
                                  uint*  gridParticleIndex,
                                  uint*  cellStart,
                                  uint*  cellEnd,
                                  uint   numParticles,
                                  uint   numFluidParticles,
                                  uint   numCells); // parameter 12



    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

    // marching cubes
    void CudaAllocateTextures(uint** DeviceEdgeTable, uint** DeviceTriTable, uint** DeviceNumVerticesTable);
    void CudaCreateVolumeTexture(uchar* DeviceVolumes, size_t BufferSize);
    void CudaDestroyAllTextureObjects();
    void CudaLaunchClassifyVoxels(dim3 Grid,
        dim3 Threads,
        uint* OutVoxelVertices,
        uint* OutOccupiedVoxels,
        uchar* Volumes,
        uint3 GridSize,
        uint3 GridSizeShift,
        uint3 GridSizeMask,
        uint NumVoxels,
        float3 VoxelSize,
        float IsoValue,
        float* SortedPositions,
        uint* GridParticleIndices,
        uint* CellStarts,
        uint* CellEnds,
        uint NumFluidParticles,
        uint NumRenderingFluidParticles);
    void CudaLaunchCompactVoxels(dim3 Grid, dim3 Threads, uint* OutCompactedVoxelArray, uint* OccupiedVoxels, uint* OccupiedScanVoxels, uint NumVoxels);
    void CudaLaunchGenerateTriangles(dim3 Grid,
        dim3 Threads,
        float4* OutPositions,
        float4* OutNormals,
        uint* CompactedVoxelArray,
        uint* NumScannedVertices,
        uint3 GridSize,
        uint3 GridSizeShift,
        uint3 GridSizeMask,
        float3 VoxelSize,
        float IsoValue,
        uint NumActiveVoxels,
        uint NumMaxVertices,
        float* SortedPositions,
        uint* GridParticleIndices,
        uint* CellStarts,
        uint* CellEnds,
        uint NumFluidParticles,
        uint NumRenderingFluidParticles);
    void CudaLaunchGenerateTriangles2(dim3 Grid,
        dim3 Threads,
        float4* OutPositions,
        float4* OutNormals,
        uint* CompactedVoxelArray,
        uint* NumScannedVertices,
        uchar* Volumes,
        uint3 GridSize,
        uint3 GridSizeShift,
        uint3 GridSizeMask,
        float3 VoxelSize,
        float IsoValue,
        uint NumActiveVoxels,
        uint NumMaxVertices,
        float4* SortedPositions,
        uint* GridParticleIndices,
        uint* CellStarts,
        uint* CellEnds,
        uint NumFluidParticles,
        uint NumRenderingFluidParticles);
    void CudaThrustScanWrapper(unsigned int* Outputs, unsigned int* Inputs, unsigned int NumElements);

    void CudaCreateVolumeFromMassAndDensities(dim3 Grid,
        dim3 Threads,
        uchar* OutVolumes,
        uint3 GridSize,
        uint3 GridSizeShift,
        uint3 GridSizeMask,
        float3 VoxelSize,
        uint NumFluidParticles,
        float4* SortedPositions,
        uint* GridParticleIndices,
        uint* CellStarts,
        uint* CellEnds);
}
