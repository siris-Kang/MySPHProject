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

// SPH compute interface exposed by the CUDA static library.
// Marching-cubes surface reconstruction was intentionally dropped during the
// UE5 port (it referenced an undefined gParameters / CudaSimParams).
#pragma once

#include "particles_kernel.cuh"   // SimParams

extern "C"
{
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, int size);
    void freeArray(void *devPtr);

    void threadSync();

    void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);
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

    void computeDensityAndPressure(float* densities,
                                   float* pressures,
                                   float* sortedPos,
                                   float* sortedVel,
                                   uint*  gridParticleIndex,
                                   uint*  cellStart,
                                   uint*  cellEnd,
                                   uint   numParticles,
                                   uint   numFluidParticles,
                                   uint   numCells);

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
                                  uint   numCells);

    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);
}
