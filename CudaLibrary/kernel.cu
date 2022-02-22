/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

#endif


#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "particles_kernel_impl.cuh"

#include "device_launch_parameters.h"
*/

#include <cooperative_groups.h>
#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"
#include "helper_functions.h"

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "defines.h"
#include "tables.h"

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"


cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__constant__ SimParams params;


struct integrate_functor
{
    float deltaTime;

    __host__ __device__
        integrate_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
        void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);

        // vel += params.gravity * deltaTime;
        // vel *= params.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;
        //printf("pos=(%f, %f, %f)\n\n", pos.x, pos.y, pos.z);

        // set this to zero to disable collisions with cube sides
#if 1

        if (pos.x > 1.0f - params.particleRadius)
        {
            pos.x = 1.0f - params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.x < -1.0f + params.particleRadius)
        {
            pos.x = -1.0f + params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.y > 1.0f - params.particleRadius)
        {
            pos.y = 1.0f - params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        if (pos.z > 1.0f - params.particleRadius)
        {
            pos.z = 1.0f - params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

        if (pos.z < -1.0f + params.particleRadius)
        {
            pos.z = -1.0f + params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

#endif

        if (pos.y < -1.0f + params.particleRadius)
        {
            pos.y = -1.0f + params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floorf((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floorf((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floorf((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x - 1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y - 1);
    gridPos.z = gridPos.z & (params.gridSize.z - 1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint* gridParticleHash,  // output
    uint* gridParticleIndex, // output
    float4* pos,               // input: positions
    uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint* cellStart,        // output: cell start index
    uint* cellEnd,          // output: cell end index
    float4* sortedPos,        // output: sorted positions
    uint* gridParticleHash, // input: sorted grid hashes
    uint* gridParticleIndex,// input: sorted particle indices
    float4* oldPos,           // input: sorted position array
    uint    numParticles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index - 1];
        }
    }

    cg::sync(cta);

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        float4 pos = oldPos[sortedIndex];

        sortedPos[index] = pos;
    }
}


// Kernel function
__device__
float kernelPoly6(float3 rVector, float smoothingLength = 0.0476f)
{
    float poly6 = (315.f / (64.f * CUDART_PI_F * pow(smoothingLength, 9.f))) * pow(pow(smoothingLength, 2.f) - pow(length(rVector), 2.f), 3.f);
    return poly6;   // for 0 < length(rVector) < supportRadius
}
__device__
float3 kernelPoly6Gradient(float3 rVector, float smoothingLength = 0.0476f)
{
    float3 poly6 = -(945.0f / (32.f * CUDART_PI_F * pow(smoothingLength, 9.f))) * rVector * pow(pow(smoothingLength, 2.f) - dot(rVector, rVector), 2.f);
    return poly6;
}

__device__
float3 kernelSpikyGradient(float3 rVector, float smoothingLength = 0.0476f)
{
    if (length(rVector) != 0.0f)
    {
        float3 spiky = -(45.f / (CUDART_PI_F * pow(smoothingLength, 6.f))) * pow(smoothingLength - length(rVector), 2.f) * normalize(rVector);
        return spiky;
    }
    else // ? 겹쳤을 때? particle이 사실 겹쳐진다는거 자체가 가능한 전제일까?
    {
        return make_float3(0.0f, 0.0f, 0.0f);
    } // 일단은 냅둬보고
}

__device__
float kernelViscosityLaplacian(float3 rVector, float smoothingLength = 0.0476)
{
    float viscosity = (45.f / (CUDART_PI_F * pow(smoothingLength, 6.f))) * (smoothingLength - length(rVector));
    return viscosity;
}


// Compute Density and Pressure
__device__
float computeDensityByCell(int3    gridPos,
    uint    index,
    float3  indexPos,
    float3  indexVel,
    uint    originalIndex,
    float4* oldPosArray,
    float4* oldVelArray,
    uint* cellStart,
    uint* cellEnd,
    uint* gridParticleIndex,
    uint    numFluidParticles)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    float density = 0.f;

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];

        for (uint j = startIndex; j < endIndex; j++)
        {
            float3 jPos = make_float3(oldPosArray[j]);
            // float3 jVel = make_float3(oldVelArray[j]);
            float3 iToj = indexPos - jPos;
            float length_iToj = length(iToj);

            if (length_iToj < params.smoothingLength)
            {
                if (originalIndex < numFluidParticles)
                    density += params.fluidParticleMass * kernelPoly6(iToj, params.smoothingLength);
                else
                    density += params.boundaryParticleMass * kernelPoly6(iToj, params.smoothingLength);
            }

            //if (j != index && length_iToj < params.smoothingLength)  // check not colliding with self and distance < h
            //{
            //    if (length_iToj < params.smoothingLength)
            //    {
            //        density += params.particleMass * kernelPoly6(iToj, params.smoothingLength);
            //    }
            //}
        }
    }

    return density;
}


__global__
void computeDensityAndPressureDevice(float* desities,
    float* pressures,
    float4* oldPosArray,               // input: sorted positions
    float4* oldVelArray,               // input: sorted velocities
    uint* gridParticleIndex,         // input: sorted particle indices
    uint* cellStart,
    uint* cellEnd,
    uint    numParticles,
    uint    numFluidParticles)
{
    uint index = (blockIdx.x * blockDim.x) + threadIdx.x; // It's me

    if (index >= numParticles) return;

    // read particle data from sorted arrays
    float3 indexPos = make_float3(oldPosArray[index]);
    float3 indexVel = make_float3(oldVelArray[index]);

    // get address in grid
    int3 gridPos = calcGridPos(indexPos);

    // examine neighbouring cells
    float density = 0.f;
    float pressure = 0.f;

    uint originalIndex = gridParticleIndex[index];

    for (int z = -1; z <= 1; z++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int x = -1; x <= 1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                density += computeDensityByCell(neighbourPos, index, indexPos, indexVel, originalIndex, oldPosArray, oldVelArray, cellStart, cellEnd, gridParticleIndex, numFluidParticles);
            }
        }
    }

    pressure = params.gasStiffnessConstant * (density - params.waterRestDensity);
    desities[originalIndex] = density;
    pressures[originalIndex] = pressure;
}


// Compute Froce
__device__
void computeForceAndViscosityByCell(float3* pressureForce,
    float3* viscosity,
    int3    gridPos,
    uint    index,
    float3  indexPos,
    float3  indexVel,
    uint    originalIndex,
    float4* oldPosArray,
    float4* oldVelArray,
    float* densities,
    float* pressures,
    uint* cellStart,
    uint* cellEnd,
    uint* gridParticleIndex,
    uint  numFluidParticles)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];


    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];

        for (uint j = startIndex; j < endIndex; j++)
        {
            uint jOriginalIndex = gridParticleIndex[j];
            uint originalIndex = gridParticleIndex[index];

            float3 jPos = make_float3(oldPosArray[j]);
            float3 jVel = make_float3(oldVelArray[j]);

            float3 iTojPos = indexPos - jPos;
            float lengthRij = length(iTojPos);
            float3 iTojVel = indexVel - jVel;
            // float lengthVij = length(iTojVel);

            if (j != index && lengthRij < params.smoothingLength)  // check not colliding with self and distance < h
            {
                if (originalIndex < numFluidParticles)
                { //fluid particle
                    *pressureForce += params.fluidParticleMass
                        * (pressures[originalIndex] / (densities[originalIndex] * densities[originalIndex]) + pressures[jOriginalIndex] / (densities[jOriginalIndex] * densities[jOriginalIndex]))
                        * kernelSpikyGradient(iTojPos, params.smoothingLength);

                    *viscosity += (params.fluidParticleMass / densities[originalIndex])
                        * (dot(iTojPos, iTojVel) / (dot(iTojPos, iTojPos) + 0.01f * params.smoothingLength * params.smoothingLength))
                        * kernelPoly6Gradient(iTojPos, params.smoothingLength);
                }
                else
                { //boundary particle
                    *pressureForce += params.boundaryParticleMass
                        * (pressures[originalIndex] / (densities[originalIndex] * densities[originalIndex]) + pressures[jOriginalIndex] / (densities[jOriginalIndex] * densities[jOriginalIndex]))
                        * kernelSpikyGradient(iTojPos, params.smoothingLength);

                    *viscosity += (params.boundaryParticleMass / densities[originalIndex])
                        * (dot(iTojPos, iTojVel) / (dot(iTojPos, iTojPos) + 0.01f * params.smoothingLength * params.smoothingLength))
                        * kernelPoly6Gradient(iTojPos, params.smoothingLength);
                }

            }
        }
    }
}

__global__
void computeForceAndViscosityDevice(float4* newVelocities,
    float4* entireForces,
    float   deltaTime,
    float* densities,
    float* pressures,
    float4* oldPosArray,               // input: sorted positions
    float4* oldVelArray,               // input: sorted velocities
    uint* gridParticleIndex,         // input: sorted particle indices
    uint* cellStart,
    uint* cellEnd,
    uint    numParticles,
    uint    numFluidParticles)
{
    uint index = (blockIdx.x * blockDim.x) + threadIdx.x; // It's me

    if (index >= numParticles) return;

    // read particle data from sorted arrays
    float3 indexPos = make_float3(oldPosArray[index]);
    float3 indexVel = make_float3(oldVelArray[index]);

    // get address in grid
    int3 gridPos = calcGridPos(indexPos);

    // examine neighbouring cells
    float3 pressureForce = make_float3(0.0f);
    float3 viscosity = make_float3(0.0f);
    float3 externalForce = make_float3(0.0f);
    float3 entireForce = make_float3(0.0f);

    uint originalIndex = gridParticleIndex[index];

    for (int z = -1; z <= 1; z++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int x = -1; x <= 1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                computeForceAndViscosityByCell(&pressureForce, &viscosity, neighbourPos, index, indexPos, indexVel, originalIndex, oldPosArray, oldVelArray, densities, pressures, cellStart, cellEnd, gridParticleIndex, numFluidParticles);
            }
        }
    }

    pressureForce *= -params.fluidParticleMass;
    viscosity *= params.fluidParticleMass * params.viscosityCoefficient * 10.f / densities[originalIndex]; // 2(d+2)
    externalForce = params.gravity * params.fluidParticleMass;  //params.gravity * densities[originalIndex]; // params.gravity * params.particleMass;
    entireForce = pressureForce + viscosity + externalForce;

    if (originalIndex < numFluidParticles)
    {
        entireForces[originalIndex] = entireForce;
        newVelocities[originalIndex] += make_float4(deltaTime * (entireForce / params.fluidParticleMass), 0.0f);
    }
    else
    {
        entireForces[originalIndex] = entireForce;
        newVelocities[originalIndex] += make_float4(deltaTime * (externalForce / params.BoundaryParticleMass), 0.0f);
    }
}

extern "C"
{
    void cudaInit(int argc, char** argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char**)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void allocateArray(void** devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void freeArray(void* devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void threadSync()
    {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void copyArrayToDevice(void* device, const void* host, int offset, int size)
    {
        checkCudaErrors(cudaMemcpy((char*)device + offset, host, size, cudaMemcpyHostToDevice));
    }

    /*void registerGLBufferObject(uint vbo, struct cudaGraphicsResource** cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
            cudaGraphicsMapFlagsNone));
    }*/

    void unregisterGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    }

    void* mapGLBufferObject(struct cudaGraphicsResource** cuda_vbo_resource)
    {
        void* ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&ptr, &num_bytes,
            *cuda_vbo_resource));
        return ptr;
    }

    void unmapGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
    }

    void copyArrayFromDevice(void* host, const void* device,
        struct cudaGraphicsResource** cuda_vbo_resource, int size)
    {
        if (cuda_vbo_resource)
        {
            device = mapGLBufferObject(cuda_vbo_resource);
        }

        checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

        if (cuda_vbo_resource)
        {
            unmapGLBufferObject(*cuda_vbo_resource);
        }
    }

    void setParameters(SimParams* hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void integrateSystem(float* pos,
        float* vel,
        float deltaTime,
        uint numParticles)
    {
        thrust::device_ptr<float4> d_pos4((float4*)pos);
        thrust::device_ptr<float4> d_vel4((float4*)vel);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4 + numParticles, d_vel4 + numParticles)),
            integrate_functor(deltaTime));
    }

    void calcHash(uint* gridParticleHash,
        uint* gridParticleIndex,
        float* pos,
        int    numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        calcHashD << < numBlocks, numThreads >> > (gridParticleHash,
            gridParticleIndex,
            (float4*)pos,
            numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    cudaError_t reorderDataAndFindCellStart(uint* cellStart,
        uint* cellEnd,
        float* sortedPos,
        uint* gridParticleHash,
        uint* gridParticleIndex,
        float* oldPos,
        uint   numParticles,
        uint   numCells)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        cudaError_t error = cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint));
        checkCudaErrors(error);

        uint smemSize = sizeof(uint) * (numThreads + 1);
        reorderDataAndFindCellStartD << < numBlocks, numThreads, smemSize >> > (
            cellStart,
            cellEnd,
            (float4*)sortedPos,
            gridParticleHash,
            gridParticleIndex,
            (float4*)oldPos,
            numParticles);
        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
        return error;
    }

    void computeDensityAndPressure(float* densities,
        float* pressures,
        float* sortedPos,
        float* sortedVel,
        uint* gridParticleIndex,
        uint* cellStart,
        uint* cellEnd,
        uint   numParticles,
        uint   numFluidParticles,
        uint   numCells)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        // execute the kernel
        computeDensityAndPressureDevice << < numBlocks, numThreads >> > (densities,
            pressures,
            (float4*)sortedPos,
            (float4*)sortedVel,
            gridParticleIndex,
            cellStart,
            cellEnd,
            numParticles,
            numFluidParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void computeForceAndViscosity(float* newVelocities,
        float* entireForce,
        float  deltaTime,
        float* desities,
        float* pressures,
        float* sortedPos,
        float* sortedVel,
        uint* gridParticleIndex,
        uint* cellStart,
        uint* cellEnd,
        uint   numParticles,
        uint   numFluidParticles,
        uint   numCells)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        // execute the kernel
        computeForceAndViscosityDevice << < numBlocks, numThreads >> > ((float4*)newVelocities,
            (float4*)entireForce,
            deltaTime,
            desities,
            pressures,
            (float4*)sortedPos,
            (float4*)sortedVel,
            gridParticleIndex,
            cellStart,
            cellEnd,
            numParticles,
            numFluidParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void sortParticles(uint* dGridParticleHash, uint* dGridParticleIndex, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
            thrust::device_ptr<uint>(dGridParticleIndex));
    }


}   // extern "C"




// marching cubes
    // textures containing look-up tables
cudaTextureObject_t gTriTex;
cudaTextureObject_t gNumVerticesTex;

// volume data
cudaTextureObject_t gVolumeTex;

// sample volume data set at a point
__device__
float CudaSampleVolume(cudaTextureObject_t VolumeTex, uchar* Data, uint3 Position, uint3 GridSize)
{
    Position.x = min(Position.x, GridSize.x - 1);
    Position.y = min(Position.y, GridSize.y - 1);
    Position.z = min(Position.z, GridSize.z - 1);
    uint Index = (Position.z * GridSize.x * GridSize.y) + (Position.y * GridSize.x) + Position.x;
    //    return (float) data[i] / 255.0f;
    return tex1Dfetch<float>(VolumeTex, Index);
}

// compute position in 3d grid from 1d index
// only works for power of 2 sizes
__device__
uint3 CudaCalculateGridPosition(uint Index, uint3 GidSizeShift, uint3 GridSizeMask)
{
    uint3 GridPosition;
    GridPosition.x = Index & GridSizeMask.x;
    GridPosition.y = (Index >> GidSizeShift.y) & GridSizeMask.y;
    GridPosition.z = (Index >> GidSizeShift.z) & GridSizeMask.z;
    return GridPosition;
}

// evaluate Field function at point
__device__
float CudaParticleFieldFunction(float3 Vertex, float3 ParticlePosition)
{
    return lengthSquared(Vertex - ParticlePosition);
}

__device__
void CudaClassifyVoxelsByGrid(int3 GridPosition,
    float* OutField,
    float3 Position,
    float4* SortedPositions,
    float3 VoxelSize,
    uint* GridParticleIndices,
    uint* CellStarts,
    uint* CellEnds,
    uint NumFluidParticles,
    uint NumRenderingFluidParticles)
{
    uint GridHash = CudaCalculateGridHash(GridPosition);

    uint StartIndex = CellStarts[GridHash];

    if (StartIndex != 0xffffffff)
    {
        uint EndIndex = CellEnds[GridHash];

        for (uint NeighborIdx = StartIndex; NeighborIdx < EndIndex; ++NeighborIdx)
        {
            if (GridParticleIndices[NeighborIdx] >= NumRenderingFluidParticles)
            {
                continue;
            }
            float3 NeighborParticlePosition = make_float3(SortedPositions[NeighborIdx]);

            OutField[0] = min(CudaParticleFieldFunction(Position, NeighborParticlePosition), OutField[0]);
            OutField[1] = min(CudaParticleFieldFunction(Position + make_float3(VoxelSize.x, 0.0f, 0.0f), NeighborParticlePosition), OutField[1]);
            OutField[2] = min(CudaParticleFieldFunction(Position + make_float3(VoxelSize.x, VoxelSize.y, 0.0f), NeighborParticlePosition), OutField[2]);
            OutField[3] = min(CudaParticleFieldFunction(Position + make_float3(0.0f, VoxelSize.y, 0.0f), NeighborParticlePosition), OutField[3]);
            OutField[4] = min(CudaParticleFieldFunction(Position + make_float3(0.0f, 0.0f, VoxelSize.z), NeighborParticlePosition), OutField[4]);
            OutField[5] = min(CudaParticleFieldFunction(Position + make_float3(VoxelSize.x, 0.0f, VoxelSize.z), NeighborParticlePosition), OutField[5]);
            OutField[6] = min(CudaParticleFieldFunction(Position + make_float3(VoxelSize.x, VoxelSize.y, VoxelSize.z), NeighborParticlePosition), OutField[6]);
            OutField[7] = min(CudaParticleFieldFunction(Position + make_float3(0.0f, VoxelSize.y, VoxelSize.z), NeighborParticlePosition), OutField[7]);
            //if (gridPosition.x <= 7 && gridPosition.y <= 7 && gridPosition.z <= 7)
            //{
            //    printf("\Alpha[%u: %u - %u], neighbor(%u, %u, %u): Position=(%f, %f, %f) particle=(%f + %f, %f + %f, %f + %f) Field: %f, %f, %f, %f, %f, %f, %f, %f\OutNormalVector",
            //        Index, StartIndex, EndIndex,
            //        gridPosition.x, gridPosition.y, gridPosition.z,
            //        position.x, VoxelSize.x, 
            //        position.y, VoxelSize.y, 
            //        position.z, VoxelSize.z,
            //        ParticlePosition.x, ParticlePosition.y, ParticlePosition.z,
            //        outField[0],
            //        outField[1],
            //        outField[2],
            //        outField[3],
            //        outField[4],
            //        outField[5],
            //        outField[6],
            //        outField[7]
            //    );
            //}
        }
    }
}

// classify voxel based on number of vertices it will generate
// one thread per voxel
__global__
void CudaClassifyVoxels(uint* OutVoxelVertices,
    uint* OutOccupiedVoxels,
    uchar* Volumes,
    uint3               GridSize,
    uint3               GridSizeShift,
    uint3               GridSizeMask,
    uint                NumVoxels,
    float3              VoxelSize,
    float               IsoValue,
    cudaTextureObject_t NumVerticesTex,
    cudaTextureObject_t VolumeTex,
    float4* SortedPositions,
    uint* GridParticleIndices,
    uint* CellStarts,
    uint* CellEnds,
    uint                NumFluidParticles,
    uint                NumRenderingFluidParticles)
{
    uint BlockIdx = (blockIdx.y * gridDim.x) + blockIdx.x;
    uint Index = (BlockIdx * blockDim.x) + threadIdx.x;

    uint3 GridPosition = CudaCalculateGridPosition(Index, GridSizeShift, GridSizeMask);
    //if (GridPosition.x < 26 && GridPosition.y < 26 && GridPosition.z < 26)
    //{
    //    printf("[%u]: grid=(%u, %u, %u)\OutNormalVector", Index,
    //        GridPosition.x, GridPosition.y, GridPosition.z);
    //}
    // read Field values at neighbouring grid vertices

    float3 Position;
    Position.x = -1.0f + (GridPosition.x * VoxelSize.x);
    Position.y = -1.0f + (GridPosition.y * VoxelSize.y);
    Position.z = -1.0f + (GridPosition.z * VoxelSize.z);

#if SAMPLE_VOLUME
    float Field[8];
    Field[0] = CudaSampleVolume(VolumeTex, Volumes, GridPosition, GridSize);
    Field[1] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(1, 0, 0), GridSize);
    Field[2] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(1, 1, 0), GridSize);
    Field[3] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(0, 1, 0), GridSize);
    Field[4] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(0, 0, 1), GridSize);
    Field[5] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(1, 0, 1), GridSize);
    Field[6] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(1, 1, 1), GridSize);
    Field[7] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(0, 1, 1), GridSize);
#else
    float Field[8] = { FLT_MAX,
                       FLT_MAX,
                       FLT_MAX,
                       FLT_MAX,

                       FLT_MAX,
                       FLT_MAX,
                       FLT_MAX,
                       FLT_MAX, };

    for (int z = -gParameters.MarchingCubesNeighborSearchDepth; z <= gParameters.MarchingCubesNeighborSearchDepth; ++z)
    {
        for (int y = -gParameters.MarchingCubesNeighborSearchDepth; y <= gParameters.MarchingCubesNeighborSearchDepth; ++y)
        {
            for (int x = -gParameters.MarchingCubesNeighborSearchDepth; x <= gParameters.MarchingCubesNeighborSearchDepth; ++x)
            {
                int3 NeighborPosition = make_int3(GridPosition) + make_int3(x, y, z);

                CudaClassifyVoxelsByGrid(NeighborPosition,
                    Field,
                    Position,
                    SortedPositions,
                    VoxelSize,
                    GridParticleIndices,
                    CellStarts,
                    CellEnds,
                    NumFluidParticles,
                    NumRenderingFluidParticles);
            }
        }
    }
#endif

    // calculate flag indicating if each Vertex is inside or outside isosurface
    uint CubeIndex;
    CubeIndex = uint(Field[0] < IsoValue);
    CubeIndex += uint(Field[1] < IsoValue) * 2u;
    CubeIndex += uint(Field[2] < IsoValue) * 4u;
    CubeIndex += uint(Field[3] < IsoValue) * 8u;
    CubeIndex += uint(Field[4] < IsoValue) * 16u;
    CubeIndex += uint(Field[5] < IsoValue) * 32u;
    CubeIndex += uint(Field[6] < IsoValue) * 64u;
    CubeIndex += uint(Field[7] < IsoValue) * 128u;
    //if (GridPosition.x <= 7 && GridPosition.y <= 7 && GridPosition.z <= 7)
    //{
    //if (CubeIndex != 0u)
    //{
    //    printf("[%u]: grid(%u, %u, %u) cube index %u%u%u%u %u%u%u%u=%u\OutNormalVector", Index,
    //        GridPosition.x, GridPosition.y, GridPosition.z,
    //        !!(CubeIndex & 1u),
    //        !!(CubeIndex & 1u << 1u),
    //        !!(CubeIndex & 1u << 2u),
    //        !!(CubeIndex & 1u << 3u),
    //        !!(CubeIndex & 1u << 4u),
    //        !!(CubeIndex & 1u << 5u),
    //        !!(CubeIndex & 1u << 6u),
    //        !!(CubeIndex & 1u << 7u),
    //        CubeIndex
    //    );
    //}
    //}


    // read number of vertices from texture
    uint NumVertices = tex1Dfetch<uint>(NumVerticesTex, CubeIndex);

    if (Index < NumVoxels)
    {
        OutVoxelVertices[Index] = NumVertices;
        OutOccupiedVoxels[Index] = (NumVertices > 0);
    }
}

// compact voxel array
__global__
void CudaCompactVoxels(uint* OutCompactedVoxelArray, uint* OccupiedVoxels, uint* OccupiedScanVoxels, uint NumVoxels)
{
    uint BlockIdx = (blockIdx.y * gridDim.x) + blockIdx.x;
    uint Index = (BlockIdx * blockDim.x) + threadIdx.x;

    if (OccupiedVoxels[Index] && (Index < NumVoxels))
    {
        OutCompactedVoxelArray[OccupiedScanVoxels[Index]] = Index;
    }
}

// compute interpolated Vertex along an edge
__device__
float3 CudaVertexLerp(float Isolevel, float3 Position0, float3 Position1, float Alpha0, float Alpha1)
{
    float Alpha = (Isolevel - Alpha0) / (Alpha1 - Alpha0);
    return lerp(Position0, Position1, Alpha);
}

// compute interpolated Vertex position and normal along an edge
__device__
void CudaVertexLerp2(float Isolevel, float3 Position0, float3 Position1, float4 AlphaVector0, float4 AlphaVector1, float3& OutPosition, float3& OutNormalVector)
{
    float Alpha = (Isolevel - AlphaVector0.w) / (AlphaVector1.w - AlphaVector0.w);
    OutPosition = lerp(Position0, Position1, Alpha);
    OutPosition = make_float3(OutPosition.x * -1.0f, OutPosition.z * -1.0f, OutPosition.y * -1.0f);
    OutNormalVector.x = -lerp(AlphaVector0.x, AlphaVector1.x, Alpha);
    OutNormalVector.y = -lerp(AlphaVector0.z, AlphaVector1.z, Alpha);
    OutNormalVector.z = -lerp(AlphaVector0.y, AlphaVector1.y, Alpha);
    //printf("\Alpha\tinterpolate: Alpha=%f, OutNormalVector=(%f, %f, %f), Alpha0=(%f, %f, %f), Alpha1=(%f, %f, %f)\OutNormalVector", Alpha, 
    //    OutNormalVector.x, OutNormalVector.y, OutNormalVector.z, Alpha0.x, Alpha0.y, Alpha0.z, Alpha1.x, Alpha1.y, Alpha1.z);
    //    OutNormalVector = normalize(OutNormalVector);
}

__device__
float4 CudaParticleFieldFunction4(float3 Vertex, float3 ParticlePosition)
{
    float Value = CudaParticleFieldFunction(Vertex, ParticlePosition);
    const float Delta = 0.001f;
    float Dx = CudaParticleFieldFunction(make_float3(Vertex.x + Delta, Vertex.y, Vertex.z), ParticlePosition) - Value;
    float Dy = CudaParticleFieldFunction(make_float3(Vertex.x, Vertex.y + Delta, Vertex.z), ParticlePosition) - Value;
    float Dz = CudaParticleFieldFunction(make_float3(Vertex.x, Vertex.y, Vertex.z + Delta), ParticlePosition) - Value;
    //printf("\Alpha\tv: %f, dx dy dz = (%f, %f, %f)\OutNormalVector", Value, dx, dy, dz);
    return make_float4(Dx, Dy, Dz, Value);
}

// evaluate Field function at a point
// returns value and gradient in float4
__device__
void CudaGetNormalVector(float4& OutNormalVector, float3 Vertex, float3 ParticlePosition)
{
    float3 ParticleToVertex = Vertex - ParticlePosition;
    if (OutNormalVector.w > lengthSquared(ParticleToVertex) - gParameters.SupportRadiusSquared)
    {
        OutNormalVector = make_float4(normalize(ParticleToVertex), lengthSquared(ParticleToVertex) - gParameters.SupportRadiusSquared);
    }
}

__device__
void CudaGenerateTrianglesByGrid(int3 GridPosition,
    float4* OutField,
    float3* Verticess,
    float4* SortedPositions,
    float3 VoxelSize,
    uint* GridParticleIndices,
    uint* CellStarts,
    uint* CellEnds,
    uint NumFluidParticles,
    uint NumRenderingFluidParticles)
{
    uint GridHash = CudaCalculateGridHash(GridPosition);

    uint StartIndex = CellStarts[GridHash];

    if (StartIndex != 0xffffffff)
    {
        uint EndIndex = CellEnds[GridHash];
        for (uint NeighborIdx = StartIndex; NeighborIdx < EndIndex; ++NeighborIdx)
        {
            if (GridParticleIndices[NeighborIdx] >= NumRenderingFluidParticles)
            {
                continue;
            }
            float3 NeighborParticlePosition = make_float3(SortedPositions[NeighborIdx]);

            //getNormalVector(outField[j], vertices[j], ParticlePosition);
            float4 Gradient = CudaParticleFieldFunction4(Verticess[0], NeighborParticlePosition);
            if (OutField[0].w > Gradient.w)
            {
                OutField[0] = Gradient;
            }
            Gradient = CudaParticleFieldFunction4(Verticess[1], NeighborParticlePosition);
            if (OutField[1].w > Gradient.w)
            {
                OutField[1] = Gradient;
            }
            Gradient = CudaParticleFieldFunction4(Verticess[2], NeighborParticlePosition);
            if (OutField[2].w > Gradient.w)
            {
                OutField[2] = Gradient;
            }
            Gradient = CudaParticleFieldFunction4(Verticess[3], NeighborParticlePosition);
            if (OutField[3].w > Gradient.w)
            {
                OutField[3] = Gradient;
            }
            Gradient = CudaParticleFieldFunction4(Verticess[4], NeighborParticlePosition);
            if (OutField[4].w > Gradient.w)
            {
                OutField[4] = Gradient;
            }
            Gradient = CudaParticleFieldFunction4(Verticess[5], NeighborParticlePosition);
            if (OutField[5].w > Gradient.w)
            {
                OutField[5] = Gradient;
            }
            Gradient = CudaParticleFieldFunction4(Verticess[6], NeighborParticlePosition);
            if (OutField[6].w > Gradient.w)
            {
                OutField[6] = Gradient;
            }
            Gradient = CudaParticleFieldFunction4(Verticess[7], NeighborParticlePosition);
            if (OutField[7].w > Gradient.w)
            {
                OutField[7] = Gradient;
            }
        }
    }
}

// generate triangles for each voxel using marching cubes
// interpolates normals from Field function
__global__ void
CudaGenerateTriangles(float4* OutPositions,
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
    cudaTextureObject_t TriTex,
    cudaTextureObject_t NumVerticesTex,
    float4* SortedPositions,
    uint* GridParticleIndices,
    uint* CellStarts,
    uint* CellEnds,
    uint NumFluidParticles,
    uint NumRenderingFluidParticles)
{
    uint BlockIdx = (blockIdx.y * gridDim.x) + blockIdx.x;
    uint Index = (BlockIdx * blockDim.x) + threadIdx.x;

    if (Index > NumActiveVoxels - 1)
    {
        // can'Alpha return here because of syncthreads()
        Index = NumActiveVoxels - 1;
    }

#if SKIP_EMPTY_VOXELS
    uint Voxel = CompactedVoxelArray[Index];
#else
    uint Voxel = OutputIdx;
#endif

    // compute position in 3d grid
    uint3 GridPosition = CudaCalculateGridPosition(Voxel, GridSizeShift, GridSizeMask);

    float3 Position;
    Position.x = -1.0f + (GridPosition.x * VoxelSize.x);
    Position.y = -1.0f + (GridPosition.y * VoxelSize.y);
    Position.z = -1.0f + (GridPosition.z * VoxelSize.z);

    // calculate cell Vertex positions
    float3 Vertices[8];
    Vertices[0] = Position;
    Vertices[1] = Position + make_float3(VoxelSize.x, 0, 0);
    Vertices[2] = Position + make_float3(VoxelSize.x, VoxelSize.y, 0);
    Vertices[3] = Position + make_float3(0, VoxelSize.y, 0);
    Vertices[4] = Position + make_float3(0, 0, VoxelSize.z);
    Vertices[5] = Position + make_float3(VoxelSize.x, 0, VoxelSize.z);
    Vertices[6] = Position + make_float3(VoxelSize.x, VoxelSize.y, VoxelSize.z);
    Vertices[7] = Position + make_float3(0, VoxelSize.y, VoxelSize.z);

    // evaluate Field values
    float4 Field[8] = {
        make_float4(0.0f, 0.0f, 0.0f, FLT_MAX),
        make_float4(0.0f, 0.0f, 0.0f, FLT_MAX),
        make_float4(0.0f, 0.0f, 0.0f, FLT_MAX),
        make_float4(0.0f, 0.0f, 0.0f, FLT_MAX),

        make_float4(0.0f, 0.0f, 0.0f, FLT_MAX),
        make_float4(0.0f, 0.0f, 0.0f, FLT_MAX),
        make_float4(0.0f, 0.0f, 0.0f, FLT_MAX),
        make_float4(0.0f, 0.0f, 0.0f, FLT_MAX)
    };
    //Field[0] = fieldFunc4(Value[0]);
    //Field[1] = fieldFunc4(Value[1]);
    //Field[2] = fieldFunc4(Value[2]);
    //Field[3] = fieldFunc4(Value[3]);
    //Field[4] = fieldFunc4(Value[4]);
    //Field[5] = fieldFunc4(Value[5]);
    //Field[6] = fieldFunc4(Value[6]);
    //Field[7] = fieldFunc4(Value[7]);

    for (int z = -gParameters.MarchingCubesNeighborSearchDepth; z <= gParameters.MarchingCubesNeighborSearchDepth; ++z)
    {
        for (int y = -gParameters.MarchingCubesNeighborSearchDepth; y <= gParameters.MarchingCubesNeighborSearchDepth; ++y)
        {
            for (int x = -gParameters.MarchingCubesNeighborSearchDepth; x <= gParameters.MarchingCubesNeighborSearchDepth; ++x)
            {
                int3 NeighborPosition = make_int3(GridPosition) + make_int3(x, y, z);
                CudaGenerateTrianglesByGrid(NeighborPosition,
                    Field,
                    Vertices,
                    SortedPositions,
                    VoxelSize,
                    GridParticleIndices,
                    CellStarts,
                    CellEnds,
                    NumFluidParticles,
                    NumRenderingFluidParticles);
            }
        }
    }

    // recalculate flag
    // (this is faster than storing it in global memory)
    uint CubeIndex;
    CubeIndex = uint(Field[0].w < IsoValue);
    CubeIndex += uint(Field[1].w < IsoValue) * 2u;
    CubeIndex += uint(Field[2].w < IsoValue) * 4u;
    CubeIndex += uint(Field[3].w < IsoValue) * 8u;
    CubeIndex += uint(Field[4].w < IsoValue) * 16u;
    CubeIndex += uint(Field[5].w < IsoValue) * 32u;
    CubeIndex += uint(Field[6].w < IsoValue) * 64u;
    CubeIndex += uint(Field[7].w < IsoValue) * 128u;
    if (CubeIndex != 0u)
    {
        //printf("[%u]: triangle grid(%u, %u, %u) cube index %u%u%u%u %u%u%u%u=%u\OutNormalVector", Index,
        //    GridPosition.x, GridPosition.y, GridPosition.z,
        //    !!(CubeIndex & 1u),
        //    !!(CubeIndex & 1u << 1u),
        //    !!(CubeIndex & 1u << 2u),
        //    !!(CubeIndex & 1u << 3u),
        //    !!(CubeIndex & 1u << 4u),
        //    !!(CubeIndex & 1u << 5u),
        //    !!(CubeIndex & 1u << 6u),
        //    !!(CubeIndex & 1u << 7u),
        //    CubeIndex
        //);
    }


    // find the vertices where the surface intersects the cube

#if USE_SHARED
    // use partioned shared memory to avoid using local memory
    __shared__ float3 VerticesList[12 * NTHREADS];
    __shared__ float3 NormalsList[12 * NTHREADS];

    CudaVertexLerp2(IsoValue, Vertices[0], Vertices[1], Field[0], Field[1], VerticesList[threadIdx.x], NormalsList[threadIdx.x]);
    CudaVertexLerp2(IsoValue, Vertices[1], Vertices[2], Field[1], Field[2], VerticesList[threadIdx.x + NTHREADS], NormalsList[threadIdx.x + NTHREADS]);
    CudaVertexLerp2(IsoValue, Vertices[2], Vertices[3], Field[2], Field[3], VerticesList[threadIdx.x + (NTHREADS * 2)], NormalsList[threadIdx.x + (NTHREADS * 2)]);
    CudaVertexLerp2(IsoValue, Vertices[3], Vertices[0], Field[3], Field[0], VerticesList[threadIdx.x + (NTHREADS * 3)], NormalsList[threadIdx.x + (NTHREADS * 3)]);
    CudaVertexLerp2(IsoValue, Vertices[4], Vertices[5], Field[4], Field[5], VerticesList[threadIdx.x + (NTHREADS * 4)], NormalsList[threadIdx.x + (NTHREADS * 4)]);
    CudaVertexLerp2(IsoValue, Vertices[5], Vertices[6], Field[5], Field[6], VerticesList[threadIdx.x + (NTHREADS * 5)], NormalsList[threadIdx.x + (NTHREADS * 5)]);
    CudaVertexLerp2(IsoValue, Vertices[6], Vertices[7], Field[6], Field[7], VerticesList[threadIdx.x + (NTHREADS * 6)], NormalsList[threadIdx.x + (NTHREADS * 6)]);
    CudaVertexLerp2(IsoValue, Vertices[7], Vertices[4], Field[7], Field[4], VerticesList[threadIdx.x + (NTHREADS * 7)], NormalsList[threadIdx.x + (NTHREADS * 7)]);
    CudaVertexLerp2(IsoValue, Vertices[0], Vertices[4], Field[0], Field[4], VerticesList[threadIdx.x + (NTHREADS * 8)], NormalsList[threadIdx.x + (NTHREADS * 8)]);
    CudaVertexLerp2(IsoValue, Vertices[1], Vertices[5], Field[1], Field[5], VerticesList[threadIdx.x + (NTHREADS * 9)], NormalsList[threadIdx.x + (NTHREADS * 9)]);
    CudaVertexLerp2(IsoValue, Vertices[2], Vertices[6], Field[2], Field[6], VerticesList[threadIdx.x + (NTHREADS * 10)], NormalsList[threadIdx.x + (NTHREADS * 10)]);
    CudaVertexLerp2(IsoValue, Vertices[3], Vertices[7], Field[3], Field[7], VerticesList[threadIdx.x + (NTHREADS * 11)], NormalsList[threadIdx.x + (NTHREADS * 11)]);
    //for (uint Index = 0; Index < 12; ++Index)
    //{
    //    if (normlist[threadIdx.x + (NTHREADS * Index)].x == 0.0f && normlist[threadIdx.x + (NTHREADS * Index)].y == 0.0f && normlist[threadIdx.x + (NTHREADS * Index)].z == 0.0f)
    //    {
    //        printf("\Alpha[%u] norm[%u]: (%f, %f, %f)\OutNormalVector", Index, threadIdx.x + (NTHREADS * Index), 
    //            normlist[threadIdx.x + (NTHREADS * Index)].x, normlist[threadIdx.x + (NTHREADS * Index)].y, normlist[threadIdx.x + (NTHREADS * Index)].z);
    //    }
    //}
    //printf("\Alpha[%6u] norm[%u]: (%f, %f, %f)\OutNormalVector"
    //    "\Alpha         norm[%u]: (%f, %f, %f)\OutNormalVector"
    //    "\Alpha         norm[%u]: (%f, %f, %f)\OutNormalVector"
    //    "\Alpha         norm[%u]: (%f, %f, %f)\OutNormalVector"
    //    "\Alpha         norm[%u]: (%f, %f, %f)\OutNormalVector"
    //    "\Alpha         norm[%u]: (%f, %f, %f)\OutNormalVector"
    //    "\Alpha         norm[%u]: (%f, %f, %f)\OutNormalVector"
    //    "\Alpha         norm[%u]: (%f, %f, %f)\OutNormalVector"
    //    "\Alpha         norm[%u]: (%f, %f, %f)\OutNormalVector"
    //    "\Alpha         norm[%u]: (%f, %f, %f)\OutNormalVector"
    //    "\Alpha         norm[%u]: (%f, %f, %f)\OutNormalVector"
    //    "\Alpha         norm[%u]: (%f, %f, %f)\OutNormalVector", Index,
    //    threadIdx.x, normlist[threadIdx.x].x, normlist[threadIdx.x].y, normlist[threadIdx.x].z,
    //    threadIdx.x + NTHREADS, normlist[threadIdx.x + NTHREADS].x, normlist[threadIdx.x + NTHREADS].y, normlist[threadIdx.x + NTHREADS].z,
    //    threadIdx.x + (NTHREADS * 2), normlist[threadIdx.x + (NTHREADS * 2)].x, normlist[threadIdx.x + (NTHREADS * 2)].y, normlist[threadIdx.x + (NTHREADS * 2)].z,
    //    threadIdx.x + (NTHREADS * 3), normlist[threadIdx.x + (NTHREADS * 3)].x, normlist[threadIdx.x + (NTHREADS * 3)].y, normlist[threadIdx.x + (NTHREADS * 3)].z,
    //    threadIdx.x + (NTHREADS * 4), normlist[threadIdx.x + (NTHREADS * 4)].x, normlist[threadIdx.x + (NTHREADS * 4)].y, normlist[threadIdx.x + (NTHREADS * 4)].z,
    //    threadIdx.x + (NTHREADS * 5), normlist[threadIdx.x + (NTHREADS * 5)].x, normlist[threadIdx.x + (NTHREADS * 5)].y, normlist[threadIdx.x + (NTHREADS * 5)].z,
    //    threadIdx.x + (NTHREADS * 6), normlist[threadIdx.x + (NTHREADS * 6)].x, normlist[threadIdx.x + (NTHREADS * 6)].y, normlist[threadIdx.x + (NTHREADS * 6)].z,
    //    threadIdx.x + (NTHREADS * 7), normlist[threadIdx.x + (NTHREADS * 7)].x, normlist[threadIdx.x + (NTHREADS * 7)].y, normlist[threadIdx.x + (NTHREADS * 7)].z,
    //    threadIdx.x + (NTHREADS * 8), normlist[threadIdx.x + (NTHREADS * 8)].x, normlist[threadIdx.x + (NTHREADS * 8)].y, normlist[threadIdx.x + (NTHREADS * 8)].z,
    //    threadIdx.x + (NTHREADS * 9), normlist[threadIdx.x + (NTHREADS * 9)].x, normlist[threadIdx.x + (NTHREADS * 9)].y, normlist[threadIdx.x + (NTHREADS * 9)].z,
    //    threadIdx.x + (NTHREADS * 10), normlist[threadIdx.x + (NTHREADS * 10)].x, normlist[threadIdx.x + (NTHREADS * 10)].y, normlist[threadIdx.x + (NTHREADS * 10)].z,
    //    threadIdx.x + (NTHREADS * 11), normlist[threadIdx.x + (NTHREADS * 11)].x, normlist[threadIdx.x + (NTHREADS * 11)].y, normlist[threadIdx.x + (NTHREADS * 11)].z
    //    );
    __syncthreads();

#else
    float3 VerticesList[12];
    float3 NormalsList[12];

    CudaVertexLerp2(IsoValue, Value[0], Value[1], Field[0], Field[1], VerticesList[0], NormalsList[0]);
    CudaVertexLerp2(IsoValue, Value[1], Value[2], Field[1], Field[2], VerticesList[1], NormalsList[1]);
    CudaVertexLerp2(IsoValue, Value[2], Value[3], Field[2], Field[3], VerticesList[2], NormalsList[2]);
    CudaVertexLerp2(IsoValue, Value[3], Value[0], Field[3], Field[0], VerticesList[3], NormalsList[3]);

    CudaVertexLerp2(IsoValue, Value[4], Value[5], Field[4], Field[5], VerticesList[4], NormalsList[4]);
    CudaVertexLerp2(IsoValue, Value[5], Value[6], Field[5], Field[6], VerticesList[5], NormalsList[5]);
    CudaVertexLerp2(IsoValue, Value[6], Value[7], Field[6], Field[7], VerticesList[6], NormalsList[6]);
    CudaVertexLerp2(IsoValue, Value[7], Value[4], Field[7], Field[4], VerticesList[7], NormalsList[7]);

    CudaVertexLerp2(IsoValue, Value[0], Value[4], Field[0], Field[4], VerticesList[8], NormalsList[8]);
    CudaVertexLerp2(IsoValue, Value[1], Value[5], Field[1], Field[5], VerticesList[9], NormalsList[9]);
    CudaVertexLerp2(IsoValue, Value[2], Value[6], Field[2], Field[6], VerticesList[10], NormalsList[10]);
    CudaVertexLerp2(IsoValue, Value[3], Value[7], Field[3], Field[7], VerticesList[11], NormalsList[11]);
#endif

    // output triangle vertices
    uint NumVertices = tex1Dfetch<uint>(NumVerticesTex, CubeIndex);

    for (int i = 0; i < NumVertices; i++)
    {
        uint Edge = tex1Dfetch<uint>(TriTex, CubeIndex * 16 + i);

        uint Index = NumScannedVertices[Voxel] + i;

        if (Index < NumMaxVertices)
        {
#if USE_SHARED
            OutPositions[Index] = make_float4(VerticesList[(Edge * NTHREADS) + threadIdx.x], 1.0f);
            OutNormals[Index] = make_float4(NormalsList[(Edge * NTHREADS) + threadIdx.x], 0.0f);
            //printf("\Alpha[%u] index: %u, edge: %u, pos=(%f, %f, %f, %f), norm=(%f, %f, %f, %f)\OutNormalVector", Index, index, edge,
            //    pos[index].x, pos[index].y, pos[index].z, pos[index].w,
            //    norm[index].x, norm[index].y, norm[index].z, norm[index].w);
#else
            OutPositions[OutputIdx] = make_float4(VerticesList[Edge], 1.0f);
            OutNormals[OutputIdx] = make_float4(NormalsList[Edge], 0.0f);
#endif
        }
    }
}

// calculate triangle normal
__device__
float3 CudaCalculateNormal(float3* Vertex0, float3* Vertex1, float3* Vertex2)
{
    float3 Edge0 = *Vertex1 - *Vertex0;
    float3 Edge1 = *Vertex2 - *Vertex0;
    // note - it's faster to perform normalization in Vertex shader rather than here
    return cross(Edge0, Edge1);
}

// version that calculates flat surface normal for each triangle
__global__
void CudaGenerateTriangles2(float4* OutPositions,
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
    cudaTextureObject_t TriTex,
    cudaTextureObject_t NumVerticesTex,
    cudaTextureObject_t VolumeTex,
    float4* SortedPositions,
    uint* GridParticleIndices,
    uint* CellStarts,
    uint* CellEnds,
    uint NumFluidParticles,
    uint NumRenderingFluidParticles)
{
    uint BlockIdx = (blockIdx.y * gridDim.x) + blockIdx.x;
    uint Index = (BlockIdx * blockDim.x) + threadIdx.x;

    if (Index > NumActiveVoxels - 1)
    {
        Index = NumActiveVoxels - 1;
    }

#if SKIP_EMPTY_VOXELS
    uint Voxel = CompactedVoxelArray[Index];
#else
    uint Voxel = OutputIdx;
#endif

    // compute position in 3d grid
    uint3 GridPosition = CudaCalculateGridPosition(Voxel, GridSizeShift, GridSizeMask);

    float3 Position;
    Position.x = -1.0f + (GridPosition.x * VoxelSize.x);
    Position.y = -1.0f + (GridPosition.y * VoxelSize.y);
    Position.z = -1.0f + (GridPosition.z * VoxelSize.z);

    // calculate cell Vertex positions
    float3 Vertices[8];
    Vertices[0] = Position;
    Vertices[1] = Position + make_float3(VoxelSize.x, 0, 0);
    Vertices[2] = Position + make_float3(VoxelSize.x, VoxelSize.y, 0);
    Vertices[3] = Position + make_float3(0, VoxelSize.y, 0);
    Vertices[4] = Position + make_float3(0, 0, VoxelSize.z);
    Vertices[5] = Position + make_float3(VoxelSize.x, 0, VoxelSize.z);
    Vertices[6] = Position + make_float3(VoxelSize.x, VoxelSize.y, VoxelSize.z);
    Vertices[7] = Position + make_float3(0, VoxelSize.y, VoxelSize.z);


#if SAMPLE_VOLUME
    float Field[8];
    Field[0] = CudaSampleVolume(VolumeTex, Volumes, GridPosition, GridSize);
    Field[1] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(1, 0, 0), GridSize);
    Field[2] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(1, 1, 0), GridSize);
    Field[3] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(0, 1, 0), GridSize);
    Field[4] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(0, 0, 1), GridSize);
    Field[5] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(1, 0, 1), GridSize);
    Field[6] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(1, 1, 1), GridSize);
    Field[7] = CudaSampleVolume(VolumeTex, Volumes, GridPosition + make_uint3(0, 1, 1), GridSize);
#else
    // evaluate Field values
    float Field[8] = { FLT_MAX,
                        FLT_MAX,
                        FLT_MAX,
                        FLT_MAX,

                        FLT_MAX,
                        FLT_MAX,
                        FLT_MAX,
                        FLT_MAX, };

    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                int3 NeighborPosition = make_int3(GridPosition) + make_int3(x, y, z);

                CudaClassifyVoxelsByGrid(NeighborPosition,
                    Field,
                    Position,
                    SortedPositions,
                    VoxelSize,
                    GridParticleIndices,
                    CellStarts,
                    CellEnds,
                    NumFluidParticles,
                    NumRenderingFluidParticles);
            }
        }
    }
#endif

    // recalculate flag
    uint CudeIndex;
    CudeIndex = uint(Field[0] < IsoValue);
    CudeIndex += uint(Field[1] < IsoValue) * 2u;
    CudeIndex += uint(Field[2] < IsoValue) * 4u;
    CudeIndex += uint(Field[3] < IsoValue) * 8u;
    CudeIndex += uint(Field[4] < IsoValue) * 16u;
    CudeIndex += uint(Field[5] < IsoValue) * 32u;
    CudeIndex += uint(Field[6] < IsoValue) * 64u;
    CudeIndex += uint(Field[7] < IsoValue) * 128u;

    // find the vertices where the surface intersects the cube

#if USE_SHARED
    // use shared memory to avoid using local
    __shared__ float3 VerticesList[12 * NTHREADS];

    VerticesList[threadIdx.x] = CudaVertexLerp(IsoValue, Vertices[0], Vertices[1], Field[0], Field[1]);
    VerticesList[NTHREADS + threadIdx.x] = CudaVertexLerp(IsoValue, Vertices[1], Vertices[2], Field[1], Field[2]);
    VerticesList[(NTHREADS * 2) + threadIdx.x] = CudaVertexLerp(IsoValue, Vertices[2], Vertices[3], Field[2], Field[3]);
    VerticesList[(NTHREADS * 3) + threadIdx.x] = CudaVertexLerp(IsoValue, Vertices[3], Vertices[0], Field[3], Field[0]);
    VerticesList[(NTHREADS * 4) + threadIdx.x] = CudaVertexLerp(IsoValue, Vertices[4], Vertices[5], Field[4], Field[5]);
    VerticesList[(NTHREADS * 5) + threadIdx.x] = CudaVertexLerp(IsoValue, Vertices[5], Vertices[6], Field[5], Field[6]);
    VerticesList[(NTHREADS * 6) + threadIdx.x] = CudaVertexLerp(IsoValue, Vertices[6], Vertices[7], Field[6], Field[7]);
    VerticesList[(NTHREADS * 7) + threadIdx.x] = CudaVertexLerp(IsoValue, Vertices[7], Vertices[4], Field[7], Field[4]);
    VerticesList[(NTHREADS * 8) + threadIdx.x] = CudaVertexLerp(IsoValue, Vertices[0], Vertices[4], Field[0], Field[4]);
    VerticesList[(NTHREADS * 9) + threadIdx.x] = CudaVertexLerp(IsoValue, Vertices[1], Vertices[5], Field[1], Field[5]);
    VerticesList[(NTHREADS * 10) + threadIdx.x] = CudaVertexLerp(IsoValue, Vertices[2], Vertices[6], Field[2], Field[6]);
    VerticesList[(NTHREADS * 11) + threadIdx.x] = CudaVertexLerp(IsoValue, Vertices[3], Vertices[7], Field[3], Field[7]);
    __syncthreads();
#else

    float3 VerticesList[12];

    VerticesList[0] = CudaVertexLerp(IsoValue, Value[0], Value[1], Field[0], Field[1]);
    VerticesList[1] = CudaVertexLerp(IsoValue, Value[1], Value[2], Field[1], Field[2]);
    VerticesList[2] = CudaVertexLerp(IsoValue, Value[2], Value[3], Field[2], Field[3]);
    VerticesList[3] = CudaVertexLerp(IsoValue, Value[3], Value[0], Field[3], Field[0]);

    VerticesList[4] = CudaVertexLerp(IsoValue, Value[4], Value[5], Field[4], Field[5]);
    VerticesList[5] = CudaVertexLerp(IsoValue, Value[5], Value[6], Field[5], Field[6]);
    VerticesList[6] = CudaVertexLerp(IsoValue, Value[6], Value[7], Field[6], Field[7]);
    VerticesList[7] = CudaVertexLerp(IsoValue, Value[7], Value[4], Field[7], Field[4]);

    VerticesList[8] = CudaVertexLerp(IsoValue, Value[0], Value[4], Field[0], Field[4]);
    VerticesList[9] = CudaVertexLerp(IsoValue, Value[1], Value[5], Field[1], Field[5]);
    VerticesList[10] = CudaVertexLerp(IsoValue, Value[2], Value[6], Field[2], Field[6]);
    VerticesList[11] = CudaVertexLerp(IsoValue, Value[3], Value[7], Field[3], Field[7]);
#endif

    // output triangle vertices
    uint NumVertices = tex1Dfetch<uint>(NumVerticesTex, CudeIndex);

    for (int VertexIdx = 0; VertexIdx < NumVertices; VertexIdx += 3)
    {
        uint OutputIdx = NumScannedVertices[Voxel] + VertexIdx;

        float3* OutputVertices[3];
        uint Edge;
        Edge = tex1Dfetch<uint>(TriTex, (CudeIndex * 16) + VertexIdx);
#if USE_SHARED
        OutputVertices[0] = &VerticesList[(Edge * NTHREADS) + threadIdx.x];
#else
        Value[0] = &VerticesList[Edge];
#endif

        Edge = tex1Dfetch<uint>(TriTex, (CudeIndex * 16) + VertexIdx + 1);
#if USE_SHARED
        OutputVertices[1] = &VerticesList[(Edge * NTHREADS) + threadIdx.x];
#else
        Value[1] = &VerticesList[Edge];
#endif

        Edge = tex1Dfetch<uint>(TriTex, (CudeIndex * 16) + VertexIdx + 2);
#if USE_SHARED
        OutputVertices[2] = &VerticesList[(Edge * NTHREADS) + threadIdx.x];
#else
        Value[2] = &VerticesList[Edge];
#endif

        // calculate triangle surface normal
        float3 Normal = CudaCalculateNormal(OutputVertices[0], OutputVertices[1], OutputVertices[2]);

        if (OutputIdx < (NumMaxVertices - 3))
        {
            OutPositions[OutputIdx] = make_float4(*OutputVertices[0], 1.0f);
            OutNormals[OutputIdx] = make_float4(Normal, 0.0f);

            OutPositions[OutputIdx + 1] = make_float4(*OutputVertices[1], 1.0f);
            OutNormals[OutputIdx + 1] = make_float4(Normal, 0.0f);

            OutPositions[OutputIdx + 2] = make_float4(*OutputVertices[2], 1.0f);
            OutNormals[OutputIdx + 2] = make_float4(Normal, 0.0f);
        }
    }
}

__device__
float CudaMcComputeDensities(int3 GridPosition,
    float3 Position,
    float4* SortedPositions,
    uint NumFluidParticles,
    uint* CellStarts,
    uint* CellEnds)
{
    uint GridHash = CudaMcCalculateGridHash(GridPosition);

    uint StartIndex = CellStarts[GridHash];

    float Density = 0.0f;
    if (StartIndex != 0xffffffff)
    {
        uint EndIndex = CellEnds[GridHash];

        for (uint NeighborIdx = StartIndex; NeighborIdx < EndIndex; ++NeighborIdx)
        {
            float3 Rij = Position - make_float3(SortedPositions[NeighborIdx]);
            float R2 = lengthSquared(Rij);

            if (R2 < gParameters.SupportRadiusSquared)
            {
                if (NeighborIdx < NumFluidParticles)
                {
                    Density += gParameters.ParticleMass * CudaKernelPoly6ByDistanceSquared(R2);
                }
                else
                {
                    Density += gParameters.BoundaryParticleMass * CudaKernelPoly6ByDistanceSquared(R2);
                }
            }
        }
    }
    return Density;
}

__global__
void CudaCreateVolumeFromMassAndDensitiesDevice(uchar* OutVolumes,
    uint3 GridSize,
    uint3 GridSizeShift,
    uint3 GridSizeMask,
    float3 VoxelSize,
    uint NumFluidParticles,
    float4* SortedPositions,
    uint* GridParticleIndices,
    uint* CellStarts,
    uint* CellEnds)
{
    uint BlockIdx = (blockIdx.y * gridDim.x) + blockIdx.x;
    uint Index = (BlockIdx * blockDim.x) + threadIdx.x;

    uint3 GridPosition = CudaCalculateGridPosition(Index, GridSizeShift, GridSizeMask);

    float3 Position;
    Position.x = -1.0f + (GridPosition.x * VoxelSize.x);
    Position.y = -1.0f + (GridPosition.y * VoxelSize.y);
    Position.z = -1.0f + (GridPosition.z * VoxelSize.z);

    float Density = 0.0f;
    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                int3 NeighborPosition = make_int3(GridPosition) + make_int3(x, y, z);

                Density += CudaMcComputeDensities(NeighborPosition,
                    Position,
                    SortedPositions,
                    NumFluidParticles,
                    CellStarts,
                    CellEnds);
            }
        }
    }

    OutVolumes[Index] = gParameters.ParticleMass / Density;
}
#pragma endregion

int main()
{
    return 0;
}


#pragma region SphExtern
//particleSystem_cuda.cu
extern "C"
{

    void CudaInit(int Argc, char** Argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(Argc, (const char**)Argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void CudaAllocateArray(void** DevicePtr, size_t Size)
    {
        checkCudaErrors(cudaMalloc(DevicePtr, Size));
    }

    void CudaFreeArray(void* DevicePtr)
    {
        checkCudaErrors(cudaFree(DevicePtr));
    }

    void CudaThreadSync()
    {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void CudaCopyArrayToDevice(void* DeviceArray, const void* HostArray, size_t Offset, size_t Size)
    {
        checkCudaErrors(cudaMemcpy((char*)DeviceArray + Offset, HostArray, Size, cudaMemcpyHostToDevice));
    }

    void CudaSetParameters(CudaSimParams* HostParameters)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(gParameters, HostParameters, sizeof(CudaSimParams)));
    }

    //Round a / b to nearest higher integer value
    uint CudaDivideUp(uint A, uint B)
    {
        return (A % B != 0) ? (A / B + 1) : (A / B);
    }

    // compute grid and thread block size for a given number of elements
    void CudaComputeGridSize(uint Size, uint BlockSize, uint& OutNumBlocks, uint& OutNumThreads)
    {
        OutNumThreads = min(BlockSize, Size);
        OutNumBlocks = CudaDivideUp(Size, OutNumThreads);
    }



    void CudaIntegrateSystem(float* Positions, float* Velocities, uint NumParticles)
    {
        thrust::device_ptr<float4> DevicePositions((float4*)Positions);
        thrust::device_ptr<float4> DeviceVelocities((float4*)Velocities);

        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(DevicePositions, DeviceVelocities)),
            thrust::make_zip_iterator(thrust::make_tuple(DevicePositions + NumParticles, DeviceVelocities + NumParticles)),
            IntegrateFunctor());
    }

    void CudaCalculateHashes(uint* OutGridParticleHashes,
        uint* OutGridParticleIndices,
        float* Positions,
        uint   NumParticles)
    {
        uint NumThreads;
        uint NumBlocks;
        CudaComputeGridSize(NumParticles, 256u, NumBlocks, NumThreads);

        // execute the kernel
        CudaCalculateHashDevice << <NumBlocks, NumThreads >> > (OutGridParticleHashes,
            OutGridParticleIndices,
            (float4*)Positions,
            NumParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void CudaMcCalculateHashes(uint* OutGridParticleHashes,
        uint* OutGridParticleIndices,
        float* Positions,
        uint   NumParticles)
    {
        uint NumThreads;
        uint NumBlocks;
        CudaComputeGridSize(NumParticles, 256u, NumBlocks, NumThreads);

        // execute the kernel
        CudaMcCalculateHashDevice << <NumBlocks, NumThreads >> > (OutGridParticleHashes,
            OutGridParticleIndices,
            (float4*)Positions,
            NumParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void CudaReorderDataAndFindCellStart(uint* OutCellStarts,
        uint* OutCellEnds,
        float* OutSortedPositions,
        float* OutSortedVelocities,
        uint* GridParticleHashes,
        uint* GridParticleIndices,
        float* Positions,
        float* Velocities,
        uint   NumParticles,
        uint   NumCells)
    {
        uint NumThreads;
        uint NumBlocks;
        CudaComputeGridSize(NumParticles, 256u, NumBlocks, NumThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(OutCellStarts, 0xffffffff, NumCells * sizeof(uint)));

        uint SmemSize = sizeof(uint) * (NumThreads + 1);
        CudaReorderDataAndFindCellStartDevice << <NumBlocks, NumThreads, SmemSize >> > (OutCellStarts,
            OutCellEnds,
            (float4*)OutSortedPositions,
            (float4*)OutSortedVelocities,
            GridParticleHashes,
            GridParticleIndices,
            (float4*)Positions,
            (float4*)Velocities,
            NumParticles);
        getLastCudaError("Kernel execution failed: CudaReorderDataAndFindCellStartDevice");

    }

    void CudaMcReorderDataAndFindCellStart(uint* OutCellStarts,
        uint* OutCellEnds,
        float* OutSortedPositions,
        uint* GridParticleHashes,
        uint* GridParticleIndices,
        float* Positions,
        uint   NumParticles,
        uint   NumVoxels)
    {
        uint NumThreads;
        uint NumBlocks;
        CudaComputeGridSize(NumParticles, 256u, NumBlocks, NumThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(OutCellStarts, 0xffffffff, NumVoxels * sizeof(uint)));

        uint SmemSize = sizeof(uint) * (NumThreads + 1);
        CudaMcReorderDataAndFindCellStartDevice << <NumBlocks, NumThreads, SmemSize >> > (OutCellStarts,
            OutCellEnds,
            (float4*)OutSortedPositions,
            GridParticleHashes,
            GridParticleIndices,
            (float4*)Positions,
            NumParticles);
        getLastCudaError("Kernel execution failed: CudaReorderDataAndFindCellStartDevice");

    }

    void CudaComputeDensitiesAndPressures(float* OutDensities,
        float* OutPressures,
        float* SortedPositions,
        uint* GridParticleIndices,
        uint* CellStarts,
        uint* CellEnds,
        uint   NumBoundaryParticles,
        uint   NumFluidParticles,
        uint   NumRenderingFluidParticles,
        uint   NumParticles)
    {

        // thread per particle
        uint NumThreads;
        uint NumBlocks;
        CudaComputeGridSize(NumParticles, 64u, NumBlocks, NumThreads);

        // execute the kernel
        CudaComputeDensitiesAndPressuresDevice << <NumBlocks, NumThreads >> > (OutDensities,
            OutPressures,
            (float4*)SortedPositions,
            GridParticleIndices,
            CellStarts,
            CellEnds,
            NumBoundaryParticles,
            NumFluidParticles,
            NumRenderingFluidParticles,
            NumParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");

    }

    void CudaComputeAllForcesAndVelocities(float* OutVelocities,
        float* OutForces,
        float* OutPressureForces,
        float* OutViscosityForces,
        float* SortedPositions, // input: sorted positions
        float* SortedVelocities,
        float* Densities,
        float* Pressures,
        uint* GridParticleIndices,    // input: sorted particle indices
        uint* CellStarts,
        uint* CellEnds,
        uint   NumFluidParticles,
        uint   NumRenderingFluidParticles,
        uint   NumParticles)
    {
        // thread per particle
        uint NumThreads;
        uint NumBlocks;
        CudaComputeGridSize(NumParticles, 64u, NumBlocks, NumThreads);

        // execute the kernel
        CudaComputeAllForcesAndVelocitiesDevice << <NumBlocks, NumThreads >> > ((float4*)OutVelocities,
            (float4*)OutForces,
            (float4*)OutPressureForces,
            (float4*)OutViscosityForces,
            (float4*)SortedPositions,   // input: sorted positions
            (float4*)SortedVelocities,
            Densities,
            Pressures,
            GridParticleIndices,         // input: sorted particle indices
            CellStarts,
            CellEnds,
            NumFluidParticles,
            NumRenderingFluidParticles,
            NumParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void CudaComputeForcesAndVelocities(float* OutVelocities,
        float* OutForces,
        float* SortedPositions, // input: sorted positions
        float* SortedVelocities,
        float* Densities,
        float* Pressures,
        uint* GridParticleIndices,    // input: sorted particle indices
        uint* CellStarts,
        uint* CellEnds,
        uint   NumFluidParticles,
        uint   NumRenderingFluidParticles,
        uint   NumParticles)
    {
        // thread per particle
        uint NumThreads;
        uint NumBlocks;
        CudaComputeGridSize(NumParticles, 64u, NumBlocks, NumThreads);

        // execute the kernel
        CudaComputeForcesAndVelocitiesDevice << <NumBlocks, NumThreads >> > ((float4*)OutVelocities,
            (float4*)OutForces,
            (float4*)SortedPositions,   // input: sorted positions
            (float4*)SortedVelocities,
            Densities,
            Pressures,
            GridParticleIndices,         // input: sorted particle indices
            CellStarts,
            CellEnds,
            NumFluidParticles,
            NumRenderingFluidParticles,
            NumParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }


    void CudaSortParticles(uint* DeviceGridParticleHashes, uint* DeviceGridParticleIndices, uint NumParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(DeviceGridParticleHashes),
            thrust::device_ptr<uint>(DeviceGridParticleHashes + NumParticles),
            thrust::device_ptr<uint>(DeviceGridParticleIndices));
    }
#pragma endregion

#pragma region MarchingCubesExtern
    // marching cubes
    void CudaAllocateTextures(uint** DeviceEdgeTable, uint** DeviceTriTable, uint** DeviceNumVerticesTable)
    {
        checkCudaErrors(cudaMalloc((void**)DeviceEdgeTable, 256u * sizeof(uint)));
        checkCudaErrors(cudaMemcpy((void*)*DeviceEdgeTable, (void*)gEdgesTable, 256 * sizeof(uint), cudaMemcpyHostToDevice));
        cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

        checkCudaErrors(cudaMalloc((void**)DeviceTriTable, 256u * 16u * sizeof(uint)));
        checkCudaErrors(cudaMemcpy((void*)*DeviceTriTable, (void*)gTrianglesTable, 256u * 16u * sizeof(uint), cudaMemcpyHostToDevice));

        cudaResourceDesc            TextureResource;
        memset(&TextureResource, 0, sizeof(cudaResourceDesc));

        TextureResource.resType = cudaResourceTypeLinear;
        TextureResource.res.linear.devPtr = *DeviceTriTable;
        TextureResource.res.linear.sizeInBytes = 256u * 16u * sizeof(uint);
        TextureResource.res.linear.desc = ChannelDesc;

        cudaTextureDesc             TexDesc;
        memset(&TexDesc, 0, sizeof(cudaTextureDesc));

        TexDesc.normalizedCoords = false;
        TexDesc.filterMode = cudaFilterModePoint;
        TexDesc.addressMode[0] = cudaAddressModeClamp;
        TexDesc.readMode = cudaReadModeElementType;

        checkCudaErrors(cudaCreateTextureObject(&gTriTex, &TextureResource, &TexDesc, nullptr));

        checkCudaErrors(cudaMalloc((void**)DeviceNumVerticesTable, 256u * sizeof(uint)));
        checkCudaErrors(cudaMemcpy((void*)*DeviceNumVerticesTable, (void*)gNumVerticesTable, 256u * sizeof(uint), cudaMemcpyHostToDevice));

        memset(&TextureResource, 0, sizeof(cudaResourceDesc));

        TextureResource.resType = cudaResourceTypeLinear;
        TextureResource.res.linear.devPtr = *DeviceNumVerticesTable;
        TextureResource.res.linear.sizeInBytes = 256 * sizeof(uint);
        TextureResource.res.linear.desc = ChannelDesc;

        memset(&TexDesc, 0, sizeof(cudaTextureDesc));

        TexDesc.normalizedCoords = false;
        TexDesc.filterMode = cudaFilterModePoint;
        TexDesc.addressMode[0] = cudaAddressModeClamp;
        TexDesc.readMode = cudaReadModeElementType;

        checkCudaErrors(cudaCreateTextureObject(&gNumVerticesTex, &TextureResource, &TexDesc, nullptr));
    }

    void CudaCreateVolumeTexture(uchar* DeviceVolumes, size_t BufferSize)
    {
        cudaResourceDesc            TextureResourceDesc;
        memset(&TextureResourceDesc, 0, sizeof(cudaResourceDesc));

        TextureResourceDesc.resType = cudaResourceTypeLinear;
        TextureResourceDesc.res.linear.devPtr = DeviceVolumes;
        TextureResourceDesc.res.linear.sizeInBytes = BufferSize;
        TextureResourceDesc.res.linear.desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

        cudaTextureDesc             TextureDesc;
        memset(&TextureDesc, 0, sizeof(cudaTextureDesc));

        TextureDesc.normalizedCoords = false;
        TextureDesc.filterMode = cudaFilterModePoint;
        TextureDesc.addressMode[0] = cudaAddressModeClamp;
        TextureDesc.readMode = cudaReadModeNormalizedFloat;

        checkCudaErrors(cudaCreateTextureObject(&gVolumeTex, &TextureResourceDesc, &TextureDesc, nullptr));
    }

    void CudaDestroyAllTextureObjects()
    {
        checkCudaErrors(cudaDestroyTextureObject(gTriTex));
        checkCudaErrors(cudaDestroyTextureObject(gNumVerticesTex));
        checkCudaErrors(cudaDestroyTextureObject(gVolumeTex));
    }

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
        uint NumRenderingFluidParticles)
    {
        // calculate number of vertices need per voxel
        CudaClassifyVoxels << <Grid, Threads >> > (OutVoxelVertices,
            OutOccupiedVoxels,
            Volumes,
            GridSize,
            GridSizeShift,
            GridSizeMask,
            NumVoxels,
            VoxelSize,
            IsoValue,
            gNumVerticesTex,
            gVolumeTex,
            reinterpret_cast<float4*>(SortedPositions),
            GridParticleIndices,
            CellStarts,
            CellEnds,
            NumFluidParticles,
            NumRenderingFluidParticles);
        getLastCudaError("CudaClassifyVoxels failed");
    }

    void CudaLaunchCompactVoxels(dim3 Grid, dim3 Threads, uint* OutCompactedVoxelArray, uint* OccupiedVoxels, uint* OccupiedScanVoxels, uint NumVoxels)
    {
        CudaCompactVoxels << <Grid, Threads >> > (OutCompactedVoxelArray,
            OccupiedVoxels,
            OccupiedScanVoxels,
            NumVoxels);
        getLastCudaError("CudaCompactVoxels failed");
    }

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
        uint NumRenderingFluidParticles)
    {
        CudaGenerateTriangles << <Grid, Threads >> > (OutPositions,
            OutNormals,
            CompactedVoxelArray,
            NumScannedVertices,
            GridSize,
            GridSizeShift,
            GridSizeMask,
            VoxelSize,
            IsoValue,
            NumActiveVoxels,
            NumMaxVertices,
            gTriTex,
            gNumVerticesTex,
            reinterpret_cast<float4*>(SortedPositions),
            GridParticleIndices,
            CellStarts,
            CellEnds,
            NumFluidParticles,
            NumRenderingFluidParticles);
        getLastCudaError("CudaGenerateTriangles failed");
    }

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
        uint NumRenderingFluidParticles)
    {
        CudaGenerateTriangles2 << <Grid, Threads >> > (OutPositions,
            OutNormals,
            CompactedVoxelArray,
            NumScannedVertices,
            Volumes,
            GridSize,
            GridSizeShift,
            GridSizeMask,
            VoxelSize,
            IsoValue,
            NumActiveVoxels,
            NumMaxVertices,
            gTriTex,
            gNumVerticesTex,
            gVolumeTex,
            SortedPositions,
            GridParticleIndices,
            CellStarts,
            CellEnds,
            NumFluidParticles,
            NumRenderingFluidParticles);
        getLastCudaError("CudaGenerateTriangles2 failed");
    }

    void CudaThrustScanWrapper(unsigned int* Outputs, unsigned int* Inputs, unsigned int NumElements)
    {
        thrust::exclusive_scan(thrust::device_ptr<unsigned int>(Inputs),
            thrust::device_ptr<unsigned int>(Inputs + NumElements),
            thrust::device_ptr<unsigned int>(Outputs));
    }

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
        uint* CellEnds)
    {
        CudaCreateVolumeFromMassAndDensitiesDevice << <Grid, Threads >> > (OutVolumes,
            GridSize,
            GridSizeShift,
            GridSizeMask,
            VoxelSize,
            NumFluidParticles,
            SortedPositions,
            GridParticleIndices,
            CellStarts,
            CellEnds);
    }
}   // extern "C"
#pragma endregion