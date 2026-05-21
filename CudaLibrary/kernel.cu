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
#include "CudaLibrary.h"

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"


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
        float4 posData = thrust::get<0>(t);
        float4 velData = thrust::get<1>(t);
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

        // Geometric tilted-floor (slide) collision — fluid only (posData.w==1; the
        // boundary particles, w==0, are visual and must keep their slab shape).
        // Hard no-penetration constraint: project onto the surface, cancel the into-
        // surface velocity. The plane matches the visual boundary slide in _addBoudaryCube2
        // (pos -0.7,-0.3,-0.1, tilt 0.4): point on top face, normal (-sin a, cos a, 0).
        // Finite slide footprint: x in [-0.7, 0.5], z in [-0.1, 0.5] (matches the
        // boundary slab). Outside it, no collision -> fluid falls off the edge.
        if (posData.w > 0.5f &&
            pos.x >= -0.7f && pos.x <= 0.5f &&
            pos.z >= -0.1f && pos.z <= 0.5f)
        {
            const float3 rampPoint  = make_float3(-0.7f, -0.25f, 0.0f);
            const float3 rampNormal = make_float3(-0.38942f, 0.92106f, 0.0f);
            float distToRamp = dot(pos - rampPoint, rampNormal);
            // Only resolve shallow penetration (just below the surface). Particles
            // deeper than smoothingLength below the plane have already fallen past /
            // under the slide -> leave them, else they'd teleport back up onto it.
            if (distToRamp < params.particleRadius && distToRamp > -params.smoothingLength)
            {
                pos += (params.particleRadius - distToRamp) * rampNormal;
                float vn = dot(vel, rampNormal);
                if (vn < 0.0f) vel -= vn * rampNormal;   // slide, no bounce
                vel *= 0.99f;                            // mild friction
            }
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
    float*  boundaryPsi,
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
            uint jOriginalIndex = gridParticleIndex[j];
            float3 jPos = make_float3(oldPosArray[j]);
            float3 iToj = indexPos - jPos;
            float length_iToj = length(iToj);

            if (length_iToj < params.smoothingLength)
            {
                // Contribution depends on the NEIGHBOUR's type: fluid uses mass,
                // boundary uses its Akinci effective volume Psi (= mass-equivalent).
                if (jOriginalIndex < numFluidParticles)
                    density += params.fluidParticleMass * kernelPoly6(iToj, params.smoothingLength);
                else
                    density += boundaryPsi[jOriginalIndex] * kernelPoly6(iToj, params.smoothingLength);
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
    float* boundaryPsi,                // input: Akinci boundary volumes (per original index)
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
                density += computeDensityByCell(neighbourPos, index, indexPos, indexVel, originalIndex, oldPosArray, oldVelArray, boundaryPsi, cellStart, cellEnd, gridParticleIndex, numFluidParticles);
            }
        }
    }

    pressure = params.gasStiffnessConstant * (density - params.waterRestDensity);
    pressure = fmaxf(pressure, 0.0f);   // clamp tensile pressure (geometric collision handles containment; this keeps the fluid stable)
    desities[originalIndex] = density;
    pressures[originalIndex] = pressure;
}


// Akinci 2012 boundary handling --------------------------------------------
// Boundary particles have no fixed mass; instead each gets an effective volume
// from its OWN packing, which corrects the one-sided neighbour deficiency at a
// wall. Psi_b = restDensity / sum_k W(b - k) over boundary neighbours (incl self).
// Boundaries are static, so this is computed once.

__device__
float boundaryKernelSumByCell(int3   gridPos,
    float3  indexPos,
    float4* oldPosArray,
    uint*   cellStart,
    uint*   cellEnd,
    uint*   gridParticleIndex,
    uint    numFluidParticles)
{
    uint gridHash = calcGridHash(gridPos);
    uint startIndex = cellStart[gridHash];

    float sum = 0.f;
    if (startIndex != 0xffffffff)
    {
        uint endIndex = cellEnd[gridHash];
        for (uint j = startIndex; j < endIndex; j++)
        {
            if (gridParticleIndex[j] < numFluidParticles) continue;   // boundary neighbours only
            float3 jPos = make_float3(oldPosArray[j]);
            float3 rij = indexPos - jPos;
            if (length(rij) < params.smoothingLength)
                sum += kernelPoly6(rij, params.smoothingLength);
        }
    }
    return sum;
}

__global__
void computeBoundaryPsiDevice(float*  boundaryPsi,
    float4* oldPosArray,
    uint*   gridParticleIndex,
    uint*   cellStart,
    uint*   cellEnd,
    uint    numParticles,
    uint    numFluidParticles)
{
    uint index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;

    uint originalIndex = gridParticleIndex[index];
    if (originalIndex < numFluidParticles) return;   // boundary particles only

    float3 indexPos = make_float3(oldPosArray[index]);
    int3 gridPos = calcGridPos(indexPos);

    float kernelSum = 0.f;
    for (int z = -1; z <= 1; z++)
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++)
                kernelSum += boundaryKernelSumByCell(gridPos + make_int3(x, y, z), indexPos, oldPosArray, cellStart, cellEnd, gridParticleIndex, numFluidParticles);

    boundaryPsi[originalIndex] = (kernelSum > 1e-8f) ? (params.waterRestDensity / kernelSum) : params.boundaryParticleMass;
}


// Compute Froce
__device__
void computeForceAndViscosityByCell(float3* pressureForce,
    float3* viscosity,
    float*  boundaryPsi,
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

            if (j != index && lengthRij < params.smoothingLength)  // not self, within support
            {
                float3 spiky = kernelSpikyGradient(iTojPos, params.smoothingLength);

                if (jOriginalIndex < numFluidParticles)
                { // fluid neighbour: standard symmetric pressure + viscosity
                    *pressureForce += params.fluidParticleMass
                        * (pressures[originalIndex] / (densities[originalIndex] * densities[originalIndex]) + pressures[jOriginalIndex] / (densities[jOriginalIndex] * densities[jOriginalIndex]))
                        * spiky;

                    *viscosity += (params.fluidParticleMass / densities[originalIndex])
                        * (dot(iTojPos, iTojVel) / (dot(iTojPos, iTojPos) + 0.01f * params.smoothingLength * params.smoothingLength))
                        * kernelPoly6Gradient(iTojPos, params.smoothingLength);
                }
                else
                { // boundary neighbour (Akinci): effective mass = Psi, pressure mirrored
                  // from the fluid particle itself -> the wall pushes back with the
                  // fluid's own pressure (physical no-penetration, no penalty fudge).
                  // Clamp to >=0 so the wall only ever PUSHES (never pulls fluid in
                  // when the fluid is under tension / negative pressure).
                    float pBoundary = fmaxf(pressures[originalIndex], 0.0f);
                    *pressureForce += boundaryPsi[jOriginalIndex]
                        * (pBoundary / (densities[originalIndex] * densities[originalIndex]))
                        * spiky;
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
    float* boundaryPsi,                // input: Akinci boundary volumes (per original index)
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
                computeForceAndViscosityByCell(&pressureForce, &viscosity, boundaryPsi, neighbourPos, index, indexPos, indexVel, originalIndex, oldPosArray, oldVelArray, densities, pressures, cellStart, cellEnd, gridParticleIndex, numFluidParticles);
            }
        }
    }

    pressureForce *= -params.fluidParticleMass;
    viscosity *= params.fluidParticleMass * params.viscosityCoefficient * 10.f / densities[originalIndex]; // 2(d+2)
    externalForce = params.gravity * params.fluidParticleMass;  //params.gravity * densities[originalIndex]; // params.gravity * params.particleMass;
    entireForce = pressureForce + viscosity + externalForce;

    if (originalIndex < numFluidParticles)
    {
        entireForces[originalIndex] = make_float4(entireForce, 0.0f);
        newVelocities[originalIndex] += make_float4(deltaTime * (entireForce / params.fluidParticleMass), 0.0f);
    }
    else
    {
        // Boundary particles are static walls: they contribute to fluid density/
        // pressure (computed above) but must never move themselves, so pin their
        // velocity (and stored force) to zero instead of accumulating gravity.
        entireForces[originalIndex] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        newVelocities[originalIndex] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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

    // Akinci boundary volumes: compute once (boundaries are static).
    void computeBoundaryPsi(float* boundaryPsi,
        float* sortedPos,
        uint*  gridParticleIndex,
        uint*  cellStart,
        uint*  cellEnd,
        uint   numParticles,
        uint   numFluidParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        computeBoundaryPsiDevice << < numBlocks, numThreads >> > (boundaryPsi,
            (float4*)sortedPos,
            gridParticleIndex,
            cellStart,
            cellEnd,
            numParticles,
            numFluidParticles);

        getLastCudaError("computeBoundaryPsi failed");
    }

    void computeDensityAndPressure(float* densities,
        float* pressures,
        float* boundaryPsi,
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
            boundaryPsi,
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
        float* boundaryPsi,
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
            boundaryPsi,
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


// ---------------------------------------------------------------------
// CUDA smoke test (kept as a sanity check; used by CudaSmokeTestActor).
// ---------------------------------------------------------------------
__global__ void smokeAddKernel(int* c, const int* a, const int* b, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        c[i] = a[i] + b[i];
    }
}

extern "C" int CudaAddArrays(const int* a, const int* b, int* c, int count)
{
    int* devA = nullptr;
    int* devB = nullptr;
    int* devC = nullptr;
    int rc = 0;

    if (cudaSetDevice(0) != cudaSuccess)
    {
        return -1;
    }

    const size_t bytes = static_cast<size_t>(count) * sizeof(int);

    if (cudaMalloc(&devA, bytes) != cudaSuccess ||
        cudaMalloc(&devB, bytes) != cudaSuccess ||
        cudaMalloc(&devC, bytes) != cudaSuccess)
    {
        rc = -3;
        goto cleanup;
    }

    if (cudaMemcpy(devA, a, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(devB, b, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        rc = -3;
        goto cleanup;
    }

    {
        const int threads = 256;
        const int blocks = (count + threads - 1) / threads;
        smokeAddKernel<<<blocks, threads>>>(devC, devA, devB, count);
    }

    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
    {
        rc = -2;
        goto cleanup;
    }

    if (cudaMemcpy(c, devC, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        rc = -3;
    }

cleanup:
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    return rc;
}

extern "C" int CudaGetDeviceName(char* outName, int bufferSize)
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        return -1;
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
    {
        return -1;
    }

    if (outName && bufferSize > 0)
    {
        strncpy(outName, prop.name, bufferSize - 1);
        outName[bufferSize - 1] = '\0';
    }
    return 0;
}
