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

#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#include "vector_types.h"
typedef unsigned int uint;

// Max number of editor-placed box colliders the fluid avoids (OBB collision).
#define MAX_SPH_COLLIDERS 16

// simulation parameters
struct SimParams
{
    float3 colliderPos;
    float  colliderRadius;

    float3 gravity;
    float globalDamping;
    float particleRadius;

    uint3 gridSize;
    uint numCells;
    float3 worldOrigin;
    float3 cellSize;

    uint numBodies;
    uint maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;

    float smoothingLength;
    float fluidParticleMass;
    float boundaryParticleMass;

    float gasStiffnessConstant;
    float waterRestDensity;
    float viscosityCoefficient;
    float restitutionCoefficient;
    float surfaceTension;

    // Direct boundary repulsion: pushes fluid away from boundary particles
    // regardless of pressure (so low-pressure surface fluid can't pass through).
    float boundaryForceScale;

    //Marching Cube
    float CflScale;

    float Poly6;
    float Poly6Gradient;
    float Poly6Laplacian;
    float SpikyGradient;
    float ViscosityLaplacian;

    float DeltaTime;

    float XScaleFactor;
    float YScaleFactor;
    float ZScaleFactor;

    int MarchingCubesNeighborSearchDepth;

    // ---- Editor-placed box colliders (oriented boxes, in SIM space) ----
    // The fluid does OBB collision against these so it avoids cubes placed in
    // the level. All in sim space (the actor converts from UE world each frame).
    int    numColliders;
    float3 colliderCenter[MAX_SPH_COLLIDERS];
    float3 colliderAxisX[MAX_SPH_COLLIDERS];      // unit local axes (orientation)
    float3 colliderAxisY[MAX_SPH_COLLIDERS];
    float3 colliderAxisZ[MAX_SPH_COLLIDERS];
    float3 colliderHalfExtent[MAX_SPH_COLLIDERS]; // half-size along each local axis
};

#endif
