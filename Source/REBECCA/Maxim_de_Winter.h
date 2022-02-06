// Fill out your copyright notice in the Description page of Project Settings.
#define WIN32_LEAN_AND_MEAN

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"

#include "ProceduralMeshComponent.h"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "helper_math.h"

#include "particles_kernel.cuh"
#include "vector_functions.h"
#include "defines.h"
#include "particleSystem.cuh"

#include "Maxim_de_Winter.generated.h"

#ifndef CUDART_PI_F
#define CUDART_PI_F       3.141592654f
#endif

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

#define GRID_SIZE       (64u)
#define NUM_PARTICLES   (10000u)
#define NUM_BOUNDARY_PARTICLES (5000u)


UCLASS(Blueprintable)
class REBECCA_API AMaxim_de_Winter : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AMaxim_de_Winter();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

private:
	float tempNum;

	float particleMeshRadius;
	float particleRenderRadius;
	float particleRadius;

protected:
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Components")
		class UInstancedStaticMeshComponent* ParticleInstancedMeshComponent = nullptr;
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Components")
		class UInstancedStaticMeshComponent* BoundaryParticleInstancedMeshComponent = nullptr;
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Components")
		UProceduralMeshComponent* ParticleProceduralMeshComponent = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Components")
		class UMaterialInterface* Material = nullptr;

public:
    enum ParticleArray
    {
        POSITION,
        VELOCITY,
        FORCE,
        DENSITY,
        PRESSURE
    };

    void reset();

    void   setArray(ParticleArray array, const float* data, int start, int count);

    int    getNumParticles() const
    {
        return m_numParticles;
    }

    float getParticleRadius()
    {
        return m_params.particleRadius;
    }
    float3 getColliderPos()
    {
        return m_params.colliderPos;
    }
    float getColliderRadius()
    {
        return m_params.colliderRadius;
    }
    uint3 getGridSize()
    {
        return m_params.gridSize;
    }
    float3 getWorldOrigin()
    {
        return m_params.worldOrigin;
    }
    float3 getCellSize()
    {
        return m_params.cellSize;
    }

    void addSphere(int index, float* pos, float* vel, int r, float spacing);
    void addCollideSphere(int index, float* pos, float* vel, int r, float spacing);
    void addBoundaryCube(int start, float* pos, float* width, float* vel, float particleDiameter, uint* addedBoundaryParticles);
    void _addBoudaryCube();
    void addFluidCube(int start, float* pos, float* width, float* vel, float particleDiameter);
    UFUNCTION(BlueprintCallable, Category = Buttonfunction)
        void _addFluidCube();

protected: // methods

    void initialize(int numParticles);
    void finalize();

    void initGrid(uint* size, float spacing, float jitter, uint numParticles);

protected: // data
    bool m_bInitialized, m_bUseOpenGL;
    uint32 m_numFluidParticles;
    uint32 m_numParticles;

    float widthScaling;
    float radiusScaling;

    FVector Gravity = FVector(0.0f, 0.0f, -9.80665f);

    // CPU data
    float* m_hPos;              // particle positions
    float* m_hVel;              // particle velocities

    float* m_hDensities;
    float* m_hPressures;
    float* m_hEntireForces;

    uint32* m_hParticleHash;
    uint32* m_hCellStart;
    uint32* m_hCellEnd;

    // GPU data
    float* m_dPos;
    float* m_dVel;

    float* m_dDensities;
    float* m_dPressures;
    float* m_dEntireForces;

    float* m_dSortedPos;
    float* m_dSortedVel;

    // grid data for sorting method
    uint32* m_dGridParticleHash; // grid hash value for each particle
    uint32* m_dGridParticleIndex;// particle index for each particle
    uint32* m_dCellStart;        // index of start of each cell in sorted list
    uint32* m_dCellEnd;          // index of end of cell

    uint32   m_gridSortBits;

    uint32   m_posVbo;            // vertex buffer object for particle positions
    uint32   m_colorVBO;          // vertex buffer object for colors

    float* m_cudaPosVBO;        // these are the CUDA deviceMem Pos
    float* m_cudaColorVBO;      // these are the CUDA deviceMem Color

    struct cudaGraphicsResource* m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource* m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

    // params
    SimParams m_params;
    uint3 m_gridSize;
    uint32 m_numGridCells;

    uint32 addedBoundaryParticles;
    uint32 addedFluidPraticles;

    StopWatchInterface* m_timer;

    uint32 m_solverIterations;

};
