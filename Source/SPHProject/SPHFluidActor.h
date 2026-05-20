// SPH fluid simulation actor (ported from the UE4 AMaxim_de_Winter).
// Runs the CUDA SPH pipeline every Tick and visualizes particles with an
// InstancedStaticMeshComponent. Marching-cubes surface reconstruction from the
// old project was intentionally dropped.
#define WIN32_LEAN_AND_MEAN

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include "helper_math.h"

#include "particles_kernel.cuh"
#include "vector_functions.h"
#include "particleSystem.cuh"

#include "SPHFluidActor.generated.h"

#ifndef CUDART_PI_F
#define CUDART_PI_F       3.141592654f
#endif

#define GRID_SIZE       (64u)
#define NUM_PARTICLES   (2000u)
#define NUM_BOUNDARY_PARTICLES (2000u)

UCLASS(Blueprintable)
class SPHPROJECT_API ASPHFluidActor : public AActor
{
	GENERATED_BODY()

public:
	ASPHFluidActor();

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:
	virtual void Tick(float DeltaTime) override;

private:
	float particleMeshRadius;
	float particleRenderRadius;
	float particleRadius;

protected:
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Components")
		class UInstancedStaticMeshComponent* ParticleInstancedMeshComponent = nullptr;
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Components")
		class UInstancedStaticMeshComponent* BoundaryParticleInstancedMeshComponent = nullptr;

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

	void   setArray(ParticleArray array, const float* data, int start, int count);

	int    getNumParticles() const { return m_numParticles; }
	float  getParticleRadius() { return m_params.particleRadius; }
	float3 getColliderPos() { return m_params.colliderPos; }
	float  getColliderRadius() { return m_params.colliderRadius; }
	uint3  getGridSize() { return m_params.gridSize; }
	float3 getWorldOrigin() { return m_params.worldOrigin; }
	float3 getCellSize() { return m_params.cellSize; }

	void addBoundaryCube(int start, float* pos, float* width, float* vel, float particleDiameter, uint* addedBoundaryParticles);
	void _addBoudaryCube();
	void addFluidCube(int start, float* pos, float* width, float* vel, float particleDiameter);
	UFUNCTION(BlueprintCallable, Category = Buttonfunction)
		void _addFluidCube();
	void _addBoudaryCube2();
	void addBoundaryRotateCube(int start, float* pos, float* width, float* vel, float rotate, float particleDiameter, uint* addedBParticles);

protected: // methods
	void initialize(int numParticles);
	void finalize();

	UFUNCTION(BlueprintCallable, Category = "Custom")
		void reset();
	UFUNCTION(BlueprintCallable, Category = "Custom")
		void resetStart();
	void initGrid(uint* size, float spacing, float jitter, uint numParticles);
	void initStartGrid(uint* size, float spacing, float jitter, uint numParticles);

protected: // data
	bool m_bInitialized;
	uint32 m_numFluidParticles;
	uint32 m_numParticles;

	float widthScaling;
	float widthYScaling;
	float radiusScaling;

	// CPU data
	float* m_hPos;
	float* m_hVel;

	float* m_hDensities;
	float* m_hPressures;
	float* m_hEntireForces;

	uint32* m_hParticleHash;
	uint32* m_hCellStart;
	uint32* m_hCellEnd;

	// GPU data
	float* m_dVel;

	float* m_dDensities;
	float* m_dPressures;
	float* m_dEntireForces;

	float* m_dSortedPos;
	float* m_dSortedVel;

	// grid data for sorting
	uint32* m_dGridParticleHash;
	uint32* m_dGridParticleIndex;
	uint32* m_dCellStart;
	uint32* m_dCellEnd;

	uint32   m_gridSortBits;

	float* m_cudaPosVBO;        // CUDA device positions (float4 per particle)
	float* m_cudaColorVBO;

	// params
	SimParams m_params;
	uint3 m_gridSize;
	uint32 m_numGridCells;

	uint32 addedBoundaryParticles;
	uint32 addedFluidPraticles;

	StopWatchInterface* m_timer;

	uint32 m_solverIterations;

	float timeStep;
};
