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
#define NUM_PARTICLES   (10000u)
#define NUM_BOUNDARY_PARTICLES (30000u)

class USPHColliderComponent;

UCLASS(Blueprintable)
class SPHPROJECT_API ASPHFluidActor : public AActor
{
	GENERATED_BODY()

public:
	ASPHFluidActor();

protected:
	virtual void BeginPlay() override;
	virtual void OnConstruction(const FTransform& Transform) override;
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
	void ApplyParticleMaterial();
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

	// Tunable live in the editor (Details panel) — no recompile needed.
	UPROPERTY(EditAnywhere, Category = "SPH|Visual")
		float widthScaling;       // horizontal/vertical spread (sim -> UE units)
	UPROPERTY(EditAnywhere, Category = "SPH|Visual")
		float widthYScaling;      // depth spread (smaller = flatter slab)
	UPROPERTY(EditAnywhere, Category = "SPH|Visual")
		float radiusScaling;      // rendered particle size

	// Simulation speed: dt = min(DeltaTime * SimSpeed, MaxStepTime). FPS-independent.
	UPROPERTY(EditAnywhere, Category = "SPH|Sim", meta = (ClampMin = "0.0"))
		float SimSpeed = 1.0f;    // 1 = real-time, <1 = slow motion
	UPROPERTY(EditAnywhere, Category = "SPH|Sim", meta = (ClampMin = "0.001", ClampMax = "0.05"))
		float MaxStepTime = 0.03f; // stability clamp on the integration step

	// SPH physics — tune live to fix instability. pressure = GasStiffness*(density - RestDensity).
	// If the fluid explodes, the kernel's rest-spacing density exceeds RestDensity:
	// raise RestDensity (toward the real value, ~2000+) and/or lower GasStiffness.
	UPROPERTY(EditAnywhere, Category = "SPH|Physics", meta = (ClampMin = "1.0"))
		float RestDensity = 998.29f;
	UPROPERTY(EditAnywhere, Category = "SPH|Physics", meta = (ClampMin = "0.0"))
		float GasStiffness = 3.0f;
	UPROPERTY(EditAnywhere, Category = "SPH|Physics", meta = (ClampMin = "0.0"))
		float Viscosity = 3.5f;
	UPROPERTY(EditAnywhere, Category = "SPH|Debug")
		bool bDrawColliderBounds = true;

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
	float* m_dBoundaryPsi;        // Akinci boundary effective volumes (computed once)
	bool   m_boundaryPsiReady;    // false until Psi has been computed on the first tick

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

	// reusable buffers for batched instance-transform updates (avoid per-frame realloc)
	TArray<FTransform> m_fluidTransforms;
	TArray<FTransform> m_boundaryTransforms;

	// Editor-placed cube colliders (actors carrying a USPHColliderComponent), gathered
	// at BeginPlay and read every frame so the fluid avoids them (and they can move).
	UPROPERTY()
	TArray<USPHColliderComponent*> m_colliders;

	void GatherColliders();
	void UpdateColliderParams();   // world boxes -> sim-space OBBs in m_params
};
