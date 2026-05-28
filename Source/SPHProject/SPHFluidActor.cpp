#include "SPHFluidActor.h"

#include "Components/InstancedStaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "UObject/ConstructorHelpers.h"
#include "SPHColliderComponent.h"
#include "EngineUtils.h"   // TActorIterator
#include "DrawDebugHelpers.h"

#include <random>

ASPHFluidActor::ASPHFluidActor()
{
	PrimaryActorTick.bCanEverTick = true;

	particleMeshRadius = 50.0f;
	particleRenderRadius = 10.0f;
	widthScaling = 500.f;
	widthYScaling = 100.f;
	radiusScaling = 0.4f;

	// Particle render component (root) + boundary render component.
	ParticleInstancedMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("ParticleInstancedMeshComponent"));
	RootComponent = ParticleInstancedMeshComponent;
	BoundaryParticleInstancedMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("BoundaryParticleInstancedMeshComponent"));
	BoundaryParticleInstancedMeshComponent->SetupAttachment(RootComponent);

	// Default to the engine's basic sphere so particles are visible without
	// any editor setup. Override the mesh/material in the placed actor if desired.
	static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereMesh(TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	if (SphereMesh.Succeeded())
	{
		ParticleInstancedMeshComponent->SetStaticMesh(SphereMesh.Object);
		BoundaryParticleInstancedMeshComponent->SetStaticMesh(SphereMesh.Object);
	}

	// Render-only particles: drop the per-frame costs. Shadows off (thousands of
	// dynamic shadow casters are the biggest render cost), no collision, no
	// navigation, Movable so transform updates are cheap.
	for (UInstancedStaticMeshComponent* C : { ParticleInstancedMeshComponent, BoundaryParticleInstancedMeshComponent })
	{
		C->SetMobility(EComponentMobility::Movable);
		C->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		C->SetCanEverAffectNavigation(false);
		C->SetCastShadow(false);
	}

	m_bInitialized = false;
	m_boundaryPsiReady = false;
	m_numParticles = NUM_PARTICLES + NUM_BOUNDARY_PARTICLES;
	m_numFluidParticles = NUM_PARTICLES;
	m_hPos = nullptr;
	m_hVel = nullptr;
	m_gridSize = make_uint3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
	m_timer = nullptr;
	m_solverIterations = 1;

	m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;
	m_gridSortBits = 18;

	timeStep = 0.03f;

	// simulation parameters
	m_params.gridSize = m_gridSize;
	m_params.numCells = m_numGridCells;
	m_params.numBodies = m_numParticles;

	m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
	m_params.colliderRadius = 0.2f;

	m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);

	m_params.spring = 0.5f;
	m_params.damping = 0.02f;
	m_params.shear = 0.1f;
	m_params.attraction = 0.0f;
	m_params.boundaryDamping = -0.5f;

	m_params.gravity = make_float3(0.0f, -9.81f, 0.0f);
	m_params.globalDamping = 1.0f;

	m_params.fluidParticleMass = 0.02f;
	m_params.boundaryParticleMass = 0.02f;
	m_params.waterRestDensity = 998.29f;
	m_params.particleRadius = 0.01f;
	m_params.viscosityCoefficient = 3.5f;
	m_params.gasStiffnessConstant = 3.f;
	m_params.restitutionCoefficient = 0.001f;
	m_params.surfaceTension = 0.0728f;
	m_params.smoothingLength = 0.04f;

	addedBoundaryParticles = 0;
	addedFluidPraticles = 0;

	// Cell size MUST be >= smoothingLength: the 3x3x3 neighbour search only reaches
	// ~1.5 cells, so cellSize 0.02 < smoothingLength 0.04 missed neighbours in
	// [0.03, 0.04] -> fluid couldn't "see" the boundary until overlapping -> it
	// passed straight through. Matching cellSize to smoothingLength finds all neighbours.
	float cellSize = m_params.smoothingLength;
	m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

	m_params.numColliders = 0;
}

void ASPHFluidActor::BeginPlay()
{
	Super::BeginPlay();

	ApplyParticleMaterial();

	initialize(m_numParticles);
	resetStart();

	// Structure is now defined by editor-placed cube colliders (no boundary particles).
	m_numParticles = m_numFluidParticles;
	GatherColliders();

	ParticleInstancedMeshComponent->MarkRenderStateDirty();
}

void ASPHFluidActor::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);

	ApplyParticleMaterial();
}

void ASPHFluidActor::ApplyParticleMaterial()
{
	if (!Material)
	{
		return;
	}

	if (ParticleInstancedMeshComponent)
	{
		ParticleInstancedMeshComponent->SetMaterial(0, Material);
	}
	if (BoundaryParticleInstancedMeshComponent)
	{
		BoundaryParticleInstancedMeshComponent->SetMaterial(0, Material);
	}
}

void ASPHFluidActor::GatherColliders()
{
	m_colliders.Reset();
	for (TActorIterator<AActor> It(GetWorld()); It; ++It)
	{
		if (USPHColliderComponent* Col = It->FindComponentByClass<USPHColliderComponent>())
		{
			m_colliders.Add(Col);
			FVector Wc, WHalf; FQuat Wr;
			if (Col->GetWorldBox(Wc, Wr, WHalf))
			{
				UE_LOG(LogTemp, Warning, TEXT("[SPH] Collider %s center=%s halfExtent=%s"),
					*It->GetName(), *Wc.ToString(), *WHalf.ToString());
			}
			if (m_colliders.Num() >= MAX_SPH_COLLIDERS) break;
		}
	}
}

void ASPHFluidActor::UpdateColliderParams()
{
	const float S = widthScaling;                 // uniform sim->world scale
	const FVector ActorLoc = GetActorLocation();
	int32 n = 0;

	for (USPHColliderComponent* Col : m_colliders)
	{
		if (!Col || n >= MAX_SPH_COLLIDERS) break;

		FVector Wc, WHalf; FQuat Wr;
		if (!Col->GetWorldBox(Wc, Wr, WHalf)) continue;

		// world -> sim: relative to the actor, scaled by 1/S, with UE Z(up)<->sim Y
		// and UE Y(depth)<->sim Z swapped.
		const FVector Sc((Wc.X - ActorLoc.X) / S, (Wc.Z - ActorLoc.Z) / S, (Wc.Y - ActorLoc.Y) / S);
		const FVector Ax = Wr.RotateVector(FVector::ForwardVector);  // local X
		const FVector Ay = Wr.RotateVector(FVector::RightVector);    // local Y
		const FVector Az = Wr.RotateVector(FVector::UpVector);       // local Z

		m_params.colliderCenter[n]     = make_float3(Sc.X, Sc.Y, Sc.Z);
		m_params.colliderAxisX[n]      = make_float3(Ax.X, Ax.Z, Ax.Y);   // dir swap Y<->Z
		m_params.colliderAxisY[n]      = make_float3(Ay.X, Ay.Z, Ay.Y);
		m_params.colliderAxisZ[n]      = make_float3(Az.X, Az.Z, Az.Y);
		m_params.colliderHalfExtent[n] = make_float3(WHalf.X / S, WHalf.Y / S, WHalf.Z / S);

		if (bDrawColliderBounds)
		{
			DrawDebugBox(GetWorld(), Wc, WHalf, Wr, FColor::Cyan, false, 0.0f, 0, 2.0f);
		}
		++n;
	}
	m_params.numColliders = n;
}

void ASPHFluidActor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (m_bInitialized)
	{
		finalize();
		m_bInitialized = false;
	}
	Super::EndPlay(EndPlayReason);
}

void ASPHFluidActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (!m_bInitialized)
	{
		return;
	}

	// FPS-independent timestep: scale by real frame time, clamp for stability.
	// (Previously a fixed 0.03 per frame -> sim speed scaled with FPS.)
	const float dt = FMath::Min(DeltaTime * SimSpeed, MaxStepTime);

	// Push live-tunable physics params to the GPU (editable in the actor Details panel).
	m_params.waterRestDensity     = RestDensity;
	m_params.gasStiffnessConstant = GasStiffness;
	m_params.viscosityCoefficient = Viscosity;

	// Refresh the cube colliders each frame (so they can be moved/animated).
	UpdateColliderParams();

	float* dPos = (float*)m_cudaPosVBO;

	setParameters(&m_params);

	calcHash(m_dGridParticleHash, m_dGridParticleIndex, dPos, m_numParticles);

	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

	reorderDataAndFindCellStart(
		m_dCellStart, m_dCellEnd, m_dSortedPos, m_dSortedVel,
		m_dGridParticleHash, m_dGridParticleIndex, dPos, m_dVel,
		m_numParticles, m_numGridCells);

	// Akinci boundary volumes depend only on (static) boundary positions, so
	// compute them once using the first frame's neighbour grid.
	if (!m_boundaryPsiReady)
	{
		computeBoundaryPsi(
			m_dBoundaryPsi, m_dSortedPos,
			m_dGridParticleIndex, m_dCellStart, m_dCellEnd,
			m_numParticles, m_numFluidParticles);
		m_boundaryPsiReady = true;
	}

	computeDensityAndPressure(
		m_dDensities, m_dPressures, m_dBoundaryPsi, m_dSortedPos, m_dSortedVel,
		m_dGridParticleIndex, m_dCellStart, m_dCellEnd,
		m_numParticles, m_numFluidParticles, m_numGridCells);

	computeForceAndViscosity(
		m_dVel, m_dEntireForces, dt, m_dDensities, m_dPressures, m_dBoundaryPsi,
		m_dSortedPos, m_dSortedVel, m_dGridParticleIndex, m_dCellStart, m_dCellEnd,
		m_numParticles, m_numFluidParticles, m_numGridCells);

	integrateSystem(dPos, m_dVel, dt, m_numParticles);

	// --- DEBUG: log fluid density/pressure ~once per second (Output Log) ---
	static float s_debugAccum = 0.f;
	s_debugAccum += DeltaTime;
	if (s_debugAccum >= 1.0f)
	{
		s_debugAccum = 0.f;
		cudaMemcpy(m_hDensities, m_dDensities, sizeof(float) * m_numParticles, cudaMemcpyDeviceToHost);
		cudaMemcpy(m_hPressures, m_dPressures, sizeof(float) * m_numParticles, cudaMemcpyDeviceToHost);
		double sumRho = 0.0, sumP = 0.0; float maxRho = 0.f, maxP = 0.f;
		for (uint32 i = 0; i < m_numFluidParticles; ++i)
		{
			sumRho += m_hDensities[i]; sumP += m_hPressures[i];
			maxRho = FMath::Max(maxRho, m_hDensities[i]); maxP = FMath::Max(maxP, m_hPressures[i]);
		}
		UE_LOG(LogTemp, Warning, TEXT("[SPH] avgRho=%.0f maxRho=%.0f avgP=%.1f maxP=%.1f (RestDensity=%.0f)"),
			sumRho / m_numFluidParticles, maxRho, sumP / m_numFluidParticles, maxP, RestDensity);
	}

	cudaMemcpy(m_hPos, dPos, sizeof(float) * 4u * m_numParticles, cudaMemcpyDeviceToHost);

	// Batch the fluid transform updates (one call, no per-frame MarkRenderStateDirty).
	// Sim -> world is anchored at the actor: uniform scale S, with sim Y (up) -> UE Z
	// and sim Z (depth) -> UE Y. So the fluid is centered on the actor's location.
	const FVector ActorLoc = GetActorLocation();
	const float S = widthScaling;
	m_fluidTransforms.SetNumUninitialized(m_numFluidParticles);
	for (uint32 i = 0; i < m_numFluidParticles; ++i)
	{
		FVector Location(
			ActorLoc.X + m_hPos[i * 4]     * S,    // sim x  -> UE X
			ActorLoc.Y + m_hPos[i * 4 + 2] * S,    // sim z  -> UE Y (depth)
			ActorLoc.Z + m_hPos[i * 4 + 1] * S);   // sim y  -> UE Z (up)
		m_fluidTransforms[i] = FTransform(FRotator::ZeroRotator, Location, FVector(radiusScaling));
	}
	ParticleInstancedMeshComponent->BatchUpdateInstancesTransforms(0, m_fluidTransforms, /*bWorldSpace*/true, /*bMarkRenderStateDirty*/true, /*bTeleport*/true);
}

void ASPHFluidActor::initialize(int numParticles)
{
	check(!m_bInitialized);

	// host storage
	m_hPos = new float[numParticles * 4];
	m_hVel = new float[numParticles * 4];
	memset(m_hPos, 0, numParticles * 4 * sizeof(float));
	memset(m_hVel, 0, numParticles * 4 * sizeof(float));

	m_hDensities = new float[numParticles];
	m_hPressures = new float[numParticles];
	m_hEntireForces = new float[numParticles * 4];
	memset(m_hDensities, 0, numParticles * sizeof(float));
	memset(m_hPressures, 0, numParticles * sizeof(float));
	memset(m_hEntireForces, 0, numParticles * 4 * sizeof(float));

	m_hCellStart = new uint32[numParticles];
	memset(m_hCellStart, 0, numParticles * sizeof(uint32));
	m_hCellEnd = new uint32[numParticles];
	memset(m_hCellEnd, 0, numParticles * sizeof(uint32));

	// GPU data
	unsigned int memSize = sizeof(float) * 4 * numParticles;

	checkCudaErrors(cudaMalloc((void**)&m_cudaPosVBO, memSize));
	allocateArray((void**)&m_dVel, memSize);

	allocateArray((void**)&m_dDensities, sizeof(float) * numParticles);
	allocateArray((void**)&m_dPressures, sizeof(float) * numParticles);
	allocateArray((void**)&m_dBoundaryPsi, sizeof(float) * numParticles);
	allocateArray((void**)&m_dEntireForces, memSize);

	allocateArray((void**)&m_dSortedPos, memSize);
	allocateArray((void**)&m_dSortedVel, memSize);

	allocateArray((void**)&m_dGridParticleHash, numParticles * sizeof(uint32));
	allocateArray((void**)&m_dGridParticleIndex, numParticles * sizeof(uint32));

	allocateArray((void**)&m_dCellStart, m_numGridCells * sizeof(uint32));
	allocateArray((void**)&m_dCellEnd, m_numGridCells * sizeof(uint32));

	checkCudaErrors(cudaMalloc((void**)&m_cudaColorVBO, sizeof(float) * numParticles * 4));

	sdkCreateTimer(&m_timer);

	setParameters(&m_params);

	// Fluid instances are fixed in count. Boundary instances are created later in
	// BeginPlay, once _addBoudaryCube2 has placed the structure (so we only make
	// instances for boundary particles actually used).
	ParticleInstancedMeshComponent->PreAllocateInstancesMemory(m_numFluidParticles);
	for (uint32 Index = 0; Index < m_numFluidParticles; ++Index)
	{
		ParticleInstancedMeshComponent->AddInstance(FTransform(FRotator::ZeroRotator, FVector(0.f), FVector(radiusScaling)));
	}

	m_bInitialized = true;
}

void ASPHFluidActor::finalize()
{
	check(m_bInitialized);

	delete[] m_hPos;
	delete[] m_hVel;
	delete[] m_hDensities;
	delete[] m_hPressures;
	delete[] m_hEntireForces;
	delete[] m_hCellStart;
	delete[] m_hCellEnd;

	freeArray(m_dVel);
	freeArray(m_dDensities);
	freeArray(m_dPressures);
	freeArray(m_dBoundaryPsi);
	freeArray(m_dEntireForces);
	freeArray(m_dSortedPos);
	freeArray(m_dSortedVel);
	freeArray(m_dGridParticleHash);
	freeArray(m_dGridParticleIndex);
	freeArray(m_dCellStart);
	freeArray(m_dCellEnd);

	checkCudaErrors(cudaFree(m_cudaPosVBO));
	checkCudaErrors(cudaFree(m_cudaColorVBO));
}

void ASPHFluidActor::setArray(ParticleArray array, const float* data, int start, int count)
{
	check(m_bInitialized);

	switch (array)
	{
	default:
	case POSITION:
		copyArrayToDevice(m_cudaPosVBO, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;
	case VELOCITY:
		copyArrayToDevice(m_dVel, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;
	case FORCE:
		copyArrayToDevice(m_dEntireForces, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;
	case DENSITY:
		copyArrayToDevice(m_dDensities, data, start * sizeof(float), count * sizeof(float));
		break;
	case PRESSURE:
		copyArrayToDevice(m_dPressures, data, start * sizeof(float), count * sizeof(float));
		break;
	}
}

static inline float frand()
{
	return rand() / (float)RAND_MAX;
}

void ASPHFluidActor::initGrid(uint* size, float spacing, float jitter, uint numParticles)
{
	srand(1973);

	for (uint z = 0; z < size[2]; z++)
	{
		for (uint y = 0; y < size[1]; y++)
		{
			for (uint x = 0; x < size[0]; x++)
			{
				uint i = (z * size[1] * size[0]) + (y * size[0]) + x;

				if (i < numParticles)
				{
					m_hPos[i * 4]     = (spacing * x) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
					m_hPos[i * 4 + 1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
					m_hPos[i * 4 + 2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
					m_hPos[i * 4 + 3] = 1.0f;

					m_hVel[i * 4] = m_hVel[i * 4 + 1] = m_hVel[i * 4 + 2] = m_hVel[i * 4 + 3] = 0.0f;

					FTransform PositionVector(FVector(m_hPos[i * 4] * widthScaling, m_hPos[i * 4 + 2] * widthYScaling, m_hPos[i * 4 + 1] * widthScaling + widthScaling));
					ParticleInstancedMeshComponent->UpdateInstanceTransform(i, PositionVector, true);
				}
			}
		}
	}
}

void ASPHFluidActor::reset()
{
	float jitter = m_params.particleRadius * 0.01f;
	uint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
	uint gridSize[3];
	gridSize[0] = gridSize[1] = gridSize[2] = s;
	initGrid(gridSize, m_params.particleRadius * 2.0f, jitter, m_numFluidParticles);

	setArray(POSITION, m_hPos, 0, m_numParticles);
	setArray(VELOCITY, m_hVel, 0, m_numParticles);
}

void ASPHFluidActor::initStartGrid(uint* size, float spacing, float jitter, uint numParticles)
{
	srand(1973);

	// Center the initial fluid block on the sim origin -> it spawns at the actor's
	// location. Rendering is handled by Tick (actor-relative), so no instance update here.
	const float halfX = size[0] * spacing * 0.5f;
	const float halfY = size[1] * spacing * 0.5f;
	const float halfZ = size[2] * spacing * 0.5f;

	for (uint z = 0; z < size[2]; z++)
	{
		for (uint y = 0; y < size[1]; y++)
		{
			for (uint x = 0; x < size[0]; x++)
			{
				uint i = (z * size[1] * size[0]) + (y * size[0]) + x;

				if (i < numParticles)
				{
					m_hPos[i * 4]     = (spacing * x) - halfX;
					m_hPos[i * 4 + 1] = (spacing * y) - halfY;
					m_hPos[i * 4 + 2] = (spacing * z) - halfZ;
					m_hPos[i * 4 + 3] = 1.0f;

					m_hVel[i * 4] = m_hVel[i * 4 + 1] = m_hVel[i * 4 + 2] = m_hVel[i * 4 + 3] = 0.0f;
				}
			}
		}
	}
}

void ASPHFluidActor::resetStart()
{
	float jitter = m_params.particleRadius * 0.01f;
	// Size the grid to hold ALL fluid particles. The old `s/3` made a grid far
	// too small (e.g. 27 cells for 500 particles), leaving the rest stacked at
	// the memset origin (0,0,0) -> density singularity -> the sim exploded.
	uint s = (uint)ceilf(powf((float)m_numFluidParticles, 1.0f / 3.0f));
	uint gridSize[3];
	gridSize[0] = gridSize[1] = gridSize[2] = s;
	initStartGrid(gridSize, m_params.particleRadius * 2.0f, jitter, m_numFluidParticles);

	setArray(POSITION, m_hPos, 0, m_numParticles);
	setArray(VELOCITY, m_hVel, 0, m_numParticles);
}

void ASPHFluidActor::addBoundaryCube(int start, float* pos, float* width, float* vel, float particleDiameter, uint* addedBParticles)
{
	uint index = start + *addedBParticles;
	uint i = 0;

	for (int z = 0; z <= width[2] / particleDiameter; z++)
	{
		for (int y = 0; y <= width[1] / particleDiameter; y++)
		{
			for (int x = 0; x <= width[0] / particleDiameter; x++)
			{
				float dx = x * particleDiameter;
				float dy = y * particleDiameter;
				float dz = z * particleDiameter;

				if (index < m_numParticles)
				{
					m_hPos[index * 4]     = pos[0] + dx;
					m_hPos[index * 4 + 1] = pos[1] + dy;
					m_hPos[index * 4 + 2] = pos[2] + dz;
					m_hPos[index * 4 + 3] = pos[3];

					m_hVel[index * 4]     = vel[0];
					m_hVel[index * 4 + 1] = vel[1];
					m_hVel[index * 4 + 2] = vel[2];
					m_hVel[index * 4 + 3] = vel[3];
					index++;
					i++;
				}
			}
		}
	}

	setArray(POSITION, m_hPos + (start + *addedBParticles) * 4, start + *addedBParticles, i);
	setArray(VELOCITY, m_hVel + (start + *addedBParticles) * 4, start + *addedBParticles, i);

	*addedBParticles += i;
}

void ASPHFluidActor::addBoundaryRotateCube(int start, float* pos, float* width, float* vel, float rotate, float particleDiameter, uint* addedBParticles)
{
	uint index = start + *addedBParticles;
	uint i = 0;
	float dTheta = tan(rotate);
	float dx, dy, dz;

	for (int z = 0; z <= width[2] / particleDiameter; z++)
	{
		for (int y = 0; y <= width[1] / particleDiameter; y++)
		{
			for (int x = 0; x <= width[0] / particleDiameter; x++)
			{
				dx = x * particleDiameter;
				dy = y * particleDiameter + dx * dTheta;
				dz = z * particleDiameter;

				if (index < m_numParticles)
				{
					m_hPos[index * 4]     = pos[0] + dx;
					m_hPos[index * 4 + 1] = pos[1] + dy;
					m_hPos[index * 4 + 2] = pos[2] + dz;
					m_hPos[index * 4 + 3] = pos[3];

					m_hVel[index * 4]     = vel[0];
					m_hVel[index * 4 + 1] = vel[1];
					m_hVel[index * 4 + 2] = vel[2];
					m_hVel[index * 4 + 3] = vel[3];
					index++;
					i++;
				}
			}
		}
	}

	setArray(POSITION, m_hPos + (start + *addedBParticles) * 4, start + *addedBParticles, i);
	setArray(VELOCITY, m_hVel + (start + *addedBParticles) * 4, start + *addedBParticles, i);

	*addedBParticles += i;
}

void ASPHFluidActor::_addBoudaryCube()
{
	float pos1[4], width1[4], vel1[4];
	float space = 2.0f;
	vel1[0] = vel1[1] = vel1[2] = vel1[3] = 0.0f;

	pos1[0] = -0.7f; pos1[1] = -0.7f; pos1[2] = -0.7f; pos1[3] = 0.0f;
	width1[0] = 0.01f; width1[1] = 0.2f; width1[2] = 0.2f; width1[3] = 0.0f;
	addBoundaryCube(m_numFluidParticles, pos1, width1, vel1, m_params.particleRadius * space, &addedBoundaryParticles);

	pos1[0] = -0.7f; pos1[1] = -0.7f; pos1[2] = -0.7f;
	width1[0] = 0.2f; width1[1] = 0.2f; width1[2] = 0.02f;
	addBoundaryCube(m_numFluidParticles, pos1, width1, vel1, m_params.particleRadius * space, &addedBoundaryParticles);

	pos1[0] = -0.5f; pos1[1] = -0.7f; pos1[2] = -0.7f;
	width1[0] = 0.01f; width1[1] = 0.2f; width1[2] = 0.2f;
	addBoundaryCube(m_numFluidParticles, pos1, width1, vel1, m_params.particleRadius * space, &addedBoundaryParticles);

	pos1[0] = -0.7f; pos1[1] = -0.7f; pos1[2] = -0.5f;
	width1[0] = 0.2f; width1[1] = 0.2f; width1[2] = 0.01f;
	addBoundaryCube(m_numFluidParticles, pos1, width1, vel1, m_params.particleRadius * space, &addedBoundaryParticles);

	pos1[0] = -0.7f; pos1[1] = -0.7f; pos1[2] = -0.7f;
	width1[0] = 0.2f; width1[1] = 0.01f; width1[2] = 0.2f;
	addBoundaryCube(m_numFluidParticles, pos1, width1, vel1, m_params.particleRadius * space, &addedBoundaryParticles);
}

void ASPHFluidActor::_addBoudaryCube2()
{
	float pos[4], width[4], vel[4];
	const float d = m_params.particleRadius * 1.3f;   // boundary particle spacing (~0.013)
	vel[0] = vel[1] = vel[2] = vel[3] = 0.0f;

	// Thick angled slide under the falling fluid. Tilted so it rises toward +x
	// => fluid slides down toward -x. (The geometric collision plane in the kernel
	// is aligned to this slab's top surface.)
	pos[0] = -0.7f; pos[1] = -0.3f; pos[2] = -0.1f; pos[3] = 0.0f;
	width[0] = 1.2f; width[1] = 0.05f; width[2] = 0.6f; width[3] = 0.0f;
	addBoundaryRotateCube(m_numFluidParticles, pos, width, vel, 0.4f, d, &addedBoundaryParticles);
}

void ASPHFluidActor::_addFluidCube()
{
	float pos1[4], width1[4], vel1[4];
	float space = 2.0f;
	pos1[0] = -0.65f; pos1[1] = -0.65f; pos1[2] = -0.65f; pos1[3] = 0.0f;
	width1[0] = 0.1f; width1[1] = 0.1f; width1[2] = 0.1f; width1[3] = 0.0f;
	vel1[0] = vel1[1] = vel1[2] = vel1[3] = 0.0f;
	addFluidCube(0, pos1, width1, vel1, m_params.particleRadius * space);
}

void ASPHFluidActor::addFluidCube(int start, float* pos, float* width, float* vel, float particleDiameter)
{
	uint index = start;

	for (int z = 0; z <= width[2] / particleDiameter; z++)
	{
		for (int y = 0; y <= width[1] / particleDiameter; y++)
		{
			for (int x = 0; x <= width[0] / particleDiameter; x++)
			{
				float dx = x * particleDiameter;
				float dy = y * particleDiameter;
				float dz = z * particleDiameter;

				if (index < m_numFluidParticles)
				{
					m_hPos[index * 4]     = pos[0] + dx;
					m_hPos[index * 4 + 1] = pos[1] + dy;
					m_hPos[index * 4 + 2] = pos[2] + dz;
					m_hPos[index * 4 + 3] = pos[3];

					m_hVel[index * 4]     = vel[0];
					m_hVel[index * 4 + 1] = vel[1];
					m_hVel[index * 4 + 2] = vel[2];
					m_hVel[index * 4 + 3] = vel[3];
					index++;
					FTransform PositionVector(FVector(m_hPos[index * 4] * widthScaling, m_hPos[index * 4 + 2] * widthYScaling, m_hPos[index * 4 + 1] * widthScaling + widthScaling));
					ParticleInstancedMeshComponent->UpdateInstanceTransform(index, PositionVector, true);
				}
			}
		}
	}

	setArray(POSITION, m_hPos, 0, index);
	setArray(VELOCITY, m_hVel, 0, index);
}
