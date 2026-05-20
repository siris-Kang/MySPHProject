#include "SPHFluidActor.h"

#include "Components/InstancedStaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "UObject/ConstructorHelpers.h"

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

	m_bInitialized = false;
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

	float cellSize = m_params.particleRadius * 2.0f;  // cell size == particle diameter
	m_params.cellSize = make_float3(cellSize, cellSize, cellSize);
}

void ASPHFluidActor::BeginPlay()
{
	Super::BeginPlay();

	initialize(m_numParticles);
	resetStart();
	_addBoudaryCube2();

	ParticleInstancedMeshComponent->MarkRenderStateDirty();
	BoundaryParticleInstancedMeshComponent->MarkRenderStateDirty();
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

	float* dPos = (float*)m_cudaPosVBO;

	setParameters(&m_params);

	calcHash(m_dGridParticleHash, m_dGridParticleIndex, dPos, m_numParticles);

	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

	reorderDataAndFindCellStart(
		m_dCellStart, m_dCellEnd, m_dSortedPos,
		m_dGridParticleHash, m_dGridParticleIndex, dPos,
		m_numParticles, m_numGridCells);

	computeDensityAndPressure(
		m_dDensities, m_dPressures, m_dSortedPos, m_dSortedVel,
		m_dGridParticleIndex, m_dCellStart, m_dCellEnd,
		m_numParticles, m_numFluidParticles, m_numGridCells);

	computeForceAndViscosity(
		m_dVel, m_dEntireForces, timeStep, m_dDensities, m_dPressures,
		m_dSortedPos, m_dSortedVel, m_dGridParticleIndex, m_dCellStart, m_dCellEnd,
		m_numParticles, m_numFluidParticles, m_numGridCells);

	integrateSystem(dPos, m_dVel, timeStep, m_numParticles);

	cudaMemcpy(m_hPos, dPos, sizeof(float) * 4u * m_numParticles, cudaMemcpyDeviceToHost);

	for (uint32 i = 0; i < m_numFluidParticles; ++i)
	{
		FVector Location(
			m_hPos[i * 4] * widthScaling,
			m_hPos[i * 4 + 2] * widthYScaling,
			m_hPos[i * 4 + 1] * widthScaling + widthScaling);
		FTransform InstancedTransform(FRotator::ZeroRotator, Location, FVector(radiusScaling));
		ParticleInstancedMeshComponent->UpdateInstanceTransform(i, InstancedTransform, true);
	}

	for (uint32 i = m_numFluidParticles; i < m_numParticles; ++i)
	{
		FVector Location(
			m_hPos[i * 4] * widthScaling,
			m_hPos[i * 4 + 2] * widthYScaling,
			m_hPos[i * 4 + 1] * widthScaling + widthScaling);
		FTransform InstancedTransform(FRotator::ZeroRotator, Location, FVector(radiusScaling));
		BoundaryParticleInstancedMeshComponent->UpdateInstanceTransform(i - m_numFluidParticles, InstancedTransform, true);
	}

	ParticleInstancedMeshComponent->MarkRenderStateDirty();
	BoundaryParticleInstancedMeshComponent->MarkRenderStateDirty();
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

	ParticleInstancedMeshComponent->PreAllocateInstancesMemory(m_numFluidParticles);
	for (uint32 Index = 0; Index < m_numFluidParticles; ++Index)
	{
		ParticleInstancedMeshComponent->AddInstance(FTransform(FRotator::ZeroRotator, FVector(0.f), FVector(radiusScaling)));
	}

	BoundaryParticleInstancedMeshComponent->PreAllocateInstancesMemory(numParticles - m_numFluidParticles);
	for (uint32 Index = 0; Index < numParticles - m_numFluidParticles; ++Index)
	{
		BoundaryParticleInstancedMeshComponent->AddInstance(FTransform(FRotator::ZeroRotator, FVector(0.f), FVector(radiusScaling)));
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

	for (uint z = 0; z < size[2]; z++)
	{
		for (uint y = 0; y < size[1]; y++)
		{
			for (uint x = 0; x < size[0]; x++)
			{
				uint i = (z * size[1] * size[0]) + (y * size[0]) + x;

				if (i < numParticles)
				{
					m_hPos[i * 4]     = (spacing * x) + m_params.particleRadius - 0.2f;
					m_hPos[i * 4 + 1] = (spacing * y) + m_params.particleRadius + 0.5f;
					m_hPos[i * 4 + 2] = (spacing * z) + m_params.particleRadius;
					m_hPos[i * 4 + 3] = 1.0f;

					m_hVel[i * 4] = m_hVel[i * 4 + 1] = m_hVel[i * 4 + 2] = m_hVel[i * 4 + 3] = 0.0f;

					FTransform PositionVector(FVector(m_hPos[i * 4] * widthScaling, m_hPos[i * 4 + 2] * widthYScaling, m_hPos[i * 4 + 1] * widthScaling + widthScaling));
					ParticleInstancedMeshComponent->UpdateInstanceTransform(i, PositionVector, true);
				}
			}
		}
	}
}

void ASPHFluidActor::resetStart()
{
	float jitter = m_params.particleRadius * 0.01f;
	uint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
	uint gridSize[3];
	gridSize[0] = gridSize[1] = gridSize[2] = s / 3;
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
					FTransform PositionVector(FVector(m_hPos[index * 4] * widthScaling, m_hPos[index * 4 + 2] * widthYScaling, m_hPos[index * 4 + 1] * widthScaling + widthScaling));
					BoundaryParticleInstancedMeshComponent->UpdateInstanceTransform(index - m_numFluidParticles, PositionVector, true);
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
					FTransform PositionVector(FVector(m_hPos[index * 4] * widthScaling, m_hPos[index * 4 + 2] * widthYScaling, m_hPos[index * 4 + 1] * widthScaling + widthScaling));
					BoundaryParticleInstancedMeshComponent->UpdateInstanceTransform(index - m_numFluidParticles, PositionVector, true);
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
	float pos1[4], width1[4], vel1[4];
	float space = 1.3f;
	vel1[0] = vel1[1] = vel1[2] = vel1[3] = 0.0f;

	pos1[0] = -0.2f; pos1[1] = 0.28f; pos1[2] = -1.1f; pos1[3] = 0.0f;
	width1[0] = 0.5f; width1[1] = 0.01f; width1[2] = 2.0f; width1[3] = 0.0f;
	addBoundaryRotateCube(m_numFluidParticles, pos1, width1, vel1, 0.3f, m_params.particleRadius * space, &addedBoundaryParticles);

	pos1[0] = -0.7f; pos1[1] = 0.1f; pos1[2] = -1.1f;
	width1[0] = 1.0f; width1[1] = 0.03f; width1[2] = 2.0f;
	addBoundaryRotateCube(m_numFluidParticles, pos1, width1, vel1, -0.4f, m_params.particleRadius * space, &addedBoundaryParticles);

	pos1[0] = -0.8f; pos1[1] = -0.9f; pos1[2] = -1.1f;
	width1[0] = 1.8f; width1[1] = 0.01f; width1[2] = 2.0f;
	addBoundaryRotateCube(m_numFluidParticles, pos1, width1, vel1, 0.2f, m_params.particleRadius * space, &addedBoundaryParticles);

	pos1[0] = 0.3f; pos1[1] = 0.5f; pos1[2] = -1.1f;
	width1[0] = 0.01f; width1[1] = 0.3f; width1[2] = 2.0f;
	addBoundaryCube(m_numFluidParticles, pos1, width1, vel1, m_params.particleRadius * space, &addedBoundaryParticles);

	pos1[0] = -0.7f; pos1[1] = 0.1f; pos1[2] = -1.1f;
	width1[0] = 0.01f; width1[1] = 0.4f; width1[2] = 2.0f;
	addBoundaryCube(m_numFluidParticles, pos1, width1, vel1, m_params.particleRadius * space, &addedBoundaryParticles);

	pos1[0] = 1.0f; pos1[1] = -0.5f; pos1[2] = -1.1f;
	width1[0] = 0.01f; width1[1] = 0.4f; width1[2] = 2.0f;
	addBoundaryCube(m_numFluidParticles, pos1, width1, vel1, m_params.particleRadius * space, &addedBoundaryParticles);
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
