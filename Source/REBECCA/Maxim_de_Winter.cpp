// Fill out your copyright notice in the Description page of Project Settings.


#include "Maxim_de_Winter.h"

#include <iostream>
#include <random>

// Sets default values
AMaxim_de_Winter::AMaxim_de_Winter()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	tempNum = 210.f;

	particleMeshRadius = 50.0f;
	particleRenderRadius = 10.0f;
	//particleRadius = 1.0f / 64.0f;
    widthScaling = 1500.f;
    radiusScaling = 0.7f;
	
	//Add particle instanced mesh component
	ParticleInstancedMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>("ParticleInstancedMeshComponent");
	BoundaryParticleInstancedMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>("BoundaryParticleInstancedMeshComponent");
	ParticleProceduralMeshComponent = CreateDefaultSubobject<UProceduralMeshComponent>("ParticleProceduralMeshComponent");

	ParticleProceduralMeshComponent->bUseAsyncCooking = true;

    m_bInitialized = false;
    m_numParticles = NUM_PARTICLES + NUM_BOUNDARY_PARTICLES;
    m_numFluidParticles = NUM_PARTICLES;
    m_hPos = 0;
    m_hVel = 0;
    m_dPos = 0;
    m_dVel = 0;
    m_gridSize = make_uint3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    m_timer = NULL;
    m_solverIterations = 1;

    m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;
    //    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;

    m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    m_params.colliderRadius = 0.2f;

    m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    // m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);


    m_params.spring = 0.5f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;

    m_params.gravity = make_float3(0.0f, -9.81f, 0.0f);//make_float3(0.0f, -0.0003f, 0.0f);
    m_params.globalDamping = 1.0f;

    m_params.fluidParticleMass = 0.02f;
    m_params.boundaryParticleMass = 0.03f;
    m_params.waterRestDensity = 998.29f;
    m_params.particleRadius = 0.01f;//1.0f / 64.0f;
    m_params.viscosityCoefficient = 3.5f;//0.001003f; //3.5f; //1.003 × 10−3
    m_params.gasStiffnessConstant = 3.f;  // 135409.f; // k = nRT
    m_params.restitutionCoefficient = 0.001f;
    m_params.surfaceTension = 0.0728f;
    m_params.smoothingLength = 0.0457f;//2.0f * m_params.particleRadius;//1.3 * pow((m_params.particleMass / m_params.waterRestDensity), (1 / 3)); //0.0457f;

    addedBoundaryParticles = 0;
    addedFluidPraticles = 0;

    particleRenderRadius = 10.f;
    particleMeshRadius = 50.f;

    float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);
}

// Called when the game starts or when spawned
void AMaxim_de_Winter::BeginPlay()
{
	Super::BeginPlay();

    if (m_bInitialized)
    {
        Destroy();
    }
    initialize(m_numParticles);
    reset();
	
    /*ParticleInstancedMeshComponent->PreAllocateInstancesMemory(5);
	for (uint32 Index = 0; Index < 5; ++Index)
	{
		FVector Location(0.0f, 0.0f, 0.0f);
		int32 Result = ParticleInstancedMeshComponent->AddInstance(FTransform(FRotator::ZeroRotator, Location, FVector(particleRenderRadius / particleMeshRadius)));
	}

    BoundaryParticleInstancedMeshComponent->PreAllocateInstancesMemory(5);
    for (uint32 Index = 0; Index <5; ++Index)
    {
        FVector Location(0.0f, 0.0f, 0.0f);
        int32 Result = BoundaryParticleInstancedMeshComponent->AddInstance(FTransform(FRotator::ZeroRotator, Location, FVector(particleRenderRadius / particleMeshRadius)));
    }*/

    _addBoudaryCube();
    _addFluidCube();

    ParticleInstancedMeshComponent->MarkRenderStateDirty();
    BoundaryParticleInstancedMeshComponent->MarkRenderStateDirty();
}

// Called every frame
void AMaxim_de_Winter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

    assert(m_bInitialized);

    float* dPos;

    dPos = (float*)m_cudaPosVBO;

    setParameters(&m_params);

    // calculate grid hash
    calcHash(
        m_dGridParticleHash,
        m_dGridParticleIndex,
        dPos,
        m_numParticles);

    // sort particles based on hash
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

    // reorder particle arrays into sorted order and find start and end of each cell
    cudaError_t error = reorderDataAndFindCellStart( //Errrorrrrr
        m_dCellStart,
        m_dCellEnd,
        m_dSortedPos,
        m_dGridParticleHash,
        m_dGridParticleIndex,
        dPos,
        m_numParticles,
        m_numGridCells);

    computeDensityAndPressure(
        m_dDensities,
        m_dPressures,
        m_dSortedPos,
        m_dSortedVel,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numFluidParticles,
        m_numGridCells);

    computeForceAndViscosity(
        m_dVel,
        m_dEntireForces,
        DeltaTime,
        m_dDensities,
        m_dPressures,
        m_dSortedPos,
        m_dSortedVel,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numFluidParticles,
        m_numGridCells);

    // integrate
    integrateSystem(
        dPos,
        m_dVel,
        DeltaTime,
        m_numParticles);

    cudaMemcpy(m_hPos, dPos, sizeof(float) * 4 * m_numParticles, cudaMemcpyDeviceToHost);

    for (uint32 FluidParticleIndex = 0; FluidParticleIndex < m_numFluidParticles; ++FluidParticleIndex)
    {
        FVector VectorLocation = FVector(m_hPos[FluidParticleIndex * 4] * widthScaling, m_hPos[FluidParticleIndex * 4 + 2] * widthScaling, m_hPos[FluidParticleIndex * 4 + 1] * widthScaling + widthScaling);
        //UE_LOG(LogTemp, Warning, TEXT("position: %f, %f, %f"), VectorLocation.X, VectorLocation.Y, VectorLocation.Z);

        //FTransform instancedTransform(VectorLocation);
        FTransform instancedTransform(FRotator::ZeroRotator, VectorLocation, FVector(radiusScaling));
        ParticleInstancedMeshComponent->UpdateInstanceTransform(FluidParticleIndex, instancedTransform, true);
    }

    for (uint32 BoundaryParticleIndex = m_numFluidParticles; BoundaryParticleIndex < m_numParticles; ++BoundaryParticleIndex)
    {
        FVector VectorLocation = FVector(m_hPos[BoundaryParticleIndex * 4] * widthScaling, m_hPos[BoundaryParticleIndex * 4 + 2] * widthScaling, m_hPos[BoundaryParticleIndex * 4 + 1] * widthScaling + widthScaling);

        //FTransform instancedTransform(VectorLocation);
        FTransform instancedTransform(FRotator::ZeroRotator, VectorLocation, FVector(radiusScaling));
        BoundaryParticleInstancedMeshComponent->UpdateInstanceTransform(BoundaryParticleIndex - m_numFluidParticles, instancedTransform, true);
    }
    UE_LOG(LogTemp, Warning, TEXT("DeltaTime: %f"), DeltaTime);
	ParticleInstancedMeshComponent->MarkRenderStateDirty();
    BoundaryParticleInstancedMeshComponent->MarkRenderStateDirty();
}


void AMaxim_de_Winter::initialize(int numParticles)
{
    assert(!m_bInitialized);

    // allocate host storage
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

    // allocate GPU data
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
        FVector Location(0.0f, 0.0f, 0.0f);
        int32 Result = ParticleInstancedMeshComponent->AddInstance(FTransform(FRotator::ZeroRotator, Location, FVector(particleRenderRadius / particleMeshRadius)));
    }

    BoundaryParticleInstancedMeshComponent->PreAllocateInstancesMemory(numParticles - m_numFluidParticles);
    for (uint32 Index = 0; Index < numParticles - m_numFluidParticles; ++Index)
    {
        FVector Location(0.0f, 0.0f, 0.0f);
        int32 Result = BoundaryParticleInstancedMeshComponent->AddInstance(FTransform(FRotator::ZeroRotator, Location, FVector(particleRenderRadius / particleMeshRadius)));
    }

    m_bInitialized = true;
}

void AMaxim_de_Winter::finalize()
{
    assert(m_bInitialized);

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

void AMaxim_de_Winter::setArray(ParticleArray array, const float* data, int start, int count)
{
    assert(m_bInitialized);

    switch (array)
    {
    default:
    case POSITION:
    {
        copyArrayToDevice(m_cudaPosVBO, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
        /*UE_LOG(LogTemp, Warning, TEXT("Hi!"));*/
    }
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

inline float frand()
{
    return rand() / (float)RAND_MAX;
}

void AMaxim_de_Winter::initGrid(uint* size, float spacing, float jitter, uint numParticles)
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
                    m_hPos[i * 4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 3] = 1.0f;

                    m_hVel[i * 4] = 0.0f;
                    m_hVel[i * 4 + 1] = 0.0f;
                    m_hVel[i * 4 + 2] = 0.0f;
                    m_hVel[i * 4 + 3] = 0.0f;

                    /*FVector positionVector(m_hPos[i * 4] * particleRenderRadius / particleMeshRadius,
                        m_hPos[i * 4 + 2] * particleRenderRadius / particleMeshRadius,
                        (m_hPos[i * 4 + 1] + 1.0f) * particleRenderRadius / particleMeshRadius);*/
                    FTransform positionVector(FVector(m_hPos[i * 4] * widthScaling, m_hPos[i * 4 + 2] * widthScaling, m_hPos[i * 4 + 1] * widthScaling + widthScaling));
                    ParticleInstancedMeshComponent->UpdateInstanceTransform(i, positionVector, true);
                    //UE_LOG(LogTemp, Warning, TEXT("m_hPos: %f"), m_hPos[i * 4 + 2]);
                }
            }
        }
    }
}

void AMaxim_de_Winter::reset()
{
    float jitter = m_params.particleRadius * 0.01f;
    uint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
    uint gridSize[3];
    gridSize[0] = gridSize[1] = gridSize[2] = s;
    initGrid(gridSize, m_params.particleRadius * 2.0f, jitter, m_numFluidParticles);

    setArray(POSITION, m_hPos, 0, m_numParticles);
    setArray(VELOCITY, m_hVel, 0, m_numParticles);
}

void AMaxim_de_Winter::_addBoudaryCube()
{
    float pos1[4], width1[4], vel1[4];
    float space = 1.5f;
    //1
    pos1[0] = -0.7f;
    pos1[1] = -0.7f;
    pos1[2] = -0.7f;
    pos1[3] = 0.0f;
    width1[0] = 0.01f;
    width1[1] = 0.2f;
    width1[2] = 0.2f;
    width1[3] = 0.0f;
    vel1[0] = 0.0f;
    vel1[1] = 0.0f;
    vel1[2] = 0.0f;
    vel1[3] = 0.0f;
    addBoundaryCube(m_numFluidParticles, pos1, width1, vel1, m_params.particleRadius * space, &addedBoundaryParticles);
    //2
    pos1[0] = -0.7f;
    pos1[1] = -0.7f;
    pos1[2] = -0.7f;
    width1[0] = 0.2f;
    width1[1] = 0.2f;
    width1[2] = 0.01f;
    addBoundaryCube(m_numFluidParticles, pos1, width1, vel1, m_params.particleRadius * space, &addedBoundaryParticles);
    //3
    pos1[0] = -0.5f;
    pos1[1] = -0.7f;
    pos1[2] = -0.7f;
    width1[0] = 0.01f;
    width1[1] = 0.2f;
    width1[2] = 0.2f;
    addBoundaryCube(m_numFluidParticles, pos1, width1, vel1, m_params.particleRadius * space, &addedBoundaryParticles);
    //4
    pos1[0] = -0.7f;
    pos1[1] = -0.7f;
    pos1[2] = -0.5f;
    width1[0] = 0.2f;
    width1[1] = 0.2f;
    width1[2] = 0.01f;
    addBoundaryCube(m_numFluidParticles, pos1, width1, vel1, m_params.particleRadius * space, &addedBoundaryParticles);
    //5
    pos1[0] = -0.7f;
    pos1[1] = -0.7f;
    pos1[2] = -0.7f;
    width1[0] = 0.2f;
    width1[1] = 0.01f;
    width1[2] = 0.2f;
    addBoundaryCube(m_numFluidParticles, pos1, width1, vel1, m_params.particleRadius * space, &addedBoundaryParticles);
}

void AMaxim_de_Winter::addBoundaryCube(int start, float* pos, float* width, float* vel, float particleDiameter, uint* addedBParticles)
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
                //float jitter = m_params.particleRadius * 0.01f;

                if (index < m_numParticles)
                {
                    m_hPos[index * 4] = pos[0] + dx;
                    m_hPos[index * 4 + 1] = pos[1] + dy;
                    m_hPos[index * 4 + 2] = pos[2] + dz;
                    m_hPos[index * 4 + 3] = pos[3];

                    m_hVel[index * 4] = vel[0];
                    m_hVel[index * 4 + 1] = vel[1];
                    m_hVel[index * 4 + 2] = vel[2];
                    m_hVel[index * 4 + 3] = vel[3];
                    index++;
                    FTransform positionVector(FVector(m_hPos[index * 4] * widthScaling, m_hPos[index * 4 + 2] * widthScaling, m_hPos[index * 4 + 1] * widthScaling + widthScaling));
                    BoundaryParticleInstancedMeshComponent->UpdateInstanceTransform(index - m_numFluidParticles, positionVector, true);
                    //index++;
                    i++;
                }
            }
        }
    }

    setArray(POSITION, m_hPos + (start + *addedBParticles) * 4, start + *addedBParticles, i);
    setArray(VELOCITY, m_hVel + (start + *addedBParticles) * 4, start + *addedBParticles, i);

    *addedBParticles += i;
}

void AMaxim_de_Winter::_addFluidCube()
{
    float pos1[4], width1[4], vel1[4];
    float space = 1.5f;
    pos1[0] = -0.65f;
    pos1[1] = -0.65f;
    pos1[2] = -0.65f;
    pos1[3] = 0.0f;
    width1[0] = 0.1f;
    width1[1] = 0.1f;
    width1[2] = 0.1f;
    width1[3] = 0.0f;
    vel1[0] = 0.0f;
    vel1[1] = 0.0f;
    vel1[2] = 0.0f;
    vel1[3] = 0.0f;
    addFluidCube(0, pos1, width1, vel1, m_params.particleRadius * space);
}

void AMaxim_de_Winter::addFluidCube(int start, float* pos, float* width, float* vel, float particleDiameter)
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
                //float jitter = m_params.particleRadius * 0.01f;

                if (index < m_numFluidParticles)
                {
                    m_hPos[index * 4] = pos[0] + dx;
                    m_hPos[index * 4 + 1] = pos[1] + dy;
                    m_hPos[index * 4 + 2] = pos[2] + dz;
                    m_hPos[index * 4 + 3] = pos[3];

                    m_hVel[index * 4] = vel[0];
                    m_hVel[index * 4 + 1] = vel[1];
                    m_hVel[index * 4 + 2] = vel[2];
                    m_hVel[index * 4 + 3] = vel[3];
                    index++;
                    FTransform positionVector(FVector(m_hPos[index * 4] * widthScaling, m_hPos[index * 4 + 2] * widthScaling, m_hPos[index * 4 + 1] * widthScaling + widthScaling));
                    ParticleInstancedMeshComponent->UpdateInstanceTransform(index, positionVector, true);
                    //index++;
                }
            }
        }
    }

    setArray(POSITION, m_hPos, 0, index);
    setArray(VELOCITY, m_hPos, 0, index);
}