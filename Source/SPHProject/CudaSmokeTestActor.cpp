#include "CudaSmokeTestActor.h"
#include "CudaLibrary.h"
#include "Engine/Engine.h"

ACudaSmokeTestActor::ACudaSmokeTestActor()
{
	PrimaryActorTick.bCanEverTick = false;
}

void ACudaSmokeTestActor::BeginPlay()
{
	Super::BeginPlay();

	// 1) Report which GPU CUDA sees.
	char deviceName[256] = {0};
	if (CudaGetDeviceName(deviceName, sizeof(deviceName)) == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("[CUDA] Device: %hs"), deviceName);
		if (GEngine)
		{
			GEngine->AddOnScreenDebugMessage(-1, 12.f, FColor::Cyan,
				FString::Printf(TEXT("CUDA device: %hs"), deviceName));
		}
	}

	// 2) Add two arrays on the GPU and verify the result.
	const int Count = 5;
	const int A[Count] = { 1, 2, 3, 4, 5 };
	const int B[Count] = { 10, 20, 30, 40, 50 };
	int C[Count] = { 0 };

	const int Result = CudaAddArrays(A, B, C, Count);

	const bool bCorrect = (Result == 0) &&
		C[0] == 11 && C[1] == 22 && C[2] == 33 && C[3] == 44 && C[4] == 55;

	if (bCorrect)
	{
		UE_LOG(LogTemp, Warning, TEXT("[CUDA] Smoke test SUCCESS: 1+10=%d ... 5+50=%d"), C[0], C[4]);
		if (GEngine)
		{
			GEngine->AddOnScreenDebugMessage(-1, 12.f, FColor::Green,
				FString::Printf(TEXT("CUDA OK: 1+10=%d, 5+50=%d"), C[0], C[4]));
		}
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("[CUDA] Smoke test FAILED (rc=%d, C[0]=%d, C[4]=%d)"), Result, C[0], C[4]);
		if (GEngine)
		{
			GEngine->AddOnScreenDebugMessage(-1, 12.f, FColor::Red,
				FString::Printf(TEXT("CUDA FAILED rc=%d"), Result));
		}
	}
}
