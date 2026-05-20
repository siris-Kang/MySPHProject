// Drop one of these into a level and press Play. It calls the CUDA library
// once on BeginPlay and reports whether the GPU round-trip worked.
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "CudaSmokeTestActor.generated.h"

UCLASS()
class SPHPROJECT_API ACudaSmokeTestActor : public AActor
{
	GENERATED_BODY()

public:
	ACudaSmokeTestActor();

protected:
	virtual void BeginPlay() override;
};
