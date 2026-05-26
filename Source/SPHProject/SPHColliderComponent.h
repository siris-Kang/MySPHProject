// Marker component: add it to any Cube actor in the level and the SPH fluid will
// collide against that cube's box (position / rotation / scale). Move or rotate the
// cube in the editor to design the structure — no code changes needed.
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "SPHColliderComponent.generated.h"

UCLASS(ClassGroup = (SPH), meta = (BlueprintSpawnableComponent))
class SPHPROJECT_API USPHColliderComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USPHColliderComponent();

	// Oriented box of the owning actor, in UE world space.
	// center = owner location, rotation = owner rotation, halfExtent = mesh local
	// extent * scale (defaults to a 100-unit cube => 50 * scale if no mesh found).
	bool GetWorldBox(FVector& OutCenter, FQuat& OutRotation, FVector& OutHalfExtent) const;
};
