#include "SPHColliderComponent.h"

#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"

USPHColliderComponent::USPHColliderComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

bool USPHColliderComponent::GetWorldBox(FVector& OutCenter, FQuat& OutRotation, FVector& OutHalfExtent) const
{
	const AActor* Owner = GetOwner();
	if (!Owner)
	{
		return false;
	}

	const UStaticMeshComponent* Mesh = Owner->FindComponentByClass<UStaticMeshComponent>();
	if (Mesh)
	{
		OutCenter = Mesh->GetComponentLocation();
		OutRotation = Mesh->GetComponentQuat();

		// Local (unscaled) half-extent of the mesh * component scale.
		FVector LocalExtent(50.f);   // fallback: UE basic cube is 100 units
		if (Mesh->GetStaticMesh())
		{
			LocalExtent = Mesh->GetStaticMesh()->GetBoundingBox().GetExtent();
		}
		OutHalfExtent = LocalExtent * Mesh->GetComponentScale();
		return true;
	}

	// No mesh: fall back to the actor transform with a default cube size.
	OutCenter = Owner->GetActorLocation();
	OutRotation = Owner->GetActorQuat();
	OutHalfExtent = FVector(50.f) * Owner->GetActorScale3D();
	return true;
}
