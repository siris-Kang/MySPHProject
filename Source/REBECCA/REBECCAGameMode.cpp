// Copyright Epic Games, Inc. All Rights Reserved.

#include "REBECCAGameMode.h"
#include "REBECCAHUD.h"
#include "REBECCACharacter.h"
#include "UObject/ConstructorHelpers.h"

AREBECCAGameMode::AREBECCAGameMode()
	: Super()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnClassFinder(TEXT("/Game/FirstPersonCPP/Blueprints/FirstPersonCharacter"));
	DefaultPawnClass = PlayerPawnClassFinder.Class;

	// use our custom HUD class
	HUDClass = AREBECCAHUD::StaticClass();
}
