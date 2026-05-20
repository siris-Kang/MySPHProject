// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System;
using System.IO;

public class SPHProject : ModuleRules
{
	// .../SPHProject (project root): two levels up from this file in Source/SPHProject
	private string ProjectRoot => Path.Combine(ModuleDirectory, "..", "..");

	public SPHProject(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		// CudaLibrary.lib (thrust/STL) is built with C++ exceptions (/EHsc). UE disables
		// exceptions by default, which drops STL exception helpers like std::_Xlength_error
		// from the link -> LNK2019. Enabling exceptions for this module pulls them back in.
		bEnableExceptions = true;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "EnhancedInput" });

		PrivateDependencyModuleNames.AddRange(new string[] {  });

		// ---------------------------------------------------------------
		// CUDA integration
		// The GPU code lives in /CudaLibrary and is built SEPARATELY into a
		// static lib by Visual Studio 2022 (CudaLibrary.vcxproj). Here we only
		// add its header path and link the resulting .lib + the CUDA runtime.
		// ---------------------------------------------------------------

		// 1) Our CUDA static library
		string CudaLibInclude = Path.Combine(ProjectRoot, "CudaLibrary");
		PublicIncludePaths.Add(CudaLibInclude);
		PublicAdditionalLibraries.Add(Path.Combine(ProjectRoot, "x64", "Release", "CudaLibrary.lib"));

		// 2) CUDA Toolkit. Prefer the CUDA_PATH env var (set by the installer to
		//    the active version's folder), fall back to v13.2 if it is missing.
		string CudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
		if (string.IsNullOrEmpty(CudaPath))
		{
			CudaPath = @"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2";
		}
		PublicIncludePaths.Add(Path.Combine(CudaPath, "include"));
		PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "lib", "x64", "cudart_static.lib"));
	}
}
