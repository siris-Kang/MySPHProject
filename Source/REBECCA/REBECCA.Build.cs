// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class REBECCA : ModuleRules
{
	private string project_root_path
	{
		get { return Path.Combine(ModuleDirectory, "../.."); }
	}
	public REBECCA(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "HeadMountedDisplay" });

		string custom_cuda_lib_include = "CudaLibrary";
		string custom_cuda_lib_lib = "x64/Release";

		PublicIncludePaths.Add(Path.Combine(project_root_path, custom_cuda_lib_include));
		PublicAdditionalLibraries.Add(Path.Combine(project_root_path, custom_cuda_lib_lib, "CudaLibrary.lib"));

		string cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5";
		string cuda_include = "include";
		string cuda_lib = "lib/x64";

		PublicIncludePaths.Add(Path.Combine(cuda_path, cuda_include));

		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "cudart_static.lib"));
	}
}
