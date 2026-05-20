// CUDA static-library public interface.
// Only C-linkage functions are exposed so the Unreal module can call them
// without ever touching the CUDA compiler (nvcc).
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Smoke test: c[i] = a[i] + b[i] computed on the GPU.
// Returns 0 on success, negative error code otherwise.
//   -1: no CUDA device / cudaSetDevice failed
//   -2: kernel launch failed
//   -3: a cudaMalloc / cudaMemcpy failed
int CudaAddArrays(const int* a, const int* b, int* c, int count);

// Fills outName with the active GPU's name (for logging). Returns 0 on success.
int CudaGetDeviceName(char* outName, int bufferSize);

#ifdef __cplusplus
}
#endif
