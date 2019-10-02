#include <dlfcn.h>
#include <cstdint>
#include <unordered_map>
#include <map>
#include <string>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <atomic>

#include "globals.hpp"
#include "dlsym_hook.hpp"

typedef int (*cuda_mem_alloc_v2_fp)(uintptr_t*, size_t);
typedef int (*cuda_mem_alloc_managed_fp)(uintptr_t*, size_t, unsigned int);
typedef int (*cuda_mem_alloc_pitch_v2_fp)(uintptr_t*, size_t*, size_t, size_t, unsigned int);
typedef int (*cuda_mem_free_v2_fp)(uintptr_t);
typedef int (*cuda_mem_alloc_host_v2_fp)(void **, size_t);
typedef int (*cuda_mem_free_host_fp)(void *);
typedef int (*cuda_mem_host_alloc_fp)(void **, size_t, unsigned int);
typedef int (*cuda_launch_kernel_fp) (void *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void *, void**, void**); 

std::atomic<std::uint64_t> launched_kernels;

static destructor void exit_handler() {
}

extern "C" {

int cuMemAlloc_v2(uintptr_t *devPtr, size_t size) {
  int ret;

  static cuda_mem_alloc_v2_fp orig_cuda_mem_alloc_v2 = NULL;
  if (orig_cuda_mem_alloc_v2 == NULL)
    orig_cuda_mem_alloc_v2 = reinterpret_cast<cuda_mem_alloc_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemAlloc_v2"));

  ret = orig_cuda_mem_alloc_v2(devPtr, size);

  return ret;
}

int cuMemAlloc(uintptr_t *devPtr, size_t size) {
  int ret;

  static cuda_mem_alloc_v2_fp orig_cuda_mem_alloc = NULL;
  if (orig_cuda_mem_alloc == NULL)
    orig_cuda_mem_alloc = reinterpret_cast<cuda_mem_alloc_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemAlloc"));

  ret = orig_cuda_mem_alloc(devPtr, size);

  return ret;
}

} /* extern "C" */

std::map<std::string, void *> fps = {
  {"cuMemAlloc_v2", reinterpret_cast<void *>(cuMemAlloc_v2)},
  {"cuMemAlloc", reinterpret_cast<void *>(cuMemAlloc)}
};
