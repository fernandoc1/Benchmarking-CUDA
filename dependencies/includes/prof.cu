#ifndef _PROF_CU_
#define _PROF_CU_

#include <prof.h>

GpuProfiling::ProfData   GpuProfiling::profInfo;
GpuProfiling::profType_t GpuProfiling::profType   = profOther;
double                   GpuProfiling::memcpyTime = 0.0;
long                     GpuProfiling::resultSize = 0;
unsigned long long*      GpuProfiling::profilingInfoGpu = NULL;
struct timeval 		 GpuProfiling::kernelStartTime = {0,0};

__device__ long long* prof_data_pt = NULL;
__device__ unsigned int prof_data_sz = 0;

#endif

