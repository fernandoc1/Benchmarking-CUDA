#ifndef __PROF_H__
#define __PROF_H__


#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <cstdlib>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cutil.h"


#define CUDA_WARP_SIZE 32

#ifdef PROF_MEMCPY
#define myCudaMemcpy GpuProfiling::profCudaMemcpy
#else
#define myCudaMemcpy cudaMemcpy
#endif


class GpuProfiling {
public:

	typedef enum {profIdleTime, profMemcpy, profBranches, profExecTimeCpu, profOther} profType_t;
	typedef std::vector<unsigned long long> ResultSet;
	typedef std::map<std::string, ResultSet> ProfData;
	
	
private:	
	static ProfData profInfo;
	static profType_t profType;
	static double memcpyTime;
	static long resultSize;
	static unsigned long long* profilingInfoGpu;
	static struct timeval kernelStartTime;

public:	 
	GpuProfiling() {}
	~GpuProfiling() {}

	static int envVarActive(std::string envVarName) {
		char* envVarValue = getenv(envVarName.c_str());
		
		if (envVarValue != NULL) {
			return atoi(envVarValue);
		}
		
		return 0;
	}
	
	static void initProf() {		
		profType = profOther;
		if (envVarActive("PROF_IDLE_TIME")) {
			profType = profIdleTime;
		}
		if (envVarActive("PROF_BRANCHES")) {
			profType = profBranches;
		}
		if (envVarActive("PROF_KERNEL_EXEC_TIME_CPU")) {
			profType = profExecTimeCpu;
		}
#ifdef PROF_MEMCPY  
		profType = profMemcpy;
#endif

		memcpyTime = 0.0;
		resultSize = 0;
		profilingInfoGpu = NULL;
	}
	
	static cudaError_t profCudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
		struct timeval start, finish;
		
		gettimeofday(&start, NULL);
		cudaError_t ret = cudaMemcpy(dst, src, count, kind);
		gettimeofday(&finish, NULL);
		
		memcpyTime += finish.tv_sec - start.tv_sec;
		memcpyTime += 0.000001 * (finish.tv_usec - start.tv_usec);
		return ret;
	}
	
	static int prepareProfiling(dim3 grid, dim3 threads, unsigned int sharedMemorySize=0) {
		unsigned int totalThreads = grid.x*threads.x*grid.y*threads.y;
		int status = -1;

		if (profType == profBranches) {
			void* gpuProfDataSizeAddr = NULL;
			unsigned int instrumentedBasicBlocks = 0;

			status = cudaGetSymbolAddress(&gpuProfDataSizeAddr, "prof_data_sz");
			if (status != cudaSuccess) {
				std::cerr << "CUDA Profiling: Could not get address of prof_data_sz variable.\n";
				std::cerr << "Verify if you are running the instrumented kernel.\n\n";
				exit(1);
			}

			CUDA_SAFE_CALL( cudaMemcpy(&instrumentedBasicBlocks, gpuProfDataSizeAddr, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
			resultSize = instrumentedBasicBlocks*3;
		} else {
			resultSize = totalThreads;
		}

		CUDA_SAFE_CALL( cudaMalloc( (void**)(&profilingInfoGpu), resultSize*sizeof(unsigned long long)) );
		CUDA_SAFE_CALL( cudaMemset(profilingInfoGpu, 0, resultSize*sizeof(unsigned long long)) );

		// write this to GPU
		void* gpuProfDataPtAddr = NULL;
		status = cudaGetSymbolAddress(&gpuProfDataPtAddr, "prof_data_pt");
		if (status != cudaSuccess) {
			std::cerr << "CUDA Profiling: Could not get address of prof_data_pt variable.\n";
			std::cerr << "Verify if you are running the instrumented kernel.\n\n";
			exit(1);
		}

		CUDA_SAFE_CALL( cudaMemcpy(gpuProfDataPtAddr, &profilingInfoGpu, sizeof(unsigned long long *), cudaMemcpyHostToDevice) );

		if (profType == profExecTimeCpu) {
			gettimeofday(&kernelStartTime, NULL);
		} 
		return 0;
	}
	
	static int addResults(std::string kernelName) {
		ResultSet& resultSet = profInfo[kernelName];

		// wait kernel execution finish
		cudaError_t err = cudaThreadSynchronize();
		if(err != cudaSuccess) {
			fprintf(stderr, "Cuda kernel %s execution error: %s.\n", kernelName.c_str(), cudaGetErrorString( (cudaError_t)err) );
			exit(EXIT_FAILURE);
		}

		if (profType == profExecTimeCpu) {
			struct timeval kernelEndTime;
			gettimeofday(&kernelEndTime, NULL);

			CUDA_SAFE_CALL( cudaFree(profilingInfoGpu); );

			if (resultSet.size() < 1) {
				resultSet.push_back(0);
			}

			unsigned long long kernelExecTime = (kernelEndTime.tv_sec - kernelStartTime.tv_sec) * 1000000;
			kernelExecTime += kernelEndTime.tv_usec - kernelStartTime.tv_usec;
			resultSet[0]   += kernelExecTime;
		} else {
			unsigned int realResultSize = resultSize;

			unsigned long long* resultsCpu = (unsigned long long*) malloc(resultSize*sizeof(unsigned long long));
			CUDA_SAFE_CALL( cudaMemcpy(resultsCpu, profilingInfoGpu, resultSize*sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
			CUDA_SAFE_CALL( cudaFree(profilingInfoGpu); );

			// add results in resultsCpu to profInfo
			unsigned int i;
			unsigned int resultSetSize = resultSet.size();
			resultSet.reserve(realResultSize);

			if (profType == profBranches) {
				for (i=0; (i<realResultSize) && (i<resultSetSize); i++) {
					if ((i % 3) != 0) {
						resultSet[i] += resultsCpu[i];
					}
				}
			} else {
				for (i=0; (i<realResultSize) && (i<resultSetSize); i++) {
					resultSet[i] += resultsCpu[i];
				}
			}
			for (; i<realResultSize; i++) {
				resultSet.push_back(resultsCpu[i]);
			}

			free(resultsCpu);
		}

		return 0;
	}
	
	static int printResults() {
		if (profType == profMemcpy) {
			printf("ProfMemcopyTime: %G\n", memcpyTime);
			return 0;
		}
	
		ProfData::const_iterator it = profInfo.begin();
		for (; it != profInfo.end(); it++) {
			const std::string kernelName = (*it).first;
			const ResultSet& resultSet   = (*it).second;
			
			std::string filename = "prof." + kernelName + ".txt";
			std::ofstream results(filename.c_str(), std::ios::out | std::ios::trunc);
		
			if (profType == profBranches) {
				for (unsigned int i=0; i<resultSet.size(); i+=3) {
					// split output in basic blocks
					results << resultSet[i] << " " << resultSet[i+1] << " " << resultSet[i+2] << std::endl;
				}
			} else {	
				for (unsigned int i=0; i<resultSet.size(); i++) {
					// split output in warps
					if ((i % CUDA_WARP_SIZE) == 0)
						results << std::endl;

					results << " " << resultSet[i];
				}
				results << std::endl;
			}
		
			results.close();
		}
		
		return 0;
	}

};

#endif
