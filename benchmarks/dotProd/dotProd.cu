#include <stdlib.h>
#include <stdio.h>
#include <cutil.h>
#include <prof.h>
#include <prof.cu>
#include <cutil_inline.h>
 
// kernel
__global__ void dotProductKernel(float* v1, float* v2, float* r) {
  if (v1[threadIdx.x] > v2[threadIdx.x]) {
    r[threadIdx.x] = v2[threadIdx.x] + v1[threadIdx.x];
  } else {
    r[threadIdx.x] = v1[threadIdx.x] * v2[threadIdx.x];
  }
}
 
// Fills a vector with random float entries.
void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i) {
	data[i] = (float)i;	
 
	if (drand48() < 0.5) {
        	data[i] *= -1.0;
	}
    }
}
 
// Print the sum of the vector
void printResult(float* data, int size) {
	float acc = 0.0;
	for (int i = 0; i < size; ++i) {
		acc += data[i];
		printf("%8.2f", data[i]);
	}
	printf("\nDot prod = %lf\n", acc);
}
 
void dotProd(int argc, char** argv, const unsigned VEC_SIZE) {
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );
 
	// set seed for drand48()
	srand48(41);
 
	// allocate host memory for matrices A and B
	float* h_v1 = (float*) malloc(VEC_SIZE * sizeof(float));
	float* h_v2 = (float*) malloc(VEC_SIZE * sizeof(float));
 
	// initialize host memory
	randomInit(h_v1, VEC_SIZE);
	randomInit(h_v2, VEC_SIZE);
 
	// allocate device memory
	float* d_v1; 
	cudaMalloc((void**) &d_v1, VEC_SIZE * sizeof(float));
	float* d_v2;
	cudaMalloc((void**) &d_v2, VEC_SIZE * sizeof(float));
	float* d_r;
	cudaMalloc((void**) &d_r, VEC_SIZE * sizeof(float));
 
	// copy host memory to device
	cudaMemcpy(d_v1, h_v1, VEC_SIZE * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_v2, h_v2, VEC_SIZE * sizeof(float),
			cudaMemcpyHostToDevice);
 
	// setup execution parameters
	dim3 threads(VEC_SIZE, 1);
	dim3 grid(1, 1);
 
	// execute the kernel
GpuProfiling::prepareProfiling(  grid, threads  );
	dotProductKernel<<< grid, threads >>>(d_v1, d_v2, d_r);
GpuProfiling::addResults("dotProd");
	// check if kernel execution generated and error
	CUT_CHECK_ERROR("Kernel execution failed");
 
	// copy result from device to host
	float* h_r = (float*) malloc(VEC_SIZE * sizeof(float));
	cudaMemcpy(h_r, d_r, VEC_SIZE * sizeof(float),
			cudaMemcpyDeviceToHost);
 
	printResult(h_r, VEC_SIZE);
 
	// clean up memory
	free(h_r);
	free(h_v1);
	free(h_v2);
	cudaFree(d_r);
	cudaFree(d_v1);
	cudaFree(d_v2);
}
 
int main(int argc, char** argv) {
	if (argc != 2) {
		fprintf(stderr, "Syntax: %s vec_size\n", argv[0]);
		return EXIT_FAILURE;
	} else {
		const int VEC_SIZE = atoi(argv[1]);
GpuProfiling::initProf();
		dotProd(argc, argv, VEC_SIZE);
GpuProfiling::printResults();
	}
}
