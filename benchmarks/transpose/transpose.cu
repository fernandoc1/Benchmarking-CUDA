/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

/* Matrix transpose with Cuda
 * Host code.

 * This example transposes arbitrary-size matrices.  It compares a naive
 * transpose kernel that suffers from non-coalesced writes, to an optimized
 * transpose with fully coalesced memory access and no bank conflicts.  On
 * a G80 GPU, the optimized transpose can be more than 10x faster for large
 * matrices.
 */

#include <prof.cu>
// Utility and system includes
#include <shrUtils.h>
#include <cutil_inline.h>

// includes, kernels
#include <transpose_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);
extern "C" void computeGold( float* reference, float* idata,
                         const unsigned int size_x, const unsigned int size_y );

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv)
{
	GpuProfiling::initProf();
      // Start logs
    shrSetLogFileName ("transpose.txt");
    shrLog("%s Starting...\n\n", argv[0]);
    runTest( argc, argv);
exit(0);
    shrEXIT(argc, (const char**)argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv)
{
    // size of the matrix
#ifdef __DEVICE_EMULATION__
    const unsigned int size_x = 32;
    const unsigned int size_y = 128;
#else
    const unsigned int size_x = 256;
    const unsigned int size_y = 4096;
#endif
    // size of memory required to store the matrix
    const unsigned int mem_size = sizeof(float) * size_x * size_y;

    unsigned int timer;
    cutCreateTimer(&timer);

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

    // allocate host memory
    float* h_idata = (float*) malloc(mem_size);
    // initalize the memory
    srand(15235911);
    for( unsigned int i = 0; i < (size_x * size_y); ++i)
    {
        h_idata[i] = (float) i;    // rand();
    }

    // allocate device memory
    float* d_idata;
    float* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));

    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );

    // setup execution parameters
    dim3 grid(size_x / BLOCK_DIM, size_y / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    // warmup so we don't time CUDA startup
	GpuProfiling::prepareProfiling(  grid, threads  );
    transpose_naive<<< grid, threads >>>(d_odata, d_idata, size_x, size_y);
	GpuProfiling::addResults("transpose_naive");
	GpuProfiling::prepareProfiling(  grid, threads  );
    transpose<<< grid, threads >>>(d_odata, d_idata, size_x, size_y);
	GpuProfiling::addResults("transpose");

	// synchronize here, so we make sure that we don't count any time from the asynchronize kernel launches.
	cudaThreadSynchronize();

    // execute the naive kernel numIterations times
    int numIterations = 100;
    shrLog("Transposing a %d by %d matrix of floats...\n", size_x, size_y);
    for (int i = -1; i < numIterations; ++i)
    {
        if (i == 0)
        {
            cudaThreadSynchronize();
            cutStartTimer(timer);
        }
	GpuProfiling::prepareProfiling(  grid, threads  );
        transpose_naive<<< grid, threads >>>(d_odata, d_idata, size_x, size_y);
	GpuProfiling::addResults("transpose_naive");
    }
    cudaThreadSynchronize();
    cutStopTimer(timer);
    float naiveTime = 1.0e-3 * cutGetTimerValue(timer)/(double)numIterations;

    // execute the optimized kernel numIterations times
    for (int i = 0; i < numIterations; ++i)
    {
        if (i == 0)
        {
            cudaThreadSynchronize();
            cutResetTimer(timer);
            cutStartTimer(timer);
        }
	GpuProfiling::prepareProfiling(  grid, threads  );
        transpose<<< grid, threads >>>(d_odata, d_idata, size_x, size_y);
	GpuProfiling::addResults("transpose");
    }
    cudaThreadSynchronize();
    cutStopTimer(timer);
    float optimizedTime = 1.0e-3*cutGetTimerValue(timer) / (double)numIterations;

    shrLog("Naive transpose average time:     %0.3f ms\n", naiveTime / numIterations);
    shrLog("Optimized transpose average time: %0.3f ms\n\n", optimizedTime / numIterations);
    shrLogEx(LOGBOTH | MASTER, 0, "transpose-naive, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n",
               1.0e-9 * (double)size_x * size_y / naiveTime, naiveTime, size_x * size_y, 1, 256);
    shrLogEx(LOGBOTH | MASTER, 0, "transpose-optimized, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n",
               1.0e-9 * (double)size_x * size_y / optimizedTime, optimizedTime, size_x * size_y, 1, 256);

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    // copy result from device to    host
    float* h_odata = (float*) malloc(mem_size);
    cutilSafeCall( cudaMemcpy( h_odata, d_odata, mem_size,
                                cudaMemcpyDeviceToHost) );

    // compute reference solution
    float* reference = (float*) malloc( mem_size);

    computeGold( reference, h_idata, size_x, size_y);

    // check result
    CUTBoolean res = cutComparef( reference, h_odata, size_x * size_y);
    shrLog("\n%s\n", (1 == res) ? "PASSED" : "FAILED");
	GpuProfiling::printResults();

    // cleanup memory
    free(h_idata);
    free(h_odata);
    free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));
    cutilCheckError( cutDeleteTimer(timer));

    cudaThreadExit();
}
