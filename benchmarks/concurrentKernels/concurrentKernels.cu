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

//
// This sample demonstrates the use of streams for concurrent execution
//
// Devices of compute capability 1.x will run the kernels one after another
// Devices of compute capability 2.0 or higher can overlap the kernels
//
#include <prof.cu>
#include <stdlib.h>
#include <stdio.h>
#include <cutil_inline.h>

__global__ void mykernel( int *a, int n )
{
    int idx = threadIdx.x;
    int value = 1;

    for(int i=0; i<n; i++)
        value *= sin( (float)i ) + tan( (float)i );

    a[idx] = value;
}

int main(int argc, const char **argv)
{
	GpuProfiling::initProf();
    int nblocks = 4;
    int nthreads = 64;
    int n = 50000;
    int nkernels = 8;
    int nbytes;

    int devID;
    cudaDeviceProp deviceProps;

    int *d_A=0;
    cudaStream_t *stream;
    cudaEvent_t start, stop;
    float elapsedTime;

    int qatest = 0;


    // begin
    printf("[concurrentKernels] - Starting...\n\n");

    // get number of kernels if overridden on the command line
    if (cutCheckCmdLineFlag(argc, (const char **)argv, "nkernels")) {
        cutGetCmdLineArgumenti(argc, (const char **)argv, "nkernels", &nkernels);
    }

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if (cutCheckCmdLineFlag(argc, (const char**)argv, "device")) {
        cutGetCmdLineArgumenti(argc, (const char **) argv, "device", &devID);
    }
    else {
        devID = cutGetMaxGflopsDeviceId();
    }
    cutilSafeCall(cudaSetDevice(devID));

    // QA testing mode
    if (cutCheckCmdLineFlag(argc, (const char**)argv, "qatest")) {
        qatest = 1;
    }

    // get number of SMs on this GPU
    cutilSafeCall(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA Device %s has %d Multi-Processors\n", deviceProps.name, deviceProps.multiProcessorCount);
    printf("CUDA Device %s is%s capable of concurrent kernel execution\n", deviceProps.name, (deviceProps.concurrentKernels==0)?" NOT":"");

    stream = (cudaStream_t *)malloc(nkernels * sizeof(cudaStream_t));
    for(int i=0; i<nkernels; i++)
    {
        cutilSafeCall(cudaStreamCreate(&stream[i]));
    }

    // note: in this sample we will repeatedly overwrite the same
    // block of device mem, but that's okay because we don't really
    // care about the output of the kernel for the purposes of this
    // example.
    nbytes = nkernels * nthreads * sizeof(int);
    cutilSafeCall(cudaMalloc((void **)&d_A, nbytes));

    cutilSafeCall(cudaEventCreate(&start));
    cutilSafeCall(cudaEventCreate(&stop));

    // start timer then launch all kernels in their streams
    cutilSafeCall(cudaEventRecord(start, 0));
    for(int i=0; i<nkernels; i++)
    {
        // avoid synchronization points (events, error checks, etc.) inside
        // this loop in order to get concurrent execution on devices that support it
	GpuProfiling::prepareProfiling( nblocks, nthreads );
        mykernel<<<nblocks, nthreads, 0, stream[i]>>>(&d_A[i*nthreads], n);
	GpuProfiling::addResults("mykernel");
    }
    cutilSafeCall(cudaEventRecord(stop, 0));

    // wait for all streams to finish
    cutilSafeCall(cudaEventSynchronize(stop));

    // get total time for all kernels
    cutilSafeCall(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("\nAll %d kernels together took %.3fs\n", nkernels, elapsedTime/1000.f);

    // check time to execute a single iteration
    cutilSafeCall(cudaEventRecord(start, 0));
	GpuProfiling::prepareProfiling( nblocks, nthreads );
    mykernel<<<nblocks, nthreads, 0, stream[0]>>>(d_A, n);
	GpuProfiling::addResults("mykernel");
    cutilCheckMsg("kernel launch failure");
    cutilSafeCall(cudaEventRecord(stop, 0));
    cutilSafeCall(cudaEventSynchronize(stop));
    cutilSafeCall(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("(~%.3fs per kernel * %d kernels = ~%.3fs if no concurrent execution)\n",
           elapsedTime/1000.f, nkernels, nkernels*elapsedTime/1000.f);
	GpuProfiling::printResults();
    // cleanup
    printf("\nCleaning up...\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (stream)
    {
        for(int i=0; i<nkernels; i++)
        {
            cutilSafeCall(cudaStreamDestroy(stream[i]));
        }
        free(stream);
    }
    if (d_A) cudaFree(d_A);

    if (qatest) {
        // any errors that might have happened will have already been reported
        printf("[concurrentKernels] - Test Results:\nPASSED\n");
    }

    exit(0);
}
