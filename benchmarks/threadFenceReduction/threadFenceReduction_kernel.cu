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

/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

#include <device_functions.h>

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n/2 threads
    - only works for power-of-2 arrays

    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    See the CUDA SDK "reduction" sample for more information.
*/

template <unsigned int blockSize>
__device__ void
reduceBlock(volatile float *sdata, float mySum, const unsigned int tid)
{
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        if (blockSize >=  64) { sdata[tid] = mySum = mySum + sdata[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { sdata[tid] = mySum = mySum + sdata[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { sdata[tid] = mySum = mySum + sdata[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { sdata[tid] = mySum = mySum + sdata[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { sdata[tid] = mySum = mySum + sdata[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { sdata[tid] = mySum = mySum + sdata[tid +  1]; EMUSYNC; }
    }
}

template <unsigned int blockSize, bool nIsPow2>
__device__ void
reduceBlocks(const float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];
        i += gridSize;
    }

    // do reduction in shared mem
    reduceBlock<blockSize>(sdata, mySum, tid);

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


template <unsigned int blockSize, bool nIsPow2>
__global__ void
reduceMultiPass(const float *g_idata, float *g_odata, unsigned int n)
{
    reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n);
}

// Global variable used by reduceSinglePass to count how many blocks have finished
__device__ unsigned int retirementCount = 0;


// This reduction kernel reduces an arbitrary size array in a single kernel invocation
// It does so by keeping track of how many blocks have finished.  After each thread
// block completes the reduction of its own block of data, it "takes a ticket" by
// atomically incrementing a global counter.  If the ticket value is equal to the number
// of thread blocks, then the block holding the ticket knows that it is the last block
// to finish.  This last block is responsible for summing the results of all the other
// blocks.
//
// In order for this to work, we must be sure that before a block takes a ticket, all
// of its memory transactions have completed.  This is what __threadfence() does -- it
// blocks until the results of all outstanding memory transactions within the
// calling thread are visible to all other threads.
//
// For more details on the reduction algorithm (notably the multi-pass approach), see
// the "reduction" sample in the CUDA SDK.
template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceSinglePass(const float *g_idata, float *g_odata, unsigned int n)
{

    //
    // PHASE 1: Process all inputs assigned to this block
    //

    reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n);

    //
    // PHASE 2: Last block finished will process all partial sums
    //

    if (gridDim.x > 1)
    {
        const unsigned int tid = threadIdx.x;
        __shared__ bool amLast;
        extern float __shared__ smem[];

        // wait until all outstanding memory instructions in this thread are finished
        __threadfence();

        // Thread 0 takes a ticket
        if( tid==0 )
        {
            unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
            // If the ticket ID is equal to the number of blocks, we are the last block!
            amLast = (ticket == gridDim.x-1);
        }
        __syncthreads();

        // The last block sums the results of all other blocks
        if( amLast )
        {
            int i = tid;
            float mySum = 0;

            while (i < gridDim.x)
            {
                mySum += g_odata[i];
                i += blockSize;
            }

            reduceBlock<blockSize>(smem, mySum, tid);

            if( tid==0 )
            {
                g_odata[0] = smem[0];

                // reset retirement count so that next run succeeds
                retirementCount = 0;
            }
        }
    }
}

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
extern "C"
void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    // choose which of the optimized versions of reduction to launch
    if (isPow2(size))
    {
        switch (threads)
        {
        case 512:
	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
            reduceMultiPass<512u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
	GpuProfiling::addResults("reduceMultiPass<512u, true>");
break;         case 256:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass<256u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
        	GpuProfiling::addResults("reduceMultiPass<256u, true>");
break;         case 128:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass<128u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
        	GpuProfiling::addResults("reduceMultiPass<128u, true>");
break;         case 64:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass< 64u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
        	GpuProfiling::addResults("reduceMultiPass< 64u, true>");
break;         case 32:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass< 32u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
        	GpuProfiling::addResults("reduceMultiPass< 32u, true>");
break;         case 16:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass< 16u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
        	GpuProfiling::addResults("reduceMultiPass< 16u, true>");
break;         case  8:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass<  8u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
        	GpuProfiling::addResults("reduceMultiPass<  8u, true>");
break;         case  4:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass<  4u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
        	GpuProfiling::addResults("reduceMultiPass<  4u, true>");
break;         case  2:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass<  2u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
        	GpuProfiling::addResults("reduceMultiPass<  2u, true>");
break;         case  1:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass<  1u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
        	GpuProfiling::addResults("reduceMultiPass<  1u, true>");
break;         }
            }
            else
            {
                switch (threads)
                {
                case 512:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass<512u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
        	GpuProfiling::addResults("reduceMultiPass<512u, false>");
 break;        case 256:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass<256u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
        	GpuProfiling::addResults("reduceMultiPass<256u, false>");
 break;        case 128:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass<128u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
        	GpuProfiling::addResults("reduceMultiPass<128u, false>");
 break;        case 64:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass< 64u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
        	GpuProfiling::addResults("reduceMultiPass< 64u, false>");
 break;        case 32:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass< 32u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
        	GpuProfiling::addResults("reduceMultiPass< 32u, false>");
 break;        case 16:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass< 16u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
        	GpuProfiling::addResults("reduceMultiPass< 16u, false>");
 break;        case  8:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass<  8u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
        	GpuProfiling::addResults("reduceMultiPass<  8u, false>");
 break;        case  4:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass<  4u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
        	GpuProfiling::addResults("reduceMultiPass<  4u, false>");
 break;        case  2:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass<  2u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
        	GpuProfiling::addResults("reduceMultiPass<  2u, false>");
 break;        case  1:
        	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                    reduceMultiPass<  1u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
        	GpuProfiling::addResults("reduceMultiPass<  1u, false>");
 break;        }
    }
}

extern "C"
void reduceSinglePass(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = threads * sizeof(float);

    // choose which of the optimized versions of reduction to launch
    if (isPow2(size))
    {
        switch (threads)
        {
        case 512:
	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
            reduceSinglePass<512u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
	GpuProfiling::addResults("reduceSinglePass<512u, true>");
break;        case 256:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                   reduceSinglePass<256u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass<256u, true>");
break;        case 128:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                   reduceSinglePass<128u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass<128u, true>");
break;        case 64:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                   reduceSinglePass< 64u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass< 64u, true>");
break;        case 32:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                   reduceSinglePass< 32u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass< 32u, true>");
break;        case 16:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                   reduceSinglePass< 16u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass< 16u, true>");
break;        case  8:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                   reduceSinglePass<  8u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass<  8u, true>");
break;        case  4:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                   reduceSinglePass<  4u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass<  4u, true>");
break;        case  2:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                   reduceSinglePass<  2u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass<  2u, true>");
break;        case  1:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                   reduceSinglePass<  1u, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass<  1u, true>");
break;        }
    }
    else
    {
        switch (threads)
        {
            case 512:
	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                reduceSinglePass<512u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
	GpuProfiling::addResults("reduceSinglePass<512u, false>");
break;            case 256:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                       reduceSinglePass<256u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass<256u, false>");
break;            case 128:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                       reduceSinglePass<128u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass<128u, false>");
break;            case 64:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                       reduceSinglePass< 64u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass< 64u, false>");
break;            case 32:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                       reduceSinglePass< 32u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass< 32u, false>");
break;            case 16:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                       reduceSinglePass< 16u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass< 16u, false>");
break;            case  8:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                       reduceSinglePass<  8u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass<  8u, false>");
break;            case  4:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                       reduceSinglePass<  4u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass<  4u, false>");
break;            case  2:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                       reduceSinglePass<  2u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass<  2u, false>");
break;            case  1:
       	GpuProfiling::prepareProfiling(  dimGrid, dimBlock, smemSize  );
                       reduceSinglePass<  1u, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
       	GpuProfiling::addResults("reduceSinglePass<  1u, false>");
break;        }
}
}

#endif // #ifndef _REDUCE_KERNEL_H_
