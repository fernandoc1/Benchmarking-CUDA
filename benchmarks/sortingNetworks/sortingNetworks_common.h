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



////////////////////////////////////////////////////////////////////////////////
// Shortcut definition
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;



///////////////////////////////////////////////////////////////////////////////
// Sort result validation routines
////////////////////////////////////////////////////////////////////////////////
//Sorted keys array validation (check for integrity and proper order)
extern "C" uint validateSortedKeys(
    uint *resKey,
    uint *srcKey,
    uint batchSize,
    uint arrayLength,
    uint numValues,
    uint dir
);

extern "C" int validateValues(
    uint *resKey,
    uint *resVal,
    uint *srcKey,
    uint batchSize,
    uint arrayLength
);



////////////////////////////////////////////////////////////////////////////////
// CUDA sorting networks
////////////////////////////////////////////////////////////////////////////////
//Return 1 if compiled for device emulation
extern "C" int isDeviceEmulation(void);

extern "C" uint bitonicSort(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint batchSize,
    uint arrayLength,
    uint dir
);

extern "C" void oddEvenMergeSort(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint batchSize,
    uint arrayLength,
    uint dir
);
