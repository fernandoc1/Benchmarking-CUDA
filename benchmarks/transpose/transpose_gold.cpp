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

/* Small Matrix transpose with Cuda (Example for a 16x16 matrix)
* Reference solution.
*/

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float* reference, float* idata,
                  const unsigned int size_x, const unsigned int size_y );

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
////////////////////////////////////////////////////////////////////////////////
void
computeGold( float* reference, float* idata,
            const unsigned int size_x, const unsigned int size_y )
{
    // transpose matrix
    for( unsigned int y = 0; y < size_y; ++y)
    {
        for( unsigned int x = 0; x < size_x; ++x)
        {
            reference[(x * size_y) + y] = idata[(y * size_x) + x];
        }
    }
}

