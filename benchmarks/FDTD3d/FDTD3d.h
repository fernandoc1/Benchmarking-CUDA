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

#ifndef _OCLFDTD3D_H_
#define _OCLFDTD3D_H_

// The values are set to give reasonable runtimes, they can
// be changed but note that running very large dimensions can
// take a very long time and you should avoid running on your
// primary display in this case.
#define k_dim_min           120
#define k_dim_max           376
#define k_dim_qa            248

// Note that the maximum radius is defined here as 4 since the
// minimum work group height is 4, if you have a larger work
// group then you can increase the radius accordingly.
#define k_radius_min        2
#define k_radius_max        4
#define k_radius_default    4

// The values are set to give reasonable runtimes, they can
// be changed but note that running a very large number of
// timesteps can take a very long time and you should avoid
// running on your primary display in this case.
#define k_timesteps_min     1
#define k_timesteps_max     10
#define k_timesteps_default 5

#endif
