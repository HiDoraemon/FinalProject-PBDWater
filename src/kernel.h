#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define blockSize 128
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define SHARED 0
#define PRESSURE 1
#define HEAP 0

#define BOX_X 10
#define BOX_Y 20
#define BOX_Z 40

#define SOLVER_ITERATIONS 3
#define MAX_NEIGHBORS 20
#define REST_DENSITY 10000.0f // 1000 kg/m^3
#define H 1.5f // smoothing radius
#define POW_H_9 (float)(H*H*H*H*H*H*H*H*H) // h^9
#define POW_H_6 (float)(H*H*H*H*H*H) // h^6
#define RELAXATION .01 // relaxation term in lambda calculation

void checkCUDAError(const char *msg, int line);
void cudaPBFUpdateWrapper(float dt);
void initCuda(int N, staticGeom* geoms, int numg);
void cudaUpdateVBO(float * vbodptr, int width, int height);
void freeCuda();
#endif
