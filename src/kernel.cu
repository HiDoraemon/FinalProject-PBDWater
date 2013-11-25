#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"

#if PRESSURE == 1
	#define DELTA_Q (float)(0.2*H)
	#define PRESSURE_K 0.001
	#define PRESSURE_N 4
#endif

#if SHARED == 1
    #define ACC(x,y,z) sharedMemAcc(x,y,z)
#else
    #define ACC(x,y,z) naiveAcc(x,y,z)
#endif

//GLOBALS
dim3 threadsPerBlock(blockSize);

int numParticles;
const float planetMass = 3e8;
const __device__ float starMass = 5e10;

const float scene_scale = 2e2; //size of the height map in simulation space

glm::vec4* particles;
glm::vec4* pred_particles;
glm::vec3* velocities;
int* neighbors;
int* num_neighbors;
glm::vec3* curl;
float* lambdas;
glm::vec3* delta_pos;
glm::vec3* external_forces;

void checkCUDAError(const char *msg, int line = -1)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        if( line >= 0 )
        {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
        exit(EXIT_FAILURE); 
    }
} 

__host__ __device__
unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Function that generates static.
__host__ __device__ 
glm::vec3 generateRandomNumberFromThread(float time, int index)
{
    thrust::default_random_engine rng(hash(index*time));
    thrust::uniform_real_distribution<float> u01(0,1);

    return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//Generate randomized starting positions for the planets in the XY plane
//Also initialized the masses
__global__
void generateRandomPosArray(int time, int N, glm::vec4 * arr, float scale, float mass)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = scale*(generateRandomNumberFromThread(time, index)-0.5f);
        arr[index].x = rand.x;
        arr[index].y = rand.y;
        arr[index].z = 0.0f;//rand.z;
        arr[index].w = mass;
    }
}

//Determine velocity from the distance from the center star. Not super physically accurate because 
//the mass ratio is too close, but it makes for an interesting looking scene
__global__
void generateCircularVelArray(int time, int N, glm::vec3 * arr, glm::vec4 * pos)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 R = glm::vec3(pos[index].x, pos[index].y, pos[index].z);
        float r = glm::length(R) + EPSILON;
        float s = sqrt(G*starMass/r);
        glm::vec3 D = glm::normalize(glm::cross(R/r,glm::vec3(0,0,1)));
        arr[index].x = s*D.x;
        arr[index].y = s*D.y;
        arr[index].z = s*D.z;
    }
}

//Generate randomized starting velocities in the XY plane
__global__
void generateRandomVelArray(int time, int N, glm::vec3 * arr, float scale)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = scale*(generateRandomNumberFromThread(time, index) - 0.5f);
        arr[index].x = rand.x;
        arr[index].y = rand.y;
        arr[index].z = 0.0;//rand.z;
    }
}

//TODO: Determine force between two bodies
__device__
glm::vec3 calculateAcceleration(glm::vec4 us, glm::vec4 them)
{
    //    G*m_us*m_them
    //F = -------------
    //         r^2
    //
    //    G*m_us*m_them   G*m_them
    //a = ------------- = --------
    //      m_us*r^2        r^2
    
    return glm::vec3(0.0f);
}

//TODO: Core force calc kernel global memory
__device__ 
glm::vec3 naiveAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
    glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
    return acc;
}


//TODO: Core force calc kernel shared memory
__device__ 
glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
    glm::vec3 acc = calculateAcceleration(my_pos, glm::vec4(0,0,0,starMass));
    return acc;
}

//Simple Euler integration scheme
__global__
void updateF(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    glm::vec4 my_pos;
    glm::vec3 accel;

    if(index < N) my_pos = pos[index];

    accel = ACC(N, my_pos, pos);

    if(index < N) acc[index] = accel;
}

__global__
void updateS(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if( index < N )
    {
        vel[index]   += acc[index]   * dt;
        pos[index].x += vel[index].x * dt;
        pos[index].y += vel[index].y * dt;
        pos[index].z += vel[index].z * dt;
    }
}

//Update the vertex buffer object
//(The VBO is where OpenGL looks for the positions for the planets)
__global__
void sendToVBO(int N, glm::vec4 * pos, float * vbo, int width, int height, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale_w = -2.0f / s_scale;
    float c_scale_h = -2.0f / s_scale;
	float c_scale_z = 2.0f / s_scale;

    if(index<N)
    {
        vbo[4*index+0] = pos[index].x*c_scale_w;
        vbo[4*index+1] = pos[index].y*c_scale_h;
        vbo[4*index+2] = pos[index].z*c_scale_z;
        vbo[4*index+3] = 1;
    }
}

//Update the texture pixel buffer object
//(This texture is where openGL pulls the data for the height map)
__global__
void sendToPBO(int N, glm::vec4 * pos, float4 * pbo, int width, int height, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    int x = index % width;
    int y = index / width;
    float w2 = width / 2.0;
    float h2 = height / 2.0;

    float c_scale_w = width / s_scale;
    float c_scale_h = height / s_scale;

    glm::vec3 color(0.05, 0.15, 0.3);
    glm::vec3 acc = ACC(N, glm::vec4((x-w2)/c_scale_w,(y-h2)/c_scale_h,0,1), pos);

    if(x<width && y<height)
    {
        float mag = sqrt(sqrt(acc.x*acc.x + acc.y*acc.y + acc.z*acc.z));
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = (mag < 1.0f) ? mag : 1.0f;
    }
}

/*************************************
 * Device Methods for Solver
 *************************************/

__device__ float wPoly6Kernel(glm::vec3 p_i, glm::vec3 p_j){
	float r = glm::length(p_i - p_j);
	float hr_term = (H * H - r * r);
	return 315.0f / (64.0 * PI * POW_H_9) * hr_term * hr_term * hr_term;
}

__device__ glm::vec3 wGradientSpikyKernel(glm::vec3 p_i, glm::vec3 p_j){
	glm::vec3 r = p_i - p_j;
	float hr_term = H - glm::length(r);
	float gradient_magnitude = 45.0f / (PI * POW_H_6) * hr_term * hr_term;
	return gradient_magnitude * glm::normalize(r);
}

__device__ float calculateRo(glm::vec4* particles, glm::vec3 p, int* p_neighbors, int p_num_neighbors, int index){
	glm::vec3 p_j;
	float ro = 0.0f;
	for(int i = 0; i < p_num_neighbors; i++){
		p_j = glm::vec3(particles[p_neighbors[i + index * MAX_NEIGHBORS]]);
		ro += wPoly6Kernel(p, p_j);
	}
	return ro;
}

__device__ glm::vec3 calculateCiGradient(glm::vec3 p_i, glm::vec3 p_j){
	return -1.0f * wGradientSpikyKernel(p_i, p_j);
}

__device__ glm::vec3 calculateCiGradientAti(glm::vec4* particles, glm::vec3 p_i, int* neighbors, int p_num_neighbors, int index){
	glm::vec3 accum = glm::vec3(0.0f);
	for(int i = 0; i < p_num_neighbors; i++){
		accum += wGradientSpikyKernel(p_i, glm::vec3(particles[neighbors[i + index * MAX_NEIGHBORS]]));
	}
	return accum;
}

/*************************************
 * Finding Neighboring Particles 
 *************************************/
__global__ void findNeighbors(glm::vec4* pred_particles, int* neighbors, int* num_neighbors, int num_particles){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if(index < num_particles){
		glm::vec3 p = glm::vec3(pred_particles[index]);
		int num_p_neighbors = 0;
		glm::vec3 p_j, r;
		for(int i = 0; i < num_particles && num_p_neighbors < MAX_NEIGHBORS; i++){
			if(i != index){
				p_j = glm::vec3(pred_particles[i]);
				r = p_j - p;
				if(glm::length(r) <= H){
					neighbors[num_p_neighbors + index * MAX_NEIGHBORS] = i;
					++num_p_neighbors;
				}
			}
		}
		num_neighbors[index] = num_p_neighbors;
	}
}

/*************************************
 * Kernels for Jacobi Solver
 *************************************/

__global__ void calculateLambda(glm::vec4* particles, int* neighbors, int* num_neighbors, float* lambdas, int num_particles){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < num_particles){
		int k = num_neighbors[index];
		glm::vec3 p = glm::vec3(particles[index]);

		float p_i = calculateRo(particles, p, neighbors, k, index);
		float C_i = (p_i / REST_DENSITY) - 1.0f;

		
		float C_i_gradient, sum_gradients = 0.0f;
		for(int i = 0; i < k; i++){
			// Calculate gradient when k = j
			C_i_gradient = glm::length(calculateCiGradient(p, glm::vec3(particles[neighbors[i + index * MAX_NEIGHBORS]])));
			sum_gradients += (C_i_gradient * C_i_gradient);
		}

		// Add gradient when k = i
		C_i_gradient = glm::length(calculateCiGradientAti(particles, p, neighbors, k, index));
		sum_gradients += (C_i_gradient * C_i_gradient);

		lambdas[index] = -1.0f * (C_i / (sum_gradients + RELAXATION)); 
	}
}

__global__ void calculateDeltaPi(glm::vec4* particles, int* neighbors, int* num_neighbors, float* lambdas, glm::vec3* delta_pos, int num_particles){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < num_particles){
		int k = num_neighbors[index];
		glm::vec3 p = glm::vec3(particles[index]);
		float l = lambdas[index];
		
		glm::vec3 delta = glm::vec3(0.0f);
		int p_j_idx;
#if PRESSURE == 1
		float k_term;
		glm::vec3 d_q = DELTA_Q * glm::vec3(1.0f) + p;
#endif
		float s_corr = 0.0f;
		for(int i = 0; i < k; i++){
			p_j_idx = neighbors[i + index * MAX_NEIGHBORS];
#if PRESSURE == 1
			k_term = wPoly6Kernel(p, glm::vec3(particles[p_j_idx])) / wPoly6Kernel(p, d_q);
			s_corr = -1.0f * PRESSURE_K * pow(k_term, PRESSURE_N);
#endif
			delta += (l + lambdas[p_j_idx] + s_corr) * wGradientSpikyKernel(p, glm::vec3(particles[p_j_idx]));
		}
		delta_pos[index] = 1.0f / REST_DENSITY * delta;
	}
}

__global__ void calculateCurl(glm::vec4* particles, int* neighbors, int* num_neighbors, glm::vec3* velocities, glm::vec3* curl, int num_particles){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < num_particles){
		int k = num_neighbors[index];
		glm::vec3 p = glm::vec3(particles[index]);
		glm::vec3 v = velocities[index];

		int j_idx;
		glm::vec3 v_ij, gradient, accum = glm::vec3(0.0f);
		for(int i = 0; i < k; i++){
			j_idx = neighbors[i + index * MAX_NEIGHBORS];
			v_ij = velocities[j_idx] - v;
			gradient = wGradientSpikyKernel(p, glm::vec3(particles[j_idx]));
			accum += glm::cross(v_ij, gradient);
		}
		curl[index] = accum;
	}
}

__global__ void applyVorticity(glm::vec4* particles, int* neighbors, int* num_neighbors, glm::vec3* curl, glm::vec3* external_forces, int num_particles){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < num_particles){
		int k = num_neighbors[index];
		glm::vec3 p = glm::vec3(particles[index]);
		glm::vec3 w = curl[index];

		int j_idx;
		float mag_w;
		glm::vec3 r, grad = glm::vec3(0.0f);
		for(int i = 0; i < k; i++){
			j_idx = neighbors[i + index * MAX_NEIGHBORS];
			r = glm::vec3(particles[j_idx]) - p;
			mag_w = glm::length(curl[j_idx] - w);
			grad.x += mag_w / r.x;
			grad.y += mag_w / r.y;
			grad.z += mag_w / r.z;
		}
		
		glm::vec3 vorticity, N;
		N = 1.0f/(glm::length(grad) + .001f) * grad;
		vorticity = float(RELAXATION) * (glm::cross(N, w));
		external_forces[index] += vorticity;
	}
}


__global__
void initializeParticles(int N, glm::vec4* particles, glm::vec3* velocities, glm::vec3* external_forces)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	float gravity = -2.0f;
    if(index < N)
    {
		particles[index].x = (index % 50)-25;
		particles[index].y = (index / 50)-25;
		particles[index].z = 50.0f;
		particles[index].w = 1.0f;

		velocities[index] = glm::vec3(0.0f);
		
		external_forces[index] = glm::vec3(0.0f,0.0f,gravity);
    }
}

//Simple Euler integration scheme
__global__
void applyExternalForces(int N, float dt, glm::vec4* pred_particles, glm::vec4* particles, glm::vec3* velocities, glm::vec3* external_forces)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    if(index < N){
		velocities[index] = velocities[index] + dt*external_forces[index];
		pred_particles[index] = particles[index] + dt*glm::vec4(velocities[index],0.0f);
	}
}

__global__
void updatePosition(int N, glm::vec4* pred_particles, glm::vec4* particles)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if(index < N){
		particles[index] = pred_particles[index];
	}
}

__global__
void updatePredictedPosition(int N, glm::vec4* pred_particles, glm::vec3* delta_pos)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if(index < N){
		pred_particles[index] = pred_particles[index]+glm::vec4(delta_pos[index],0.0f);
	}
}

__global__
void updateVelocity(int N, glm::vec4* pred_particles, glm::vec4* particles, glm::vec3* velocities, float dt)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if(index < N){
		velocities[index] = glm::vec3((1.0f/dt)*(pred_particles[index] - particles[index]));
	}
}

__global__
void tempCollisionResponse(int N, glm::vec4* pred_particles, glm::vec3* velocities){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if(index < N){
		if(pred_particles[index].z < 0.0f){
			pred_particles[index].z = 0.0001f;
			velocities[index].z = -1.0f*velocities[index].z;
		}
		if(pred_particles[index].z > 100.0f){
			pred_particles[index].z = 100.0f-0.0001f;
			velocities[index].z = -1.0f*velocities[index].z;
		}
		if(pred_particles[index].y < -50.0f){
			pred_particles[index].y = -50.0f+0.0001f;
			velocities[index].y = -1.0f*velocities[index].y;
		}
		if(pred_particles[index].y > 50.0f){
			pred_particles[index].y = 50.0f-0.0001f;
			velocities[index].y = -1.0f*velocities[index].y;
		}
		if(pred_particles[index].x < -50.0f){
			pred_particles[index].x = -50.0f+0.0001f;
			velocities[index].x = -1.0f*velocities[index].x;
		}
		if(pred_particles[index].x > 50.0f){
			pred_particles[index].x = 50.0f-0.0001f;
			velocities[index].x = -1.0f*velocities[index].x;
		}
	}
}

/*************************************
 * Wrappers for the __global__ calls *
 *************************************/

//Initialize memory, update some globals
void initCuda(int N)
{
	numParticles = N;
    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));

    cudaMalloc((void**)&particles, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("particles cudamalloc failed!");
	cudaMalloc((void**)&pred_particles, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("pred_particles cudamalloc failed!");
    cudaMalloc((void**)&velocities, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("velocities cudamalloc failed!");

	cudaMalloc((void**)&neighbors, MAX_NEIGHBORS*N*sizeof(int));
	cudaMalloc((void**)&num_neighbors, N*sizeof(int));
    checkCUDAErrorWithLine("num_neighbors cudamalloc failed!");

    cudaMalloc((void**)&curl, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("curl cudamalloc failed!");
	cudaMalloc((void**)&lambdas, N*sizeof(float));
    checkCUDAErrorWithLine("lambdas cudamalloc failed!");
	cudaMalloc((void**)&delta_pos, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("delta_pos cudamalloc failed!");
	cudaMalloc((void**)&external_forces, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("external_forces cudamalloc failed!");

    initializeParticles<<<fullBlocksPerGrid, blockSize>>>(N, particles, velocities, external_forces);
    checkCUDAErrorWithLine("Kernel failed!");
    cudaThreadSynchronize();
}

void cudaNBodyUpdateWrapper(float dt)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numParticles)/float(blockSize)));
    applyExternalForces<<<fullBlocksPerGrid, blockSize>>>(numParticles, dt, pred_particles, particles, velocities, external_forces);
    checkCUDAErrorWithLine("applyExternalForces failed!");
	findNeighbors<<<fullBlocksPerGrid, blockSize>>>(pred_particles, neighbors, num_neighbors, numParticles);
    checkCUDAErrorWithLine("findNeighbors failed!");
	for(int i = 0; i < SOLVER_ITERATIONS; i++){
		calculateLambda<<<fullBlocksPerGrid, blockSize>>>(pred_particles, neighbors, num_neighbors, lambdas, numParticles);
		calculateDeltaPi<<<fullBlocksPerGrid, blockSize>>>(pred_particles, neighbors, num_neighbors, lambdas, delta_pos, numParticles);
		//PEFORM COLLISION DETECTION AND RESPONSE
		tempCollisionResponse<<<fullBlocksPerGrid, blockSize>>>(numParticles, pred_particles, velocities);
		updatePredictedPosition<<<fullBlocksPerGrid, blockSize>>>(numParticles,pred_particles, delta_pos);
	}


	updateVelocity<<<fullBlocksPerGrid, blockSize>>>(numParticles, pred_particles, particles, velocities, dt);
	//calculateCurl<<<fullBlocksPerGrid, blockSize>>>(particles, neighbors, num_neighbors, velocities, curl, numParticles);
	//applyVorticity<<<fullBlocksPerGrid, blockSize>>>(particles, neighbors, num_neighbors, curl, external_forces, numParticles);
	updatePosition<<<fullBlocksPerGrid, blockSize>>>(numParticles, pred_particles, particles);
    checkCUDAErrorWithLine("updatePosition failed!");
    cudaThreadSynchronize();
}

void cudaUpdateVBO(float * vbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numParticles)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numParticles, particles, vbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}

void cudaUpdatePBO(float4 * pbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(width*height)/float(blockSize)));
    sendToPBO<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numParticles, particles, pbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}

void freeCuda(){
	cudaFree(particles);
	cudaFree(pred_particles);
	cudaFree(velocities);
	cudaFree(neighbors);
	cudaFree(num_neighbors);
	cudaFree(curl);
	cudaFree(lambdas);
	cudaFree(delta_pos);
	cudaFree(external_forces);
}

