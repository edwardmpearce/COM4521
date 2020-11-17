/* Lab 6 Exercise 1 Program
In this program we use shared memory to improve performance of a memory bound matrix multiplication algorithm and
manually calculate the occupancy of a kernel at runtime.
The program populates matrices A and B with randomly generated entries, then calculates the product C = AB.
The widths and heights of the matrices can be modified but for simplicity must be a factor of `BLOCK_SIZE`.
The initial implementation is highly memory inefficient as it performs `A_WIDTH` * `B_HEIGHT` memory loads to compute each
entry in matrix C. To improve this we will implement a blocked matrix multiply which uses CUDA's shared memory to reduce
the number of memory reads by a factor of `BLOCK_SIZE`.
First note the performance of the original version and then modify the existing code to perform a blocked matrix multiply.
To implement a blocked matrix multiply will require that we load `NUM_SUBS` square sub matrices of
the matrix A and B into shared memory so that we can compute the intermediate result of the sub
matrix products. In the example Figure 1 (above) the sub matrix C(1,1) can be calculated by a square
thread block of BLOCK_SIZE by BLOCK_SIZE where each thread (with location tx, ty in the
square thread block) performs the following steps which requires two stages of loading matrix tiles
into shared memory.


*/
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define A_HEIGHT 1024
#define A_WIDTH 1024
#define B_HEIGHT 1024
#define B_WIDTH 1024
#define C_HEIGHT A_HEIGHT
#define C_WIDTH B_WIDTH

#define BLOCK_SIZE 8
#define NUM_SUBS (A_WIDTH / BLOCK_SIZE)

__device__ float d_A[A_HEIGHT][A_WIDTH];
__device__ float d_B[B_HEIGHT][B_WIDTH];
__device__ float d_C[C_HEIGHT][C_WIDTH];

float h_A[A_HEIGHT][A_WIDTH];
float h_B[B_HEIGHT][B_WIDTH];
float h_C[C_HEIGHT][C_WIDTH];
float h_C_ref[C_HEIGHT][C_WIDTH];

void checkCUDAError(const char* msg);
void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[B_HEIGHT][B_WIDTH], float C[C_HEIGHT][C_WIDTH]);
int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]);

__global__ void matrixMulCUDA()
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = bx * BLOCK_SIZE + tx;
	int y = by * BLOCK_SIZE + ty;


	float Csub = 0;
	//iterate A_WIDTH (same as B_HEIGHT) to calculate the product
	for (int k = 0; k < A_WIDTH; k++) {
		Csub += d_A[y][k] * d_B[k][x];
	}

	// Store the product value of C matrix
	d_C[y][x] = Csub;
}

__global__ void matrixMulCUDASharedMemory()
{
	//Define some shared memory for a sub block of matrices A an B
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	//Running sum of product of A and B matrices
	float Csub = 0;

	//iterate through the number of sub matrices of A and B
	for (int i = 0; i < NUM_SUBS; i++) {
		//TODO: Calculate indices of A and B matrix required to load the shared block of memory
		int a_x;
		int a_y;
		int b_x;
		int b_y;

		//TODO: Each thread should load a single element of sub block of matrices A an B into shared memory

		// Sync to ensure sub matrix is fully loaded
		__syncthreads();

		//TODO: sum products of A and B sub matrices
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{

		}

		// Sync to prevent run ahead (blocks loading new SM values before others have completed)
		__syncthreads();

	}

	//TODO: caluclate the indices of sub matrix C
	int c_x;
	int c_y;

	// Store the product value of C matrix
	d_C[c_y][c_x] = Csub;
}


int main(int argc, char** argv) {
	unsigned int mem_size_A, mem_size_B, mem_size_C;
	unsigned int x, y, errors;
	int maxActiveBlocks;
	float msec, occupancy;
	cudaDeviceProp props;
	cudaEvent_t start, stop;

	if (A_WIDTH != B_HEIGHT) {
		printf("Error: A_HEIGHT and B_WIDTH do not match\n");
	}

	mem_size_A = A_HEIGHT * A_WIDTH * sizeof(float);
	mem_size_B = B_HEIGHT * B_WIDTH * sizeof(float);
	mem_size_C = C_HEIGHT * C_WIDTH * sizeof(float);

	// Initialise matrix A with values (pseudo)randomly distributed uniformly in [0,1]
	for (y = 0; y < A_HEIGHT; y++) {
		for (x = 0; x < A_WIDTH; x++) {
			h_A[y][x] = (float)rand() / RAND_MAX;
		}
	}
	// Initialise matrix B with values (pseudo)randomly distributed uniformly in [0,1]
	for (y = 0; y < B_HEIGHT; y++) {
		for (x = 0; x < B_WIDTH; x++) {
			h_B[y][x] = (float)rand() / RAND_MAX;
		}
	}

	// Copy matrix data from host memory to device
	cudaMemcpyToSymbol(d_A, h_A, mem_size_A);
	cudaMemcpyToSymbol(d_B, h_B, mem_size_B);
	checkCUDAError("CUDA Memcpy from Host to Device");

	// Allocate CUDA events that we'll use for timing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	checkCUDAError("CUDA event creation");

	// Setup kernelexecution parameters
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(C_WIDTH / BLOCK_SIZE, C_HEIGHT / BLOCK_SIZE);

	cudaEventRecord(start);
	matrixMulCUDA << < grid, threads >> > ();
	//TODO: Comment out the above line and complete the shared memory version of the kernel
	//matrixMulCUDASharedMemory << < grid, threads >> >();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	checkCUDAError("CUDA kernel execution and timing");

	cudaEventElapsedTime(&msec, start, stop);
	cudaThreadSynchronize();
	checkCUDAError("CUDA timing");

	// Compute the ocupancy
	occupancy = 0;

	// Copy result from device to host
	cudaMemcpyFromSymbol(h_C, d_C, mem_size_C);
	checkCUDAError("CUDA Memcpy from Device to Host");

	// Compute reference CPU version
	matrixMulCPU(h_A, h_B, h_C_ref);

	// Check for errors
	errors = matrixMulTest(h_C, h_C_ref);
	if (errors) {
		printf("%d total errors\n", errors);
	}
	else {
		printf("Test passed successfully\n");
	}
	printf("Kernel time was %fms with theoretical occupancy of %f\n", msec, occupancy);
}

void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[C_HEIGHT][C_WIDTH], float C[C_HEIGHT][C_WIDTH]) {
	int x, y, k;
	for (y = 0; y < C_HEIGHT; y++) {
		for (x = 0; x < C_WIDTH; x++) {
			C[y][x] = 0;
			for (k = 0; k < A_WIDTH; k++) {
				C[y][x] += A[y][k] * B[k][x];
			}
		}
	}
}

int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]) {
	int errors = 0;
	int y, x;
	for (y = 0; y < C_HEIGHT; y++) {
		for (x = 0; x < C_WIDTH; x++) {
			if (C[y][x] != Cref[y][x]) {
				errors++;
				printf("Device item c[%d][%d] = %f does not mach host result %f\n", y, x, C[y][x], Cref[y][x]);
			}
		}
	}
	return errors;
}

void checkCUDAError(const char* msg) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
