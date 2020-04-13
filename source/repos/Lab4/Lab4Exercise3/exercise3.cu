/* Lab 4 Exercise 3 Program
In this exercise we write a CUDA program to implement a matrix addition kernel, using a GPU for parallel computation.
The kernel performs the addition of two `N` by `M` matrices filled with random integer entries. 
This program has been adapted from the code written for exercise 2 in which random vectors are added. 
We implement matrices as one-dimensional arrays using row-major vectorization to reduce the amount of modification required.
See the following for further reference:
https://en.wikipedia.org/wiki/Row-_and_column-major_order
https://en.wikipedia.org/wiki/Vectorization_(mathematics)
http://www.cplusplus.com/doc/tutorial/arrays/
See the "Multidimensional arrays" section of the last link for discussion of the subtle differences */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 2048 // The number of matrix rows (i.e. matrix height)
#define M 1000 // The number of matrix columns (i.e. matrix width)
/* We are required to use a 2D grid of thread blocks with 256 threads per block to perform the matrix addition
We choose to arrange the 256 threads within each block into a 16 by 16 square.
Blocks are split into 'warps' (groups of 32 threads) by the hardware, so threads per block should always be a multiple of 32.
More consideration needed on optimal block dimensions to minimise expected leftover threads over typical matrix shapes */
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
#define THREADS_PER_BLOCK BLOCK_HEIGHT*BLOCK_WIDTH // 16 * 16 = 256, which is divisible by 32

void checkCUDAError(const char*);
void random_ints(int *a);

/* 3.3 Rename your CPU implementation to `matrixAddCPU` and update it and the `validate` function accordingly.
Implements matrix addition using a CPU, taking inputs `a`, `b`, and writing the output to `c`. */
void matrixAddCPU(int *a, int *b, int *c) {
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < M; j++) {
			unsigned int idx = i * M + j;
			c[idx] = a[idx] + b[idx];
		}
	}
}

/* The `validate` function compares the GPU result to the CPU result.
It prints an error for each value which is incorrect and returns a value indicating the total number of errors. */
int validate(int *c_1, int *c_2) {
	int errors = 0;
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < M; j++) {
			unsigned int idx = i * M + j;
			if (c_1[idx] != c_2[idx]) {
				errors++;
				fprintf(stderr, "Error at index (%d,%d): GPU result %d does not match CPU value %d\n", i, j, c_2[idx], c_1[idx]);
			}
		}
	}
	return errors;
}

/* 3.4 Change your launch parameters to launch a 2D grid of thread blocks with 256 threads per block.
Create a new kernel `matrixAdd` to perform the matrix addition. */
__global__ void matrixAdd(int *a, int *b, int *c, unsigned int height, unsigned int width) {
	/* `blockDim.x` and `blockDim.y` are set by preprocessor definitions `BLOCK_WIDTH` and `BLOCK_HEIGHT`, respectively. */
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	// To avoid out-of-bounds memory errors with leftover threads in end blocks 
	if ((i < height) && (j < width)) {
		unsigned int idx = i * width + j;
		c[idx] = a[idx] + b[idx];
	}
}

int main(void) {
	int *a, *b, *c, *c_ref;	// Host copies of a, b, c
	int *d_a, *d_b, *d_c;	// Device copies of a, b, c
	int errors; // Count the number of errors/differences between GPU output and CPU output
	/* 3.1 Modify the value of `size` so that enough memory is allocated and subsequently moved between host and device
	for an `N` by `M` matrix of `int` values. */
	unsigned int size = N * M * sizeof(int); // Memory size of the matrix data

	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	checkCUDAError("CUDA malloc");

	// Allocate space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);
	c_ref = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA Memcpy Host to Device");

	/* 3.4 Change your launch parameters to launch a 2D grid of thread blocks with 256 threads per block.
	Create a new kernel `matrixAdd` to perform the matrix addition.*/
	// Launch the `matrixAdd` kernel on the GPU device
	unsigned int grid_width = (unsigned int) ceil((double) M / BLOCK_WIDTH);
	unsigned int grid_height = (unsigned int) ceil((double) N / BLOCK_HEIGHT);
	dim3 blocksPerGrid(grid_width, grid_height, 1);
	dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
	matrixAdd << <blocksPerGrid, threadsPerBlock >> >(d_a, d_b, d_c, N, M);
	checkCUDAError("CUDA kernel");

	// Run CPU version of the matrix addition function
	matrixAddCPU(a, b, c_ref);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA Memcpy Device to Host");

	// Validate the GPU result
	errors = validate(c, c_ref);
	printf("CUDA GPU result has %d errors.\n", errors);

	// Cleanup
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	checkCUDAError("CUDA cleanup");
	free(a); free(b); free(c);

	return 0;
}

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

/* 3.2 Modify the `random_ints` function to generate a matrix (rather than a vector) of random `int` values.
We use (row-major) vectorization to implement matrices in this exercise, so could merge the two loops into one anyway. */
void random_ints(int *a) {
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < M; j++) {
			a[i * M + j] = rand();
		}
	}
}
