/* Lab 4 Exercise 2 Program
In this exercise we write a CUDA program to implement vector addition on vectors of random integers of size N (= 2048). */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 2050
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);

/* 2.1.1 Implement a CPU version of vector addition for validation purposes
Implement a CPU version of the vector addition function called `vectorAddCPU` and 
store the result in an array called `c_ref`.*/
void vectorAddCPU(int *a, int *b, int *c) {
	for (unsigned int i = 0; i < N; i++) {
		c[i] = a[i] + b[i];
	}
}

/* 2.1.2 Implement a new function called `validate` which compares the GPU result to the CPU result.
It should print an error for each value which is incorrect and return a value indicating 
the total number of errors. You should also print the number of errors to the console. */
int validate(int *c_1, int *c_2) {
	int errors = 0;
	for (unsigned int i = 0; i < N; i++) {
		if (c_1[i] != c_2[i]) {
			errors++;
			fprintf(stderr, "Error at index %d: GPU result %d does not match CPU value %d\n", i, c_2[i], c_1[i]);
		}
	}
	return errors;
}

__global__ void vectorAdd(int *a, int *b, int *c, int max) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	// To avoid out-of-bounds memory errors with any leftover threads in last block (equal to `ceil(N/THREADS_PER_BLOCK) - N`)
	if (i < max) {
		c[i] = a[i] + b[i]; // Originally subtracting rather than adding
	}
}

int main(void) {
	int *a, *b, *c, *c_ref;	// Host copies of a, b, c
	int *d_a, *d_b, *d_c;	// Device copies of a, b, c
	int errors; // Count the number of errors/differences between GPU output and CPU output
	unsigned int size = N * sizeof(int); // Memory size of the vector data

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

	// Launch `vectorAdd` kernel on GPU
	// Ensure we have a whole number of blocks so that the total number of threads is at least `N`
	dim3 blocksPerGrid((unsigned int) ceil((double)N / THREADS_PER_BLOCK), 1, 1);
	dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
	vectorAdd << <blocksPerGrid, threadsPerBlock >> >(d_a, d_b, d_c, N);
	checkCUDAError("CUDA kernel");

	// Run CPU version of the vector addition function
	vectorAddCPU(a, b, c_ref);

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

void random_ints(int *a) {
	for (unsigned int i = 0; i < N; i++) {
		a[i] = rand();
	}
}
