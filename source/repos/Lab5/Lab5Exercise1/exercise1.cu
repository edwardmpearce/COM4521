/* Lab 5 Exercise 1 Program
In this exercise we extend our CUDA program which implements vector addition by using statically defined
global device memory, timing kernel execution, and comparing theoretical and measured bandwidth. */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 65536
#define THREADS_PER_BLOCK 128
#define PUMP_RATE 2		// Used in theoretical bandwidth calculation as device uses double-data-rate memory
#define READ_BYTES 2 * 4  // Number of bytes read by the kernel; from reading an `int` each to `d_a[i]` and `d_b[i]`
#define WRITE_BYTES 4 // Number of bytes written by the kernel; from writing an `int` from `d_c[i]`

void checkCUDAError(const char*);
void random_ints(int *a);

/* 1.1 Modify the example to use statically defined global variables. 
This means array size will be declared at compile time rather than at runtime. */
__device__ int d_a[N], d_b[N], d_c[N];

/* 1.1 A device symbol (statically defined CUDA memory) is not the same as a device address in the host code. 
Passing a symbol as an argument to the kernel launch will cause invalid memory accesses in the kernel. 
We removed the pointer arguments from the kernel definition and kernel function call */
__global__ void vectorAdd() {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		d_c[i] = d_a[i] + d_b[i];
	}
}

/* 1.1 We removed the device pointer declarations and associated `cudaMalloc` and `cudaFree` calls, 
changed `cudaMemcpy` to `cudaMemcpyToSymbol` and `cudaMemcpyFromSymbol` as appropriate (no need to specify
`cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost` arguments) in the `main` function. */
int main(void) {
	int *a, *b, *c;			// Host copies of a, b, c
	unsigned int size = N * sizeof(int);

	/* 1.2 Modify the code to record timing data of the kernel execution (using `cudaEvent`)
	Print this data to the terminal */
	cudaEvent_t start, stop;	// CUDA Event timers
	float milliseconds = 0;		// Variable to store timer results

	/* 1.3 We would like to query the device properties so that we can calculate the theoretical memory
	bandwidth of the device. The formula for theoretical bandwidth is given by
	theoreticalBW = memoryClockRate * memoryBusWidth */
	int deviceCount = 0;		// Variable to store count of available devices (GPUs)
	double theoreticalBW;		// To store the theoretical bandwidth
	double measuredBW;			// To store the measured bandwidth

	/* 1.3	Using `cudaDeviceProp`, query the `memoryClockRate` and `memoryBusWidth` from the first 
	`cudaDevice` available and multiply their product by two to calculate the theoretical bandwidth.
	The factor of two is introduced since the device uses Double-data-rate memory, which is double pumped.
	See https://en.wikipedia.org/wiki/Double_data_rate */
	cudaGetDeviceCount(&deviceCount);	// Count the number of available devices and write to `deviceCount`
	if (deviceCount > 0) {
		cudaSetDevice(0);		// Set device to be used for GPU executions (the first available)
		cudaDeviceProp deviceProp;	// Declare variable to hold device properties
		cudaGetDeviceProperties(&deviceProp, 0);	// Write properties of first device to `deviceProp`
		theoreticalBW = deviceProp.memoryClockRate * deviceProp.memoryBusWidth * PUMP_RATE;
		/* This will calculate the result in kilobits/second (as `memoryClockRate` is measured in kilohertz 
		   and `memoryBusWidth` is measured in bits) */
	}

    // Create CUDA Events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpyToSymbol(d_a, a, size);
	cudaMemcpyToSymbol(d_b, b, size);
	checkCUDAError("CUDA Memcpy Host to Device");

	// Launch `vectorAdd()` kernel on GPU
	// Ensure we have a whole number of blocks so that the total number of threads is at least `N`
	dim3 blocksPerGrid((unsigned int)ceil((double)N / THREADS_PER_BLOCK), 1, 1);
	dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
	cudaEventRecord(start);		// Record the start time before calling the kernel
	vectorAdd << <blocksPerGrid, threadsPerBlock >> >();
	cudaEventRecord(stop);		// Record the stop time once the kernel is finished
	cudaEventSynchronize(stop); // Ensure stop time has finished recording before continuing
	checkCUDAError("CUDA kernel");

	cudaEventElapsedTime(&milliseconds, start, stop);	// Write the elapsed time to `milliseconds`

	/* 1.4 Theoretical bandwidth is the maximum bandwidth we could achieve in ideal conditions. 
	Now we calculate the measured bandwidth of the `vectorAdd` kernel. Measured bandwidth is given by
	measuredBW = (RBytes + WBytes) / t
	where `RBytes` is the number of bytes read and `WBytes` is the number of bytes written by the kernel. 
	You can calculate these values by considering how many bytes the kernel reads and writes per thread
	and multiplying by the number of threads that are launched. Get `t` value from your timing data. */
	measuredBW = (double) N * (READ_BYTES + WRITE_BYTES) / milliseconds;
	/* This will calculate the result in kilobytes/second as we divide by time measured in milliseconds */

	// Copy result back to host
	cudaMemcpyFromSymbol(c, d_c, size);
	checkCUDAError("CUDA Memcpy Device to Host");

	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(a); free(b); free(c);
	checkCUDAError("CUDA cleanup");

	printf("Execution time is %f ms\n", milliseconds);
	/* 1.3 Print the theoretical bandwidth to the console in GB/s (Giga Bytes per second).
	We need to convert from kilobits/second to GB/s. */
	printf("Theoretical Bandwidth is %f GB/s\n", (theoreticalBW / 8.0) / 1e6); ; // Convert kb/s to GB/s
	/* 1.4 Print the measured bandwidth to the console for comparison with the theoretical bandwidth. */
	printf("Measured Bandwidth is %f GB/s\n", measuredBW / 1e6); // Convert kB/s to GB/s
	/* Don’t forget to switch to Release mode to profile your code execution times.
	                                         Theoretical Bandwidth: 243.328 GB/s
	In Debug Mode:   Execution time: 0.016  ms, Measured Bandwidth:  48.956 GB/s
	In Release Mode: Execution time: 0.0064 ms, Measured Bandwidth: 122.279 GB/s */
	return 0;
}

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int *a) {
	for (unsigned int i = 0; i < N; i++) {
		a[i] = rand();
	}
}
