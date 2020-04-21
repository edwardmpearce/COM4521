/* Lab 4 Exercise 1 Program
In this exercise we write a CUDA program to decipher some text encoded using an affine cipher (in the file `encrypted1.bin`)
An affine cipher is a monoalphabetic substitution cipher, and it is decrypted by an affine decipher
The encryption function is E(x) = (Ax + B) mod M
The decryption function is D(x) = A^{−1}(x − B) mod M
In this exercise `M` is 128 (the size of the ASCII alphabet), `A` is 15, `B` is 27, and so `A^{-1}` is 111.
As each of the encrypted character values are independent we can use the GPU to decrypt them in parallel. 
To do this we will launch a thread for each of the encrypted character values and use a
kernel function to perform the decryption. */
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 1024 // The number of characters in the encrypted text
#define A 111 // The inverse multiplier to the encryption multiplier
#define B 27 // The shift value in the encryption function
#define M 128 // The modulus of the encryption (the size of the ASCII alphabet)
#define MULTIBLOCK 1 // Set this definition to `0` for single block, `1` to run the multiblock kernel

void checkCUDAError(const char*);
void read_encrypted_file(int*);

/* 1.1 Modify the `modulo` function so that it can be called on the device by the `affine_decrypt` kernel. */
__device__ int modulo(int a, int m){
	int r = a % m; // The remainder operator works differently for negative numbers (as we always want positive output)
	r = (r < 0) ? r + m : r; // We add `m` to the remainder `r` when `r` is negative, else do nothing
	return r;
}

/* 1.2 Implement the decryption kernel for a single block of threads with an `x` dimension of `N` (1024). 
The function should store the result in `d_output`. You can define the 
inverse modulus `A`, `B` and `M` using pre-processor definitions. */
__global__ void affine_decrypt(int *d_input, int *d_output)
{
	int i = threadIdx.x;
	d_output[i] = modulo(A * (d_input[i] - B), M);
}

/* 1.8 Complete the `affine_decrypt_multiblock` kernel to work using multiple blocks of threads. 
Change your grid and block dimensions so that you launch 8 blocks of 128 threads. Note: 8 * 128 = 1024. */
__global__ void affine_decrypt_multiblock(int *d_input, int *d_output)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	d_output[i] = modulo(A * (d_input[i] - B), M);
}

int main(int argc, char *argv[])
{
	int *h_input, *h_output;
	int *d_input, *d_output;
	unsigned int size;
	int i;

	size = N * sizeof(int); // Define the size of the data

	/* Allocate the host memory */
	h_input = (int *)malloc(size);
	h_output = (int *)malloc(size);

	/* 1.3 Allocate memory on the device for the input `d_input` and output `d_output` */
	cudaMalloc((void **)&d_input, size);
	cudaMalloc((void **)&d_output, size);
	checkCUDAError("Memory allocation");

	/* Read the encryted text from file to `h_input` */
	read_encrypted_file(h_input);

	/* 1.4 Copy the host input values in `h_input` to the device memory `d_input`. */
	cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
	checkCUDAError("Input transfer to device");

	/* Configure the grid of thread blocks and run the GPU kernel. */
	if (MULTIBLOCK == 0) {
		// 1.5 Configure a single block of `N` threads and launch the `affine_decrypt` kernel.
		dim3 blocksPerGrid(1, 1, 1);
		dim3 threadsPerBlock(N, 1, 1);
		affine_decrypt <<<blocksPerGrid, threadsPerBlock >>>(d_input, d_output);
	}
	else if (MULTIBLOCK == 1) {
		/* 1.8 Configure 8 blocks of 128 threads and launch the `affine_decrypt_multiblock` kernel */
		dim3 blocksPerGrid(8, 1, 1);
		dim3 threadsPerBlock(N/8, 1, 1); // Note: 8 * 128 = 1024.
		affine_decrypt_multiblock <<<blocksPerGrid, threadsPerBlock >>>(d_input, d_output);
	}

	/* Wait for all threads to complete */
	cudaThreadSynchronize();
	checkCUDAError("Kernel execution");

	/* 1.6 Copy the GPU output back to the host.
	Copy the device output values in `d_output` to the host memory `h_output`. */
	cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
	checkCUDAError("Result transfer to host");

	/* Print out the result to screen */
	for (i = 0; i < N; i++) {
		printf("%c", (char)h_output[i]);
	}
	printf("\n");

	/* 1.7: Free device memory */
	cudaFree(d_input);
	cudaFree(d_output);
	checkCUDAError("Free memory");

	/* Free host buffers */
	free(h_input);
	free(h_output);

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void read_encrypted_file(int* input)
{
	FILE *f = NULL;
	f = fopen("encrypted1.bin", "rb"); // Read-only and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find encrypted1.bin file \n");
		exit(1);
	}
	// Read the encrypted data
	fread(input, sizeof(unsigned int), N, f);
	fclose(f);
}
