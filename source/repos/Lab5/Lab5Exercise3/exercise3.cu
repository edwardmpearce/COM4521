/* Lab 5 Exercise 3 Program
In this program we experiment with using texture memory whilst blurring an image using a GPU. 
We will explicitly use texture binding rather than using qualifiers to force memory loads through the read-only cache. 
There are good reasons for doing this when dealing with problems which relate to images or with problems which 
decompose naturally to 2D layouts. Potential benefits include improved caching, address wrapping and filtering.
An image of a dog, `input.ppm`, is provided. Build and execute the code to see the result of executing the image blur kernel.
You can modify the macro `SAMPLE_SIZE` to increase the scale of the blur. */
/* GPU devices possess several different types of memory and caches, including:
Registers (very fast, but limited space), Thread-Local Global memory (very slow),
Shared memory (allows data to be shared between threads in the same block),
Constant memory (useful for broadcasting the same value to multiple threads within a half-warp or larger thread block),
L1 Cache/read-only memory (useful when multiple threads access the same piece of data),
texture memory (useful for normalized values and reading 2D image data). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_texture_types.h"
#include "texture_fetch_functions.hpp"

#define IMAGE_DIM 2048
#define SAMPLE_SIZE 6
#define NUMBER_OF_SAMPLES (((2*SAMPLE_SIZE)+1)*((2*SAMPLE_SIZE)+1))

// Takes input, multiply by pseudo-random `int` between `0` and `RAND_MAX`, then divide by `RAND_MAX`
#define rnd( x ) (x * rand() / RAND_MAX) // Approximates multiplication of `x` by a uniform random number from [0,1]
#define INF 2e10f // Represents an infinite value

void output_image_file(uchar4* image);
void input_image_file(char* filename, uchar4* image);
void checkCUDAError(const char *msg);

/* Device Code */
__global__ void image_blur(uchar4 *image, uchar4 *image_output) {
	// Map from thread position in grid to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int output_offset = y * blockDim.x * gridDim.x + x; // Linearized index into the output image buffer
	uchar4 pixel;
	float4 average = make_float4(0,0,0,0);

	// For each thread (x,y) iterate over the sample grid of neighbouring pixels (x,y) + [-SAMPLE_SIZE, SAMPLE_SIZE]^2
	for (int i = -SAMPLE_SIZE; i <= SAMPLE_SIZE; i++) {
		for (int j = -SAMPLE_SIZE; j <= SAMPLE_SIZE; j++) {
			// Calculate the position of an offset pixel within the sample grid around (x,y)
			int x_offset = x + i;
			int y_offset = y + j;
			// Wrap the boundaries of the image like a torus in case the sample grid leaves the image boundaries
			// At most one of each pair of conditional statements will hold for any offset pixel, and there is no interference
			if (x_offset < 0) {
				x_offset += IMAGE_DIM;
			}
			if (x_offset >= IMAGE_DIM) {
				x_offset -= IMAGE_DIM;
			}	
			if (y_offset < 0) {
				y_offset += IMAGE_DIM;
			}
			if (y_offset >= IMAGE_DIM) {
				y_offset -= IMAGE_DIM;
			}
			// Linearized index of the offset pixel used to read from the input image buffer
			int offset = y_offset * blockDim.x * gridDim.x + x_offset;
			pixel = image[offset];

			// Sum the rgb values over the sample grid `(x,y) + [-SAMPLE_SIZE, SAMPLE_SIZE]^2`
			average.x += pixel.x;
			average.y += pixel.y;
			average.z += pixel.z;
		}
	}
	// Calculate the average of the rgb values by dividing by `(2s+1)^2`
	average.x /= (float)NUMBER_OF_SAMPLES;
	average.y /= (float)NUMBER_OF_SAMPLES;
	average.z /= (float)NUMBER_OF_SAMPLES;
	// Cast the average rgb values from `float` to `unsigned char` and write them to `image_output` at the linear index for (x,y)
	image_output[output_offset].x = (unsigned char)average.x;
	image_output[output_offset].y = (unsigned char)average.y;
	image_output[output_offset].z = (unsigned char)average.z;
	image_output[output_offset].w = 255; // Leave the `a` value at full opacity (see "RGBA Colour Model")
}

/* 3.1 Create a copy of the `image_blur` kernel called `image_blur_texture1D`. 
Declare a 1-dimensional texture reference with `cudaReadModeElementType`. 
Modify the new kernel to perform a texture lookup using `tex1Dfetch`. 
Modify the host code to execute the `texture1D` version of the kernel after the first version 
saving the timing value to the `.y` component of the variable `ms`. 
You will need to add appropriate host code to bind and unbind the texture before and after the kernel execution, respectively. */

texture<uchar4, cudaTextureType1D, cudaReadModeElementType> sample1D;
// Texture references can only be declared as static global variables and cannot be passed as function/kernel arguments
__global__ void image_blur_texture1D(uchar4* image_output) {
	// Map from thread position in grid to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int output_offset = y * blockDim.x * gridDim.x + x; // Linearized index into the output image buffer
	uchar4 pixel;
	float4 average = make_float4(0, 0, 0, 0);

	// For each thread (x,y) iterate over the sample grid of neighbouring pixels (x,y) + [-SAMPLE_SIZE, SAMPLE_SIZE]^2
	for (int i = -SAMPLE_SIZE; i <= SAMPLE_SIZE; i++) {
		for (int j = -SAMPLE_SIZE; j <= SAMPLE_SIZE; j++) {
			// Calculate the position of an offset pixel within the sample grid around (x,y)
			int x_offset = x + i;
			int y_offset = y + j;
			// Wrap the boundaries of the image like a torus in case the sample grid leaves the image boundaries
			// At most one of each pair of conditional statements will hold for any offset pixel, and there is no interference
			if (x_offset < 0) {
				x_offset += IMAGE_DIM;
			}
			if (x_offset >= IMAGE_DIM) {
				x_offset -= IMAGE_DIM;
			}
			if (y_offset < 0) {
				y_offset += IMAGE_DIM;
			}
			if (y_offset >= IMAGE_DIM) {
				y_offset -= IMAGE_DIM;
			}
			int offset = y_offset * blockDim.x * gridDim.x + x_offset;
			// Linearized index of the offset pixel used to read from texture memory
			pixel = tex1Dfetch(sample1D, offset);

			// Sum the rgb values over the sample grid `(x,y) + [-SAMPLE_SIZE, SAMPLE_SIZE]^2`
			average.x += pixel.x;
			average.y += pixel.y;
			average.z += pixel.z;
		}
	}
	// Calculate the average of the rgb values by dividing by `(2s+1)^2`
	average.x /= (float)NUMBER_OF_SAMPLES;
	average.y /= (float)NUMBER_OF_SAMPLES;
	average.z /= (float)NUMBER_OF_SAMPLES;
	// Cast the average rgb values from `float` to `unsigned char` and write them to `image_output` at the linear index for (x,y)
	image_output[output_offset].x = (unsigned char)average.x;
	image_output[output_offset].y = (unsigned char)average.y;
	image_output[output_offset].z = (unsigned char)average.z;
	image_output[output_offset].w = 255; // Leave the `a` value at full opacity (see "RGBA Colour Model")
}

/* 3.2 Create a copy of the `image_blur` kernel called `image_blur_texture2D`.
Declare a 2-dimensional texture reference with `cudaReadModeElementType`.
Modify the new kernel to perform a texture lookup using `tex2D`. 
Modify the host code to execute the `texture2D` version of the kernel after the `texture1D` version
saving the timing value to the `.z` component of the variable `ms`. 
You will need to add appropriate host code to bind and unbind the texture before and after the kernel execution, respectively. */

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> sample2D;
// Texture references can only be declared as static global variables and cannot be passed as function/kernel arguments
__global__ void image_blur_texture2D(uchar4* image_output) {
	// Map from thread position in grid to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int output_offset = y * blockDim.x * gridDim.x + x; // Linearized index into the output image buffer
	uchar4 pixel;
	float4 average = make_float4(0, 0, 0, 0);

	// For each thread (x,y) iterate over the sample grid of neighbouring pixels (x,y) + [-SAMPLE_SIZE, SAMPLE_SIZE]^2
	for (int i = -SAMPLE_SIZE; i <= SAMPLE_SIZE; i++) {
		for (int j = -SAMPLE_SIZE; j <= SAMPLE_SIZE; j++) {
			/* 3.3 In the case of the 2D texture version it is possible to perform wrapping of the index values without
			explicitly checking the values of `x_offset`, `y_offset`. To do this remove the checks from your kernel, 
			and set the structure members `addressMode[0]`, `addressMode[1]` of your 2D texture reference
			to `cudaAddressModeWrap` in the `main` function. */
			// Calculate the position of an offset pixel within the sample grid around (x,y)
			int x_offset = x + i;
			int y_offset = y + j;
			// For 2D texture lookup, we don't need to calculate linearized indices
			pixel = tex2D(sample2D, x_offset, y_offset);
			
			// Sum the rgb values over the sample grid `(x,y) + [-SAMPLE_SIZE, SAMPLE_SIZE]^2`
			average.x += pixel.x;
			average.y += pixel.y;
			average.z += pixel.z;
		}
	}
	// Calculate the average of the rgb values by dividing by `(2s+1)^2`
	average.x /= (float)NUMBER_OF_SAMPLES;
	average.y /= (float)NUMBER_OF_SAMPLES;
	average.z /= (float)NUMBER_OF_SAMPLES;
	// Cast the average rgb values from `float` to `unsigned char` and write them to `image_output` at the linear index for (x,y)
	image_output[output_offset].x = (unsigned char)average.x;
	image_output[output_offset].y = (unsigned char)average.y;
	image_output[output_offset].z = (unsigned char)average.z;
	image_output[output_offset].w = 255; // Leave the `a` value at full opacity (see "RGBA Colour Model")
}

/* Host Code */
int main(void) {
	unsigned int image_size; // Memory requirement for image data
	uchar4 *d_image, *d_image_output; // Pointer variables for image input and output on device
	uchar4 *h_image; // Pointer variable for image on host
	cudaEvent_t start, stop; // CUDA event timestamps
	float3 ms; // 3-tuple of timing data where .x = normal, .y = tex1D, .z = tex2D
	image_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar4); // Calculate memory requirements

	// Create GPU event timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate memory on the GPU for the image input and output
	cudaMalloc((void**)&d_image, image_size);
	cudaMalloc((void**)&d_image_output, image_size);
	checkCUDAError("CUDA Malloc");

	// Allocate and load host image
	h_image = (uchar4*)malloc(image_size);
	input_image_file("input.ppm", h_image);

	// Copy input image from host to device memory
	cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA Memcpy Host to Device");

	// CUDA grid layout
	dim3 blocksPerGrid(IMAGE_DIM / 16, IMAGE_DIM / 16);
	dim3 threadsPerBlock(16, 16);

	// Normal version
	cudaEventRecord(start, 0);
	image_blur << <blocksPerGrid, threadsPerBlock >> >(d_image, d_image_output);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms.x, start, stop);
	checkCUDAError("Kernel (normal)");

	/* 3.1 Execute the `texture1D` version of the kernel after the normal version, saving the timing value to the `ms.y`.
	You will need to bind and unbind the texture before and after the kernel execution, respectively. */
	// Bind the texture reference `sample1D` declared earlier to the memory buffer for the input image `d_image`
	cudaBindTexture(0, sample1D, d_image, image_size);
	checkCUDAError("Bind cudaTextureType1D");
	cudaEventRecord(start, 0);
	image_blur_texture1D << <blocksPerGrid, threadsPerBlock >> > (d_image_output);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms.y, start, stop);
	cudaUnbindTexture(sample1D);
	checkCUDAError("Kernel (tex1D)");

	/* 3.2 Execute the `texture2D` kernel after the `texture1D` version, saving the timing value to the `ms.z`.
	You will need to bind and unbind the texture before and after the kernel execution, respectively. 
	Moreover, the CUDA runtime requires that we provide a `cudaChannelFormatDesc` when we bind 2D textures */
	/* 3.3 When using 2D textures we can wrap index values around without explicitly checking them. 
	To do this remove the checks from your kernel, and set the structure members `addressMode[0]`, `addressMode[1]` 
	of your 2D texture reference `sample2D` to `cudaAddressModeWrap` in the `main` function before binding. */
	// Results in wrapping the image boundaries around like a torus when we access outside the boundaries
	sample2D.addressMode[0] = cudaAddressModeWrap;
	sample2D.addressMode[1] = cudaAddressModeWrap;

	// Declare a channel format descriptor called `desc` with data type `uchar4`
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
	// Bind the formatted texture reference `sample2D` to the memory buffer for the input image `d_image`
	cudaBindTexture2D(0, sample2D, d_image, desc, IMAGE_DIM, IMAGE_DIM, IMAGE_DIM * sizeof(uchar4));
	checkCUDAError("Bind cudaTextureType2D");
	cudaEventRecord(start, 0);
	image_blur_texture2D << <blocksPerGrid, threadsPerBlock >> > (d_image_output);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms.z, start, stop);
	checkCUDAError("Kernel (tex2D)");
	cudaUnbindTexture(sample2D);

	// Copy the blurred output image from device to host for output to file
	cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA Memcpy Device to Host");

	// Output timings
	printf("Execution times:\n");
	printf("\tNormal version: %f\n", ms.x); // 16.917631ms
	printf("\ttex1D version: %f\n", ms.y);  // 10.342144ms
	printf("\ttex2D version: %f\n", ms.z);  // 10.314624ms

	// Output image to file `output.ppm`
	output_image_file(h_image);

	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_image);
	cudaFree(d_image_output);
	free(h_image);

	return 0;
}

void output_image_file(uchar4* image) {
	FILE *f; // Output file handle
	// Open the output file and write the header info for the `.ppm` filetype
	f = fopen("output.ppm", "wb");
	if (f == NULL) {
		fprintf(stderr, "Error opening 'output.ppm' output file\n");
		exit(1);
	}
	fprintf(f, "P6\n");
	fprintf(f, "# COM4521 Lab 05 Exercise03\n");
	fprintf(f, "%d %d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);

	for (int y = 0; y < IMAGE_DIM; y++) {
		for (int x = 0; x < IMAGE_DIM; x++) {
			int i = y * IMAGE_DIM + x;
			fwrite(&image[i], sizeof(unsigned char), 3, f); // Only write rgb values (ignoring a) for each pixel
		}
	}
	fclose(f);
}

void input_image_file(char* filename, uchar4* image) {
	FILE *f; // Input file handle
	char temp[256];
	unsigned int x, y, s;

	// Open the input file and write the header info for the `.ppm` filetype
	// See http://netpbm.sourceforge.net/doc/ppm.html for the PPM file specification
	// See also https://en.wikipedia.org/wiki/Netpbm for further information and history
	f = fopen(filename, "rb");
	if (f == NULL){
		fprintf(stderr, "Error opening '%s' input file\n", filename);
		exit(1);
	}
	fscanf(f, "%s\n", &temp); // Read the first line of the file to a temporary buffer
	fscanf(f, "%d %d\n", &x, &y); // Read the image width and height to variables `x` and `y`, respectively.
	fscanf(f, "%d\n",&s); // Read the maximum colour value setting to `s`
	if ((x != IMAGE_DIM) || (y != IMAGE_DIM)) {
		fprintf(stderr, "Error: Input image file has wrong fixed dimensions\n");
		exit(1);
	}

	for (int y = 0; y < IMAGE_DIM; y++) {
		for (int x = 0; x < IMAGE_DIM; x++) {
			int i = y * IMAGE_DIM + x;
			fread(&image[i], sizeof(unsigned char), 3, f); // Only read rgb values (ignoring a) for each pixel
			//image[i].w = 255; // Otherwise could explicitly set `a` value to full opacity
		}
	}
	fclose(f);
}

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
