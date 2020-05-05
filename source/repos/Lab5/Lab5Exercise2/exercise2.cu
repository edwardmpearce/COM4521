/* Lab 5 Exercise 2 Program
Source code for this lab class is modifed from the book CUDA by Example and provided with permission from NVIDIA Corporation.
See https://developer.nvidia.com/cuda-example for the original source code for the book examples. */
/* For this exercise we are going to optimise a simple ray tracer application by changing the memory types which are used.
The ray tracer is a simple ray casting algorithm. For each pixel we cast a ray into a scene consisting of sphere objects. 
The ray checks for intersections with the spheres, and where there is an intersection a colour value for the pixel 
is generated based on the intersection position of the ray on the sphere (giving an impression of forward facing lighting). 
For more information on the ray tracing technique read Chapter 6 of the "CUDA by Example" book which this exercise is based on.*/
/* GPU devices possess several different types of memory and caches, including:
Registers (very fast, but limited space), Thread-Local Global memory (very slow), 
Shared memory (allows data to be shared between threads in the same block), 
Constant memory (useful for broadcasting the same value to multiple threads within a half-warp or larger thread block), 
L1 Cache/read-only memory (useful when multiple threads access the same piece of data), 
texture memory (useful for normalized values and reading 2D image data). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define IMAGE_DIM 2048
/* 2.3 How does the performance compare between using global, read-only, and constant memory? 
Is this what you expected and why? Modify your code to repeat the experiment whilst changing
the number of spheres in the scene to 2^i for i = 4,5,6,7,8,9,10. Do this automatically so that you loop over 
all these sphere count sizes and record the timing results in a 2D array. */
#define MIN_SPHERES 16
#define MAX_SPHERES 2048

// Takes input, multiply by pseudo-random `int` between `0` and `RAND_MAX`, then divide by `RAND_MAX`
#define rnd( x ) (x * rand() / RAND_MAX) // Approximates multiplication of `x` by a uniform random number from [0,1]
#define INF 2e10f // Represents an infinite value

void output_image_file(uchar4* image);
void checkCUDAError(const char *msg);

// Coloured spheres are modelled with a data structure that stores a sphere's center coordinates, radius, and colour.
struct Sphere {
	float x, y, z;
	float radius;
	float r, g, b;
};

/* Device Code */
__constant__ unsigned int d_sphere_count; // Here the constant modifier means the value doesn't change during kernel calls

/* Given a vertical ray shot downwards from the pixel at `(ox, oy)`, the kernel below computes whether the ray 
intersects the sphere `s`, in which case we return the `z` coordinate of the intersection point. This is so that
in the event that a ray hits more than one sphere, only the closest sphere (highest intersection point) can actually be seen. 
We also record the cosine of the polar angle between the intersection point and the center of `s` using the `n` pointer.
If the given ray does not intersection the given sphere `s`, we return a large negative value. */
__device__ float sphere_intersect(Sphere *s, float ox, float oy, float *n) {
	float dx = ox - s->x;
	float dy = oy - s->y;
	float radius = s->radius;
	if (dx*dx + dy*dy < radius*radius) {
		float dz = sqrtf(radius*radius - dx*dx - dy*dy); // Height of intersection point relative to sphere center
		*n = dz / radius; // Cosine of polar angle between intersection point and (positive vertical through) sphere center
		return dz + s->z; // Height of intersection point relative to origin
	}
	return -INF;
}

/* 2.0 Try executing the starting code and examining the output image `output.ppm` using GIMP or Adobe Photoshop.
The initial code places the spheres in GPU global memory. We know that there are two options for improving this 
in the form of constant memory and texture/read only memory. */
__global__ void ray_trace(uchar4 *image, Sphere *d_spheres) {
	// Map from thread position in grid to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int array_idx = y * blockDim.x * gridDim.x + x; // Linearized index into the output image buffer
	// Scale observation coordinates between [-IMAGE_DIM/2, IMAGE_DIM/2] so that z-axis runs through the center of the image
	float ox = (x - IMAGE_DIM / 2.0f);
	float oy = (y - IMAGE_DIM / 2.0f);

	float r = 0, g = 0, b = 0; // Initialize the pixel colours (background colour when no spheres detected by a pixel)
	float maxz = -INF; // Initialize the z-coordinate of the highest intersection point to a large negative value
	for (int i = 0; i < d_sphere_count; i++) {
		// Iterate over the input sphere data
		Sphere *s = &d_spheres[i];
		float n;
		float t = sphere_intersect(s, ox, oy, &n);
		// If the ray from `(ox, oy)` intersects `s` at a point higher than previously seen, update pixel colours and `maxz`
		if (t > maxz) {
			/* We calculate the pixel colour based on the sphere's colour and the position of the intersection point on `s`.
			Scaling by the cosine of the polar angle gives an impression of forward facing lighting */
			float fscale = n;
			r = s->r * fscale;
			g = s->g * fscale;
			b = s->b * fscale;
			maxz = t;
		}
	}
	// Scale the rgb values from floats in [0,1] to integer values in [0,255], with full opacity
	image[array_idx].x = (int)(r * 255);
	image[array_idx].y = (int)(g * 255);
	image[array_idx].z = (int)(b * 255);
	image[array_idx].w = 255;
}

/* 2.1 Create a modified version of ray tracing kernel, called `ray_trace_read_only`, which uses the read-only data cache. 
You should implement this by using the `const` and `__restrict__` qualifiers. 
Calculate the execution time of the new version alongside the old version so that they can be directly compared: 
You will also need to create a modified version of the sphere intersect function, called `sphere_intersect_read_only`. */
__device__ float sphere_intersect_read_only(Sphere const * __restrict__ s, float ox, float oy, float * n) {
	float dx = ox - s->x;
	float dy = oy - s->y;
	float radius = s->radius;
	if (dx * dx + dy * dy < radius * radius) {
		float dz = sqrtf(radius * radius - dx * dx - dy * dy); // Height of intersection point relative to sphere center
		*n = dz / radius; // Cosine of polar angle between intersection point and (positive vertical through) sphere center
		return dz + s->z; // Height of intersection point relative to origin
	}
	return -INF;
}

__global__ void ray_trace_read_only(uchar4* image, Sphere const * __restrict__ d_spheres) {
	// Map from thread position in grid to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int array_idx = y * blockDim.x * gridDim.x + x; // Linearized index into the output image buffer
	// Scale observation coordinates between [-IMAGE_DIM/2, IMAGE_DIM/2] so that z-axis runs through the center of the image
	float ox = (x - IMAGE_DIM / 2.0f);
	float oy = (y - IMAGE_DIM / 2.0f);

	float r = 0, g = 0, b = 0; // Initialize the pixel colours (background colour when no spheres detected by a pixel)
	float maxz = -INF; // Initialize the z-coordinate of the highest intersection point to a large negative value
	for (int i = 0; i < d_sphere_count; i++) { // Iterate over the input sphere data
		// Load sphere data into read-only cache since we will read multiple times without modification
		Sphere const * __restrict__ s = &d_spheres[i];
		float n;
		float t = sphere_intersect_read_only(s, ox, oy, &n);
		// If the ray from `(ox, oy)` intersects `s` at a point higher than previously seen, update pixel colours and `maxz`
		if (t > maxz) {
			/* We calculate the pixel colour based on the sphere's colour and the position of the intersection point on `s`.
			Scaling by the cosine of the polar angle gives an impression of forward facing lighting */
			float fscale = n;
			r = s->r * fscale;
			g = s->g * fscale;
			b = s->b * fscale;
			maxz = t;
		}
	}
	// Scale the rgb values from floats in [0,1] to integer values in [0,255], with full opacity
	image[array_idx].x = (int)(r * 255);
	image[array_idx].y = (int)(g * 255);
	image[array_idx].z = (int)(b * 255);
	image[array_idx].w = 255;
}

/* 2.2 Create a modified version of ray tracing kernel, called `ray_trace_const`, which uses the constant data cache.
Calculate the execution time of the new version alongside the two other versions so that they can be directly compared. */
// Statically declare the array of sphere data that the new kernel will use - doesn't need to be passed as kernel input
// We must also use `cudaMemcpyToSymbol` for copying data from host into constant cache (rather than global memory) in `main`.
__constant__ Sphere d_spheres_const[MAX_SPHERES];

__global__ void ray_trace_const(uchar4* image) {
	// Map from thread position in grid to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int array_idx = y * blockDim.x * gridDim.x + x; // Linearized index into the output image buffer
	// Scale observation coordinates between [-IMAGE_DIM/2, IMAGE_DIM/2] so that z-axis runs through the center of the image
	float ox = (x - IMAGE_DIM / 2.0f);
	float oy = (y - IMAGE_DIM / 2.0f);

	float r = 0, g = 0, b = 0; // Initialize the pixel colours (background colour when no spheres detected by a pixel)
	float maxz = -INF; // Initialize the z-coordinate of the highest intersection point to a large negative value
	for (int i = 0; i < d_sphere_count; i++) {
		// Iterate over the input sphere data
		Sphere* s = &d_spheres_const[i];
		float n;
		float t = sphere_intersect(s, ox, oy, &n);
		// If the ray from `(ox, oy)` intersects `s` at a point higher than previously seen, update pixel colours and `maxz`
		if (t > maxz) {
			/* We calculate the pixel colour based on the sphere's colour and the position of the intersection point on `s`.
			Scaling by the cosine of the polar angle gives an impression of forward facing lighting */
			float fscale = n;
			r = s->r * fscale;
			g = s->g * fscale;
			b = s->b * fscale;
			maxz = t;
		}
	}
	// Scale the rgb values from floats in [0,1] to integer values in [0,255], with full opacity
	image[array_idx].x = (int)(r * 255);
	image[array_idx].y = (int)(g * 255);
	image[array_idx].z = (int)(b * 255);
	image[array_idx].w = 255;
}

/* Host code */
int main(void) {
	unsigned int image_size, spheres_size; // Memory requirements for image and sphere data
	uchar4 *d_image; // Device image
	uchar4 *h_image; // Host image
	cudaEvent_t start, stop; // CUDA event timestamps
	Sphere h_spheres[MAX_SPHERES]; // Array of sphere data on host
	Sphere *d_spheres; // Sphere data on device
	float3 timing_data[7]; // 3-tuples of timing data where .x = normal, .y = read-only, .z = const, other index for num_spheres

	// Calculate memory requirements
	image_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar4);
	spheres_size = sizeof(Sphere)*MAX_SPHERES;

	// Create GPU event timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate memory on the GPU for the output image and sphere data
	cudaMalloc((void**)&d_image, image_size);
	cudaMalloc((void**)&d_spheres, spheres_size);
	checkCUDAError("CUDA Malloc");

	// Generate random sphere data on the host CPU
	for (int i = 0; i < MAX_SPHERES; i++) {
		// `float` colour values (pseudo-)uniformly distributed between [0,1]
		h_spheres[i].r = rnd(1.0f);
		h_spheres[i].g = rnd(1.0f);
		h_spheres[i].b = rnd(1.0f);
		// Sphere center coordinates (pseudo-)uniformly distributed between [-IMAGE_DIM/2, IMAGE_DIM/2]
		h_spheres[i].x = rnd((float)IMAGE_DIM) - (IMAGE_DIM / 2.0f);
		h_spheres[i].y = rnd((float)IMAGE_DIM) - (IMAGE_DIM / 2.0f);
		h_spheres[i].z = rnd((float)IMAGE_DIM) - (IMAGE_DIM / 2.0f);
		// Sphere radii (pseudo-)uniformly distributed between [20, 120]
		h_spheres[i].radius = rnd(100.0f) + 20;
	}
	// Copy sphere data from host to device memory
	cudaMemcpy(d_spheres, h_spheres, spheres_size, cudaMemcpyHostToDevice);
	// Copy sphere data from host to constant memory on device
	cudaMemcpyToSymbol(d_spheres_const, h_spheres, spheres_size);
	checkCUDAError("CUDA Memcpy Host to Device");

	// Allocate memory for image on host 
	h_image = (uchar4*)malloc(image_size);

	// CUDA grid layout
	dim3 blocksPerGrid(IMAGE_DIM / 16, IMAGE_DIM / 16);
	dim3 threadsPerBlock(16, 16);

	/* 2.3 How does the performance compare between using global, read-only, and constant memory?
	Is this what you expected and why? Modify your code to repeat the experiment whilst changing
	the number of spheres in the scene to 2^i for i = 4,5,6,7,8,9,10. Do this automatically so that you loop over
	all these sphere count sizes and record the timing results in a 2D array. */
	for (int i = 0; i < 7; i++) {
		/* For further reference on bitwise operators in C, please see the following links:
		https://www.geeksforgeeks.org/left-shift-right-shift-operators-c-cpp/
		https://en.wikipedia.org/wiki/Bitwise_operations_in_C
		https://docs.microsoft.com/en-us/cpp/cpp/left-shift-and-right-shift-operators-input-and-output?view=vs-2019
		https://www.tutorialspoint.com/cprogramming/c_bitwise_operators.htm
		https://stackoverflow.com/questions/141525/what-are-bitwise-shift-bit-shift-operators-and-how-do-they-work 
		We use the left bitwise operator to quickly multiply `MIN_SPHERES` by `2^i`.
		This gives the desired range from 16 to 2048 increasing in powers of 2. */
		unsigned int sphere_count = MIN_SPHERES << i;
		cudaMemcpyToSymbol(d_sphere_count, &sphere_count, sizeof(unsigned int));
		checkCUDAError("CUDA copy sphere count to device");

		// Generate an image from the sphere data, and time the kernel execution (normal/gloabl memory version)
		cudaEventRecord(start, 0);
		ray_trace << <blocksPerGrid, threadsPerBlock >> > (d_image, d_spheres);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timing_data[i].x, start, stop);
		checkCUDAError("Kernel (normal)");

		// Generate an image from the sphere data, using the read-only cache, and time the kernel execution
		// Note that this will overwrite the device image data from the previous kernel
		cudaEventRecord(start, 0);
		ray_trace_read_only << <blocksPerGrid, threadsPerBlock >> > (d_image, d_spheres);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timing_data[i].y, start, stop);
		checkCUDAError("Kernel (read-only)");

		// Generate an image from the sphere data, using the constant memory cache, and time the kernel execution
		// Note that this will overwrite the device image data from the previous kernel
		cudaEventRecord(start, 0);
		ray_trace_const << <blocksPerGrid, threadsPerBlock >> > (d_image);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timing_data[i].z, start, stop);
		checkCUDAError("Kernel (constant cache)");
	}

	// Copy the image back from the GPU for output to file
	cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA Memcpy Device to Host");

	// Tabulate the execution times for the three different kernels producing ray-traced images of spheres in a scene
	// where the number of spheres varies from 16, 32, 64, 128, 256, 512, 1024, 2048.
	printf("Timing Data Table\n Spheres | Normal | Read-only | Const\n");
	for (int i = 0; i < 7; i++) {
		int sphere_count = MIN_SPHERES << i; // Left bitwise shift operator effectively returns `MIN_SPHERES * 2^i`
		printf(" %-7i | %-6.3f | %-9.3f | %.3f\n", sphere_count, timing_data[i].x, timing_data[i].y, timing_data[i].z);
	}

	// Output the image to file `output.ppm`
	output_image_file(h_image);

	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_image);
	cudaFree(d_spheres);
	free(h_image);

	return 0;
}

/* See https://en.wikipedia.org/wiki/RGBA_color_model for more information on the "Red Green Blue Alpha colour model"
Each pixel has 4 unsigned characters (4 bytes) of associated data, but in our application the alpha value,
which indicates opacity/transparency, is omitted. */
void output_image_file(uchar4* image) {
	FILE *f; // Output file handle
	// Open the output file and write the header info for the `.ppm` filetype
	f = fopen("output.ppm", "wb");
	if (f == NULL) {
		fprintf(stderr, "Error opening 'output.ppm' output file\n");
		exit(1);
	}
	fprintf(f, "P6\n");
	fprintf(f, "# COM4521 Lab 05 Exercise02\n");
	fprintf(f, "%d %d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);
	for (int x = 0; x < IMAGE_DIM; x++) {
		for (int y = 0; y < IMAGE_DIM; y++) {
			int i = x + y * IMAGE_DIM;
			fwrite(&image[i], sizeof(unsigned char), 3, f); // Only write rgb values (ignoring a) for each pixel
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
