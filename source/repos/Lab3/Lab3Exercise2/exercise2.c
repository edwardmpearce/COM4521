/* Lab 3 Exercise 2 Program
We are going to parallelise an implementation of a Mandelbrot set calculation. 
The Mandelbrot set is the set of complex numbers `c` for which the function `f_{c}(z) = z^{2} + c` does not diverge 
when iterated from `z = 0`, i.e., for which the sequence f_{c}(0), f_{c}(f_{c}(0)), etc., remains bounded in absolute value.
We test whether the sequence leaves the predetermined bounded neighborhood of `0` within a predetermined number of iterations
The neighbourhood is determined by `ESCAPE_RADIUS_SQ` defined in `mandelbrot.h` and the number of iterations by `MAX_ITERATIONS`
Find out more about the Mandelbrot set at https://en.wikipedia.org/wiki/Mandelbrot_set */
/* 2.1 Start by parallelising the outer loop over the pixels in stage 1.
Ensure that you scope variables correctly using the private and shared clauses.
Test your code by comparing the result with the serial image.
Next parallelise the outer loop over the pixels in stage 2. Test your code again by comparing images with the serial version.
You should observe a speed up of the code. Try performing a minimum of 1000 iterations to ensure the speed up is measurable. */
/* After verifying that the correct output is produced after each modification of the code, we record performance results below:
TRANSFER_FUNCTION = ESCAPE_VELOCITY; MAX_ITERATIONS = 1000;
| Machine | Optimisation                 | Execution time(s) |
| :-----: | :--------------------------: | :---------------: |
| Laptop  | Fully Serial                 | 1.09s - 1.14s |
| Laptop  | Stage1 Parallel Outer Loop   | 0.44s - 0.53s |
| Laptop  | Stage1 Parallel Inner Loop   | 0.71s - 0.82s |
| Laptop  | Stage1 Nested Parallel Loops | 0.38s - 0.48s |
| Laptop  | Stage1 Nested, Stage2 Outer  | 0.36s - 0.46s |
| Library | Fully Serial                 | 0.63s  |
| Library | Stage1 Parallel Outer Loop   | 0.12s  |
| Library | Stage1 Parallel Inner Loop   | 0.17s  |
| Library | Stage1 Nested Parallel Loops | 0.091s |
| Library | Stage1 Nested, Stage2 Outer  | 0.08s  | */
/* 2.2 We now compare the performance of various methods for incrementing the histogram frequency counters whilst
avoiding race conditions. After each modification, we check outputs are consistent.
/* After verifying that the correct output is produced after each modification of the code, we record performance results below:
TRANSFER_FUNCTION = HISTOGRAM_ESCAPE_VELOCITY; MAX_ITERATIONS = 100; Stage 1 Parallel outer loop (over `y`) only
| Machine | HISTOGRAM_METHOD       | Description                    | Execution time(s) |
| :-----: | :--------------------: | :----------------------------: | :---------------: |
| Laptop  | SERIAL                 | Fully Serial                   | 0.170s - 0.192s |
| Laptop  | CRITICAL_SECTION       | Critical region                | 0.121s - 0.126s |
| Laptop  | LOCAL_HIST_AND_COMBINE | Barrier/master for aggregation | 0.0757s - 0.115s |
| Laptop  | OMP_ATOMIC             | Atomic operator directive      | 0.0780s - 0.131s |
| Library | SERIAL                 | Fully Serial                   | 0.103533s |
| Library | CRITICAL_SECTION       | Critical region                | 0.372621s |
| Library | OMP_ATOMIC             | Atomic operator directive      | 0.061277s | */
/* 2.3 Our Mandelbrot image generator is now parallelised and normalised but shows clear colour banding
as escape times are integer valued. Modify your code so that `tf` is equal to `HISTOGRAM_NORMALISED_ITERATION_COUNT`.
This will calculate an approximation of the fractional part of the escape time, which is used in the `h_nic_transfer` function
to perform a linear interpolation and give smooth shading between the bands.
Ensure that the variable `mu` is correctly scoped and your existing OpenMP pragma will work correctly.
Change `MAX_ITERATIONS` to `1000`. We are now going to experiment with different scheduling
approaches for parallelisation of stage 1. */
/* The table below records performance results for different parallel thread workload scheduling
TRANSFER_FUNCTION = HISTOGRAM_NORMALISED_ITERATION_COUNT; MAX_ITERATIONS = 1000;
HISTOGRAM_METHOD = OMP_ATOMIC; num_threads = 4 (laptop)
| Scheduling Method | Chunk Size | Execution time(s) |
| :---------------: | :--------: | :---------------: |
| Default (Static)  | Default (HEIGHT/num_threads) | 0.483s - 0.521s |
| Static            | HEIGHT     | 1.21s - 1.25s     | // This is serial with extra overhead since all work given to one thread
| Static            | 1          | 0.378s - 0.451s   |
| Static            | 2          | 0.377s - 0.444s   |
| Static            | 4          | 0.368s - 0.432s   |
| Static            | 8          | 0.380s - 0.478s   |
| Dynamic           | 1          | 0.357s - 0.464s   |
| Dynamic           | 2          | 0.357s - 0.414s   |
| Dynamic           | 4          | 0.355s - 0.404s   |
| Dynamic           | 8          | 0.347s - 0.415s   |
| Guided            | Default    | 0.366s - 0.449s   |
It appears that dynamic scheduling is quickest approach to the task. This is because there is uneven workload amongst threads.
This follows from the fact that more computation is required for image rows (`y` values) containing more of the Mandelbrot set
as these require `MAX_ITERATIONS`, whilst pixels on distant rows will escape the threshold region in fewer iterations */
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
/* To enable OpenMP support in your project you will need to include the OpenMP header file
`omp.h` and enable the compiler to use the OpenMP runtime. 
Set 'OpenMP Support' to 'Yes' (for both `Debug` and `Release` builds) in Project->Properties->C/C++->Language.
Add `_CRT_SECURE_NO_WARNINGS` to 'Preprocessor Definitions' in Project->Properties->C/C++->Preprocessor. */
#include <omp.h>
#include "mandelbrot.h"

// Image size
#define WIDTH 1024
#define HEIGHT 768

#define MAX_ITERATIONS 100 // Maximum number of iterations of `f_{c}(z) = z^{2} + c` to calculate

// C parameters (modify these to change the zoom and position of the Mandelbrot set image)
#define ZOOM 1.0
#define X_DISPLACEMENT -0.5
#define Y_DISPLACEMENT 0.0

static int iterations[HEIGHT][WIDTH];		// Store the escape time (iteration count) as an `int`
static double iterations_d[HEIGHT][WIDTH];	// Store the normalised escape time as a `double` for NIC method
/* Array to hold histogram (frequencies) of escape time data for possible escape time values from `0` to `MAX_ITERATIONS`
Local histograms for each image row (indexed by `y` from `0` to `HEIGHT - 1`) are used in exercise 2.2.2 and selected by 
setting `hist_method` to `LOCAL_HIST_AND_COMBINE`. Note that `HEIGHT` is the maximum possible number of threads 
that could be initialised, as it the maximum size of the number of work units (i.e. the width of the parallel loop) */
static int histogram[MAX_ITERATIONS + 1];	
static int local_histogram[HEIGHT][MAX_ITERATIONS + 1];	
static rgb rgb_output[HEIGHT][WIDTH];			// Output data
static rgb rand_banding[MAX_ITERATIONS + 1];	// Random colour banding

/* 2.2, 2.3 Change the transfer function by setting the global variable `tf` */
const TRANSFER_FUNCTION tf = RANDOM_NORMALISED_ITERATION_COUNT;
// Set the method to increment the histogram counter whilst avoiding race conditions (for histogram transfer functions)
const HISTOGRAM_METHOD hist_method = OMP_ATOMIC;

int main(int argc, char *argv[]) {
	int i, x, y;							// Loop counters. `x` and `y` denote pixel coordinates.
	double c_re, c_im;						// Real and imaginary part of the parameter `c`
	double z_re, z_im, temp_re, temp_im;	// Real and imaginary parts of `z_{i}` and `z_{i+1}`
	double mu;								// For normalised escape iteration count with fractional component
	double begin, end;						// Timestamp variables
	double elapsed;							// Elapsed time
	FILE *f;								// Output file handle

	// Open the output file and write header info for PPM filetype
	f = fopen("output.ppm", "wb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'output.ppm' output file\n");
		exit(1);
	}
	fprintf(f, "P6\n");
	fprintf(f, "# COM4521 Lab 03 Exercise02\n");
	fprintf(f, "%d %d\n%d\n", WIDTH, HEIGHT, 255);

	int max_threads = omp_get_max_threads();
	printf("OpenMP using %d threads\n", max_threads);

	// Start timer
	begin = omp_get_wtime();

	// Initialise all data values stored in the histogram to `0`
	/* See the following links for more details on the syntax and usage of the `memset` function (requires `#include <string.h>`)
	https://www.tutorialspoint.com/c_standard_library/c_function_memset.htm
	https://www.geeksforgeeks.org/memset-c-example/
	https://www.includehelp.com/c-programs/memset-function-in-c-with-example.aspx */
	memset(histogram, 0, sizeof(histogram));
	if (hist_method == LOCAL_HIST_AND_COMBINE) {
		memset(local_histogram, 0, sizeof(local_histogram));
	}

	// STAGE 1) Calculate the escape time (iteration count) for each pixel
	//omp_set_nested(1);
	#pragma omp parallel for default(none) private(y, x, z_re, z_im, temp_re, temp_im, c_re, c_im, i, mu) shared(tf, iterations_d, iterations, histogram) if (hist_method != SERIAL) schedule(dynamic, 4)
	for (y = 0; y < HEIGHT; y++) {
		// Can treat `y` as a shared variable on the inner loop since we read without changing within each outer loop iteration
		// Otherwise could use `firstprivate(y)` declaration to pass in the value to each thread
		//#pragma omp parallel for default(none) private(x, z_re, z_im, temp_re, temp_im, c_re, c_im, i, mu) shared(y, tf, iterations_d, iterations, histogram)
		for (x = 0; x < WIDTH; x++) {
			// Zero complex number values (for the initial value `z = 0`)
			z_re = 0;
			z_im = 0;

			// Sample the parameter `c` across the `HEIGHT` and `WIDTH` of the image, accounting for `ZOOM` and `DISPLACEMENT`
			c_re = ((double) x - (WIDTH / 2)) * 1.5 / (0.5 * ZOOM * WIDTH) + X_DISPLACEMENT;
			c_im = ((double) y - (HEIGHT / 2)) / (0.5 * ZOOM * HEIGHT) + Y_DISPLACEMENT;

			// Iterate whilst within the predetermined escape threshold, up to at most `MAX_ITERATIONS`
			// Upon exiting this loop, `i` will count the number of iterations to escape the threshold
			// or equal `MAX_ITERATIONS`, in which case we judge `c` to lie in the Mandelbrot set
			for (i = 0; (i < MAX_ITERATIONS) && ((z_re * z_re + z_im * z_im) < ESCAPE_RADIUS_SQ); i++) {
				// Store current values
				temp_re = z_re;
				temp_im = z_im;

				// Calculate the next value in the sequence according to the rule `z_{i+1} = z_{i}^{2} + c`
				z_re = temp_re * temp_re - temp_im * temp_im + c_re;
				z_im = 2.0 * temp_re * temp_im + c_im;
			}

			// Algorithm to count iterations until escape for `NORMALISED_ITERATION_COUNT` transfer functions (`HISTOGRAM_` or `RANDOM_`)
			if ((tf == HISTOGRAM_NORMALISED_ITERATION_COUNT) && (i < MAX_ITERATIONS)) {
				// Subtract log_{2}(log(|z|)) from `i` and cast as `double`. This accounts for (log-log) escape distance
				mu = (double) i - log(log(sqrt(z_re * z_re + z_im * z_im))) / log(2);
				// Store the normalised escape iteration count at `double` and `int` precision
				iterations_d[y][x] = mu;
				i = (int) mu;
			}

			iterations[y][x] = i;	// Record the escape time (iteration count)

			if ((tf == HISTOGRAM_ESCAPE_VELOCITY) || (tf == HISTOGRAM_NORMALISED_ITERATION_COUNT)) {
				
				/* See https://www.programiz.com/c-programming/c-switch-case-statement for an explanation of switch statements */
				switch (hist_method) {
				case (SERIAL): {
					// In the serial case, we should also manually comment out the `omp parallel` directive(s) at Stage 1 start
					histogram[i]++;
					break;
				}
				case (CRITICAL_SECTION) : {
					// Use a critical section to avoid race conditions (typically the slowest safe parallel implementation)
					#pragma omp critical 
					{
					histogram[i]++;
					}
					break;
				}
				case (LOCAL_HIST_AND_COMBINE): {
					/* The method implemented here for thread-local histograms, later serially combined into a single histogram
					is designed to work in the case where the outer loop (over `y`) only is parallelised, and will
					lead to a race condition to increment `local_histogram` if paralellising over the inner loop over `x`.
					Thus for nested parallel loops, the memory for `local_histogram` should be arranged into a 
					higher-dimensional array with indices `y`, `x`, and `i` before aggregating over both `x` and `y` */
					local_histogram[y][i]++;
					if (i == 0) {
						// This error check should only be relevant when we normalise the iteration count
						printf("Error: recorded escape time of 0 iterations.\n");
					}
					break;
				}
				case (OMP_ATOMIC): {
					/* Atomic operations can be used to safely increment a shared numeric value and are 
					usually faster than critical sections, but one should benchmark to confirm this */
					#pragma omp atomic
					histogram[i]++;
					break;
				}
				}
				
			}
		}
	}

	/* 2.2.2 Here we serially aggregate the thread-local histograms into a single histogram using the master thread (only)
	since this section of code exists outside the scope of any parallel structured block 
	If we did this within a parallel structured block (but outside a parallel `for` loop) then explicit `omp barrier` 
	and `omp master` directives should be used to avoid race conditions. */
	if (hist_method == LOCAL_HIST_AND_COMBINE && (tf == HISTOGRAM_ESCAPE_VELOCITY) || (tf == HISTOGRAM_NORMALISED_ITERATION_COUNT)) {
		for (y = 0; y < HEIGHT; y++)
			for (i = 0; i < MAX_ITERATIONS; i++)
				histogram[i] += local_histogram[y][i];
	}

	if (tf == RANDOM_NORMALISED_ITERATION_COUNT) {
		for (i = 0; i < MAX_ITERATIONS; i++) {
			rand_banding[i].r = rand() % 128;
			rand_banding[i].g = rand() % 64;
			rand_banding[i].b = rand() % 255;
		}
	}

	// STAGE 2) Calculate the transfer function (`rgb` output) for each pixel
	//omp_set_nested(1);
	#pragma omp parallel for default(none) private(y, x) shared(tf, rgb_output)	 schedule(dynamic)
	for (y = 0; y < HEIGHT; y++) {
		// Parallelising this inner loop doesn't seem to infer speedup (nested or otherwise)
		//#pragma omp parallel for default(none) private(x) shared(y, tf, rgb_output)
		for (x = 0; x < WIDTH; x++) {
			/* See https://www.programiz.com/c-programming/c-switch-case-statement for an explanation of switch statements */
			switch (tf) {
			case (ESCAPE_VELOCITY): {
				rgb_output[y][x] = ev_transfer(x, y);
				break;
			}
			case (HISTOGRAM_ESCAPE_VELOCITY): {
				rgb_output[y][x] = h_ev_transfer(x, y);
				break;
			}
			case (HISTOGRAM_NORMALISED_ITERATION_COUNT): {
				rgb_output[y][x] = h_nic_transfer(x, y);
				break;
			}
			case (RANDOM_NORMALISED_ITERATION_COUNT): {
				rgb_output[y][x] = rand_nic_transfer(x, y);
				break;
			}
			}
		}
	}

	// STAGE 3) Write the Mandelbrot set colour plot to file
	fwrite(rgb_output, sizeof(char), sizeof(rgb_output), f);
	fclose(f);

	// Stop timer
	end = omp_get_wtime();

	elapsed = end - begin;
	printf("Complete in %f seconds\n", elapsed);

	return 0;
}

/* Colour transfer functions 
Hue: https://en.wikipedia.org/wiki/Hue 
Hue, Saturation, Lightness: https://en.wikipedia.org/wiki/HSL_and_HSV */
rgb ev_transfer(int x, int y){
	rgb a;
	double hue;
	int its;

	its = iterations[y][x];
	if (its == MAX_ITERATIONS) {
		// Colour black for values of `c` where the sequence has not crossed the threshold within `MAX_ITERATIONS`
		a.r = a.g = a.b = 0;
	}
	else {
		// `hue` proportional to iteration count, scaled to `MAX_ITERATIONS`
		hue = (double) its / MAX_ITERATIONS;
		a.r = a.g = 0;
		// Brighter blue values for slower escape times to clearly highlight the boundary against the set interior
		a.b = (char)(hue * 255.0); // Clamp to range of 0-255
	}
	return a;
}

/* Using the default transfer function `ESCAPE_VELOCITY` has the effect of decreasing image brightness
as the number of iterations increases. This is because the colour value is based on the ratio of the
escape velocity (iterations) and the maximum iterations. As the number of iterations increases,
detail is added at finer levels along the edge of the Mandelbrot set and so the outer parts of the
image become fainter. A better method of colouring uses a histogram normalisation by
keeping track the number of pixels that reached a given iteration. Take a look at the `h_ev_transfer` function.
For each iteration that a pixel has passed it sums the histogram count by the total number of pixels
to the total output to produce a normalised colour. */
rgb h_ev_transfer(int x, int y){
	rgb a;
	double hue;
	int its;
	int i;

	its = iterations[y][x];
	if (its == MAX_ITERATIONS) {
		// Colour black for values of `c` where the sequence has not crossed the threshold within `MAX_ITERATIONS`
		a.r = a.g = a.b = 0;
	}
	else {
		hue = 0;
		// `hue` proportional to sum of other pixels which escaped in fewer iterations
		for (i = 0; i < its; i++) {
			hue += histogram[i];
		}
		// Scale `hue` to image resolution (total number of pixels)
		hue /= (double) WIDTH * HEIGHT;
		a.r = a.g = 0;
		// Brighter blue values for slower escape times to clearly highlight the boundary against the set interior
		a.b = (char)(hue * 255.0); // Clamp to range of 0-255
	}
	return a;
}

/* We calculate an approximation of the exact escape time as a rational number for each pixel during stage 1 and store it 
in the array `iterations_d`, then use it to perform linear interpolation to give smooth shading between colour bands. */
rgb h_nic_transfer(int x, int y) {
	rgb a;
	double hue, hue1, hue2, its_d, frac;
	int i, its;

	its_d = iterations_d[y][x];
	its = iterations[y][x];

	hue1 = hue2 = 0;
	for (i = 0; (i < its) && (its < MAX_ITERATIONS); i++) {
		hue1 += (histogram[i] / (double)(WIDTH * HEIGHT));
	}
	if (i <= MAX_ITERATIONS) { // Probably should be strict inequality in order to set Mandelbrot set as black?
		hue2 = hue1 + (histogram[i] / (double)(WIDTH * HEIGHT));
	}
	a.r = a.g = 0;
	frac = its_d - (int)its_d;
	hue = (1 - frac) * hue1 + frac * hue2;	// Linear interpolation between hues
	a.b = (char)(hue * 255.0);			// Clamp to range of 0-255
	return a;
}

/*  Rather than varying only a single colour channel (i.e. b), we vary all of r, g and b in different ways. 
We do this by having an array of pre-determined random colours for each integer escape velocity 
(smaller values for `MAX_ITERATION` work best for this) */
rgb rand_nic_transfer(int x, int y) {
	rgb a;
	double r_hue, g_hue, b_hue, its_d;
	int its;

	its_d = iterations_d[y][x];
	its = iterations[y][x];

	r_hue = g_hue = b_hue = 0;
	if (its < MAX_ITERATIONS) {
		double frac = its_d - (int)its_d;
		r_hue = (1 - frac) * (double)rand_banding[its].r + frac * (double)rand_banding[its + 1].r;
		g_hue = (1 - frac) * (double)rand_banding[its].g + frac * (double)rand_banding[its + 1].g;
		b_hue = (1 - frac) * (double)rand_banding[its].b + frac * (double)rand_banding[its + 1].b;
	}
	a.r = (char)(r_hue);
	a.g = (char)(g_hue);
	a.b = (char)(b_hue);
	return a;
}
