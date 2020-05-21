/* Assignment 1 Program
This program implements an N-body simulation and visualization on a CPU, along with code for benchmarking performance.
We implement both a serial CPU version, and a version for multi-core processors using OpenMP.
The accompanying report provides discussion on design considerations regarding performance optimization and validation. */
/* Problem Description
We consider a system of N bodies in frictionless 2D space exerting gravitational force on each other.
See https://en.wikipedia.org/wiki/N-body_problem for further background on the physics of the N-body problem.
We simulate the progression of such an N-body system through time using numerical integration by evaluating all pairwise
gravitational interactions between bodies in the system. The force `F_{ij}` of gravity on a body `i` exerted by a body `j`
can be calculated through the following formula: `F_{ij} = G*m_{i}*m_{j}*r_{ji}/|r_{ji}|^{3}` where `G` is the gravitational
constant, `m` denotes the mass of a body, and `r_{ji}` denotes the displacement vector from `i` towards `j`.
This is known as [Newton's Law of Universal Gravitation](https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation)
We add a softening factor `eps` to the denominator to avoid the force between two approaching bodies growing without bound.
This replaces `|r_{ji}|` with `sqrt(|r_{ji}|^{2} + eps^{2})` in the expression in the denominator.
At each time `t_{k}` we calculate the resultant (sum total) force `F_{i;k}` on each body and use this to calculate 
acceleration `a_{i;k}`, then use the [Forward Euler method](https://en.wikipedia.org/wiki/Euler_method) to update the 
velocity and position at time `t_{k+1} = t_{k} + dt` based on `a_{i;k}`, `v_{i;k}`, respectively, where `dt` is the time step. */
/* C Language Library headers */
#include <stdio.h> // http://www.cplusplus.com/reference/cstdio/
#include <stdlib.h> // http://www.cplusplus.com/reference/cstdlib/
#include <string.h> // http://www.cplusplus.com/reference/cstring/
#include <ctype.h> // http://www.cplusplus.com/reference/cctype/
#include <time.h> // http://www.cplusplus.com/reference/ctime/
#include <math.h> // http://www.cplusplus.com/reference/cmath/
/* To enable OpenMP support in your project you will need to include the OpenMP header file `omp.h` 
and enable the compiler to use the OpenMP runtime. 
Set 'OpenMP Support' to 'Yes' (for both Debug and Release builds) in Project->Properties->C/C++->Language
Add `_CRT_SECURE_NO_WARNINGS` to 'Preprocessor Definitions' in Project->Properties->C/C++->Preprocessor */
#include <omp.h>
/* Local header files */
#include "NBody.h"
#include "NBodyVisualiser.h"
/* Preprocessor definitions/macros */
#define USER_NAME "smp16emp" // Replace with your username
#define BUFFER_SIZE 128 // Maximum line length accepted from input file (reasonable as only 5 (comma separated) floating point numbers expected)
/* Function declarations/prototypes */
void print_help();
void parseNDM(const char* argv[3]);
void parse_one_option(const char* argv[2]);
void parse_two_options(const char* argv[4]);
unsigned int parse_str_as_uint(const char* str);
void read_nbody_file(const char* filename, const int N);
void checkLastError(const char* msg);
void step_serial(void);
void step_OpenMP(void);
void swap_float_pointers(float** p1, float** p2);
/* Global variables (shared by/used in multiple functions) */
/* Command line inputs */
unsigned int N; // Number of bodies in the system
unsigned int D; // Dimension of the activity grid
MODE M; // Operation mode. Allows CPU = 0, OPENMP = 1, CUDA = 2, but CUDA mode simulation and rendering not supported in part 1
unsigned int I = 0; // Number of iterations of the simulation to calculate when the `-i` flag is set, else 0
unsigned int f_flag = 0; // Input file flag. 0 if not specified, else such that `input_filename = options[f_flag]` in `main`.
/* Data buffers */
nbody_soa* nbody_in; // Pointer to a structure of arrays (preferred over an array of structures for coalesced memory access)
/* Separate output buffers for updated particle positions are required to avoid interference between loop iterations/threads
when calculating forces based on current particle positions. Buffers for output velocity components are not required 
because a given particle's velocity is only used to calculate its own new position and nothing else. However this requires
each particle's new position be calculated first before its velocity is updated in-place.
Pointer swapping can be used to reduce memory copying between multiple buffers when updating system state.
See https://en.wikipedia.org/wiki/Multiple_buffering for more on double/multiple buffering 
The visualiser only (re)reads position data once after each time the simulation `step` function completes, 
rather than throughout the whole `step` calculation process, so the particles update positions in sync anyway */
float* out_x; // Pointer to store the new `x` coordinate of each body before updating in sync after loops complete
float* out_y; // Pointer to store the new `y` coordinate of each body before updating in sync after loops complete
float* activity_map; // Pointer to flattened array of D*D float values storing normalised particle density values in a 2D grid

/* For information on how to parse command line parameters, see http://www.cplusplus.com/articles/DEN36Up4/ 
`argc` in the count of the command arguments, and `argv` is an array (of length `argc`) of the arguments. 
The first argument is always the executable name (including path) */
int main(const int argc, const char *argv[]) {
	/* Process the command line arguments */
	switch (argc) {
	case 4: // No optional flags used
		parseNDM(&argv[1]);
		break;
	case 6: // One optional flag and argument used
		parse_one_option(&argv[4]);
		parseNDM(&argv[1]);
		break;
	case 8: // Two optional flags with arguments used
		parse_two_options(&argv[4]);
		parseNDM(&argv[1]);
		break;
	default: // The expected arguments are: "nbody.exe N D M [-i I] [-f input_file]"
		fprintf(stderr, "Error: Unexpected number of arguments. %d arguments (including executable name) received\n", argc);
		print_help();
		exit(EXIT_FAILURE);
	}

	// Declare a function pointer to a simulation step function and set its value according to the operation mode `M`
	void (*simulate)(void) = NULL; // Declare `simulate` as pointer to function (void) returning void
	switch (M) {
	case CPU:
		simulate = &step_serial;
		break;
	case OPENMP:
		simulate = &step_OpenMP;
		printf("OpenMP using %d threads\n", omp_get_max_threads());
		break;
	case CUDA:
		fprintf(stderr, "Error: CUDA Mode simulation not supported until assignment 2!\n");
		exit(EXIT_FAILURE);
		break;
	}

	/* Allocate Heap Memory */
	// Calculate memory requirements
	const unsigned int data_column_size = sizeof(float) * N;
	const unsigned int activity_grid_size = sizeof(float) * D * D;

	// Memory allocation. See http://www.cplusplus.com/reference/cstdlib/malloc/
	nbody_in = (nbody_soa*)malloc(sizeof(nbody_soa));
	nbody_in->x = (float*)malloc(data_column_size);
	nbody_in->y = (float*)malloc(data_column_size);
    // Allocates memory block for length N array of floats, and initialize all bits to zero (for default zero initial velocity).
	// See http://www.cplusplus.com/reference/cstdlib/calloc/
	nbody_in->vx = (float*)calloc(N, sizeof(float)); // Zero initial velocity
	nbody_in->vy = (float*)calloc(N, sizeof(float)); // Zero initial velocity
	nbody_in->m = (float*)malloc(data_column_size);
	if ((nbody_in == NULL) || (nbody_in->x == NULL) || (nbody_in->y == NULL) || (nbody_in->vx == NULL) || (nbody_in->vy == NULL) || (nbody_in->m == NULL)) {
		fprintf(stderr, "Error allocating host memory (`nbody_in`) for system with %d bodies\n", N);
		exit(EXIT_FAILURE);
	}
	out_x = (float*)malloc(data_column_size);
	out_y = (float*)malloc(data_column_size);
	if ((out_x == NULL) || (out_y == NULL)) {
		fprintf(stderr, "Error allocating host memory (output position buffers) for system with %d bodies\n", N);
		exit(EXIT_FAILURE);
	}
	activity_map = (float*)malloc(activity_grid_size);
	if (activity_map == NULL) {
		fprintf(stderr, "Error allocating host memory (`activity map`) for system with %d bodies, activity grid size %d\n", N, D);
		exit(EXIT_FAILURE);
	}

	/* Read initial data from file, or generate random initial state according to optional program flag `-f`. */
	if (f_flag == 0) { // No input file specified, so a random initial N-body state will be generated
		const float one_over_N = (float)1 / N; // Store the inverse of `N` as a constant to avoid recalculating in loop
		for (unsigned int i = 0; i < N; i++) {
			nbody_in->x[i] = (float)rand() / RAND_MAX; // Random position in [0,1]
			nbody_in->y[i] = (float)rand() / RAND_MAX; // Random position in [0,1]
			nbody_in->m[i] = one_over_N; // Mass distributed equally among N bodies
			// Note that velocity data has already been initialized to zero for all bodies
		}
	}
	else { // Attempt to read initial N-body system state from input csv file
		read_nbody_file(argv[f_flag], N);
	}

	/* According to the value of program argument `I` either configure and start the visualiser, 
	or perform a fixed number of simulation steps and output the timing results. */
	if (I == 0) { // Run visualiser when number of iterations not specified with `-i` flag, or otherwise `I` was set to 0
		initViewer(N, D, M, simulate); // The simulation step function has been set earlier according to operation mode `M`
		setNBodyPositions(nbody_in); // This is where the visualiser will check for particle position data after each iteration
		setActivityMapData(activity_map); // This is where the visualiser will check for activity data after each iteration
		startVisualisationLoop();
	}
	else { // Run the simulation for `I` iterations and output the timing results
		// Declare timing variables
		clock_t t; // Clock ticks for serial CPU timing
		double start, end; // Timestamps for OpenMP timing
		double seconds = 0; // Variable to hold execution timing results
		switch (M) {
		case CPU:
			t = clock(); // Starting timestamp. See http://www.cplusplus.com/reference/ctime/clock/
			for (unsigned int i = 0; i < I; i++) {
				step_serial();
			}
			t = clock() - t; // Take end timestamp and calculate difference from start in clock ticks
			seconds = (double)t / CLOCKS_PER_SEC;
			break;
		case OPENMP:
			start = omp_get_wtime(); // Starting timestamp. See https://www.openmp.org/spec-html/5.0/openmpsu160.html
			for (unsigned int i = 0; i < I; i++) {
				step_OpenMP();
			}
			end = omp_get_wtime();
			seconds = end - start;
			break;
		case CUDA:
			fprintf(stderr, "Error: CUDA Mode simulation not supported until assignment 2!\n");
			exit(EXIT_FAILURE);
			break;
		}
		printf("Execution time %d seconds %d milliseconds for %d iterations\n", (int)seconds, (int)((seconds-(int)seconds)*1000), I);
	}

	// Cleanup
	free(nbody_in->x);
	free(nbody_in->y);
	free(nbody_in->vx);
	free(nbody_in->vy);
	free(nbody_in->m);
	free(nbody_in);
	free(out_x);
	free(out_y);
	free(activity_map);

	return 0;
}

/* Functions to perform the main simulation of the Nbody system by updating the state by one time step */
// Serial CPU version
void step_serial(void) {
	/* The index `i` is used to iterate over the `N` bodies in the system. For each body `i`, we choose to calculate the
	`N-1` interactions of the other bodies `j != i` on `i`, as opposed to the action of `i` on all of the other bodies `j != i`.
	When computed in parallel, the former avoids a synchronisation step so that the velocity of each body `i`
	can be updated independently of the other threads, reducing idle time. Afterwards, we can also update 
	the position of body `i` and calculate which histogram bin/activity grid cell the body `i` is in, all within one loop.
	This is known as loop jamming or loop fusion. See http://www.it.uom.gr/teaching/c_optimization/tutorial.html
	Calculating the histogram contribution of each body is far more efficient than iterating over histogram bins/grid cells
	since we exploit the fact that each body can only be in at most one grid cell at a time (D*D times fewer calculations). */
	unsigned int i, j; // Counter variables
	float ax, ay; // Components of resultant acceleration of a particle as a result of gravitational force
	float local_xi, local_yi; // Local position variables to reduce global memory accesses, especially during inner loop
	float local_vxi, local_vyi; // Local velocity variables to exchange two global memory reads for one plus two local reads
	float x_ji, y_ji; // Components of displacement vector from particle `i` to particle `j`
	float dist_ij; // To hold softened distance `sqrt(|r_{ji}|^{2} + eps^{2})` from `i` to `j`

	// Reset histogram values to zero with `memset`. See http://www.cplusplus.com/reference/cstring/memset/
	memset(activity_map, 0, sizeof(activity_map));

	for (i = 0; i < N; i++) { // Iterating over bodies in the Nbody system
		ax = 0; // Reset resultant acceleration in `x` direction to zero for new particle
		ay = 0; // Reset resultant acceleration in `y` direction to zero for new particle
		// Read position data from global memory to the stack
		local_xi = nbody_in->x[i];
		local_yi = nbody_in->y[i];

		for (j = 0; j < N; j++) {
			if (j == i) { // Skip the calculation when i = j (saves calculation time; could consider branching effects on GPU)
				continue;
			}
			// Calculate displacement from particle `i` to particle `j`, since common expression in force equation
			// Using local variables for `x[i]`, `y[i]` here removes a global memory read from each inner loop iteration
			x_ji = nbody_in->x[j] - local_xi;
			y_ji = nbody_in->y[j] - local_yi;
			// Calculate distance from `i` to `j` with softening factor since used in denominator of force expression
			// Explicit casting required since `sqrt` function expects `double` type input and output; operation execution order
			dist_ij = (float)sqrt((double)x_ji * x_ji + (double)y_ji * y_ji + eps_sq);
			/* Add unscaled contribution to acceleration due to gravitational force of `j` on `i`
			Universal Gravitation: `F_ij = G * m_i * m_j * r_ji / |r_ji|^3` ; Newton's 2nd Law: F_i = m_i * a_i
			See top of file for further explanation of calculation, physical background */
			ax += nbody_in->m[j] * x_ji / (dist_ij * dist_ij * dist_ij); // Need to scale by `G` later
			ay += nbody_in->m[j] * y_ji / (dist_ij * dist_ij * dist_ij); // Need to scale by `G` later
			/* It would be possible to add force/acceleration contributions to `nbody_in->v` directly within this inner loop.
			However this would cause this function to be bound by memory access latency (repeated writes to `nbody_in->v`).
			Therefore we use the temporary/local variables `ax` and `ay` instead */
		}
		/* Use current velocity, acceleration to calculate position, velocity at next time step, respectively. */
		/* Former code uses extra heap memory buffers for velocity, adding extra steps pointer swapping and using more memory
		However this implementation scores highly for readability, as it makes the intended outcome clear (no race conditions)
		out_x[i] = nbody_in->x[i] + nbody_in->vx[i] * dt;
		out_y[i] = nbody_in->y[i] + nbody_in->vy[i] * dt;
		out_vx[i] = nbody_in->vx[i] + G * ax * dt;
		out_vy[i] = nbody_in->vy[i] + G * ay * dt; */
		// Using local velocity variables also reduces global memory reads, but only marginally compared `local_xi`, `local_yi`
		local_vxi = nbody_in->vx[i];
		local_vyi = nbody_in->vy[i];
		// More care has to be taken about the order of execution to ensure the output positions are calculated correctly
		// Use current velocity to calculate next position
		local_xi += local_vxi * dt;
		local_yi += local_vyi * dt;
		// Now the local position variables hold the new positions and can be used to update the activity map
		// Use current acceleration (based on current positions) to calculate the new velocity
		// Scale `ax`, `ay` by gravitational constant `G`. See `NBody.h` for definition and comment.
		nbody_in->vx[i] = local_vxi + G * ax * dt; // Write the new velocity back to `nbody_in->vx[i]`
		nbody_in->vy[i] = local_vyi + G * ay * dt; // Write the new velocity back to `nbody_in->vy[i]`
		// We can update particle velocities in-place without adversely affecting subsequent iterations/other threads

		// Update the activity map - a flat array of D*D float values storing normalised particle density values in a 2D grid
		// First check whether the new position of particle `i` is within the activity grid [0,1)^{2}
//		if ((out_x[i] >= 0) && (out_x[i] < 1) && (out_y[i] >= 0) && (out_y[i] < 1)) {
		if ((local_xi >= 0) && (local_xi < 1) && (local_yi >= 0) && (local_yi < 1)) {
			// If so, calculate the index of the grid element that particle `i` is in and increment the associated histogram bin
			// Multiply position vector by `D` then truncate components to `int` to find position in \{0,...,D-1\}^{2} grid
			// Can result in race condition when outer `i` loop parallel as multiple threads could increment at once
			// Possible solutions: Critical section; atomic operator; move section outside parallel loop (barrier/master method)
			activity_map[D * (int)(D * local_yi) + (int)(D * local_xi)]++; // Linearize the index from 2D grid into 1D array
		}
		// Write the new position of particle `i` to the output buffers to avoid interfering with subsequent iterations
		out_x[i] = local_xi;
		out_y[i] = local_yi;
	}
	// Scale activity map values by `D / N` to normalize the histogram values and then scale by D to increase brightness
	const float one_over_N = (float)1 / N; // Store the inverse of global variable `N` as a constant to cache value
	for (i = 0; i < D * D; i++) {
		activity_map[i] *= one_over_N * D;
	}
	/* Finally, update the `nbody` data pointers to reference the newly calculated arrays of position data.
	We swap the input and output pointers rather than simply overwriting the input pointers because that would result
	in losing the original input pointers, losing allocated heap memory addresses and causing a memory leak! */
	float* temp; // Declare a temporary pointer to `float` to hold addresses whilst swapping the input and output pointers
	temp = nbody_in->x; // Keep track of the old input pointer for later use so we don't lose any allocated memory
	nbody_in->x = out_x; // Update the `nbody_in->x` pointer which is used for visualisation, and the next `step` iteration
	out_x = temp; // Reset `out_x` to a 'fresh', 'empty' piece of memory
	temp = nbody_in->y; // Keep track of the old input pointer for later use so we don't lose any allocated memory
	nbody_in->y = out_y; // Update the `nbody_in->y` pointer which is used for visualisation, and the next `step` iteration
	out_y = temp; // Reset `out_y` to a distinct piece of 'fresh' and 'empty' memory
}

/* Profiling with Visual Studio's Diagnostic Tools and PerfTips by setting breakpoints to time code segments and using
Debug->Windows->Show Diagnostic Tools https://docs.microsoft.com/en-us/visualstudio/profiling/profiling-feature-tour?view=vs-2019
Shows that as `N` increases the majority of time spent running the programme is spent calling the simulation step function,
and within that the loop over `N` particles (indexed by `i`) occupies most of the compute time rather than the loop over
the activity grid cells. This makes sense since there are far more compute steps within the `i` loop, and generally D will be
much smaller than `N` and is effectively limited in visualisation by screen resolution.
Therefore it is most important to parallelise the outer `i` loop. Further analysis suggests that as `N` increases further,
the majority of time spent within each outer loop is spent iterating over the inner `j` loop of interactions between particles,
so nested parallel loops should also be considered. Amongst all the operations/function calls within each simulation step,
it appears that the call to the `sqrt` function in the inner loop is the most expensive. */

// OpenMP version (For parallel computation on a multicore CPU)
/* Benchmarking results for parallelising outer loop over `i` (on my 4 core personal laptop)
Command Line Arguments | Histogram Race Handling | Scheduling | Execution Time(s)
"16384 16 CPU -i 10" | N/A | N/A | 61.357s, 61.988s
"16384 16 OPENMP -i 10" | Atomic | schedule(static) | 17.184s, 17.624s, 17.713s
"16384 16 OPENMP -i 10" | Atomic | schedule(static, 1) | 17.314s, 17.399s
"16384 16 OPENMP -i 10" | Atomic | schedule(static, 2) | 17.182s, 17.478s
"16384 16 OPENMP -i 10" | Atomic | schedule(static, 4) | 17.225s, 17.363s
"16384 16 OPENMP -i 10" | Atomic | schedule(static, 8) | 17.248s, 17.446s
"16384 16 OPENMP -i 10" | Atomic | schedule(guided) | 15.769s, 16.650s, 16.830s, 17.110s
"16384 16 OPENMP -i 10" | Atomic | schedule(dynamic) | 11.895s, 12.028s, 12.210s
"16384 16 OPENMP -i 10" | Atomic | schedule(dynamic, 2) | 11.841s, 11.997s, 12.266s
"16384 16 OPENMP -i 10" | Atomic | schedule(dynamic, 4) | 11.766s, 12.047s, 12.210s
"16384 16 OPENMP -i 10" | Critical | schedule(static, 4) | 17.341s, 18.007s
"16384 16 OPENMP -i 10" | Critical | schedule(guided) | 16.720s, 16.994s
"16384 16 OPENMP -i 10" | Critical | schedule(dynamic) | 11.617s, 11.888s, 11.999s, 13.054s
"8192 16 CPU -i 100" | N/A | N/A | 153.307s, 154.100s
"8192 16 OPENMP -i 100" | Atomic | schedule(dynamic) | 29.687s, 30.332s
"8192 16 OPENMP -i 100" | Critical | schedule(dynamic) | 28.490s, 30.305s
"2048 16 CPU -i 1000" | N/A | N/A | 95.132s, 95.174s, 95.677s
"2048 16 OPENMP -i 1000" | Atomic | schedule(static) | 29.240s, 29.584s
"2048 16 OPENMP -i 1000" | Atomic | schedule(static, 4) | 29.685s, 30.073s
"2048 16 OPENMP -i 1000" | Atomic | schedule(guided) | 29.357s, 29.521s
"2048 16 OPENMP -i 1000" | Atomic | schedule(dynamic) | 17.910s, 18.774s
"2048 16 OPENMP -i 1000" | Critical | schedule(dynamic) | 18.470s, 18.577s
"512 16 CPU -i 10000" | N/A | N/A | 59.003s, 59.404s
"512 16 OPENMP -i 10000" | Atomic | schedule(dynamic) | 12.734s, 13.735s
"256 16 CPU -i 100000" | N/A | N/A | 148.552s, 149.194s
"256 16 OPENMP -i 100000" | Atomic | schedule(static, 4) | 47.987s, 48.889s
"256 16 OPENMP -i 100000" | Atomic | schedule(dynamic) | 34.568s, 36.429s
"128 16 CPU -i 100000" | N/A | N/A | 37.436s, 37.520s
"128 16 OPENMP -i 100000" | Atomic | schedule(static, 4) | 12.587s, 13.110s
"128 16 OPENMP -i 100000" | Atomic | schedule(guided) | 12.448s, 12.541s
"128 16 OPENMP -i 100000" | Atomic | schedule(dynamic) | 11.717s, 12.071s
"64 16 CPU -i 1000000" | N/A | N/A | 92.477s, 92.617s
"64 16 OPENMP -i 1000000" | Atomic | schedule(static, 4) | 31.302s, 32.235s
"64 16 OPENMP -i 1000000" | Atomic | schedule(guided) | 32.292s, 33.169s
"64 16 OPENMP -i 1000000" | Atomic | schedule(dynamic) | 38.062s, 38.133s */
/* Initial Remarks on parallelising outer loop
The data shows that dynamic scheduling is faster for values of `N` greater than 100 or so, but slower than static scheduling for
smaller values on `N`, with guided scheduling always performing between to static and dynamic scheduling and never optimal. 
This is because there is uneven workload amongst threads, which favours dynamic scheduling, but the overhead cost of dynamic 
scheduling at runtime becomes a limiting factor for relatively small parallel loops.
A trend of increasing OpenMP performance relative to serial CPU performance as `N` increases can also be seen as the benefits
of parallelism outweigh their overhead costs.
I believe the main source of difference in workload between threads arises from whether the particle `i` lies within 
the activity grid or not. If so, a slow (atomic/critical/serial) incrementation of an activity grid cell must occur, which also
involves writing to global memory at an index of the `activity_map` array that cannot be predicted at compile time, but if
the particle `i` lies outside the activity grid this step can be skipped, causing an uneven workload between different threads.
Scheduling approach seems to have a more important impact on performance than chunk size.
On the other hand, it appears that there's no major difference in performance between using a critical section or an atomic
directive to ensure the safe incrementation of the activity grid histogram, perhaps with only a slight leaning towards atomic.
For reference information on the OMP Atomic directive, see the following links:
https://www.openmp.org/spec-html/5.0/openmpsu95.html
https://www.ibm.com/support/knowledgecenter/SSGH2K_13.1.2/com.ibm.xlc131.aix.doc/compiler_ref/prag_omp_atomic.html */
/* Benchmarking results for parallelising inner loop over `j` only (on my 4 core personal laptop)
Command Line Arguments | Acceleration Sum Handling | Scheduling | Execution Time(s)
"8192 16 OPENMP -i 10" | Two Atomic Directives | schedule(static) | 54.309s
"8192 16 OPENMP -i 10" | Two Atomic Directives | schedule(dynamic) | 49.997s
"8192 16 OPENMP -i 10" | Critical Section | schedule(static) | 54.520s
"8192 16 OPENMP -i 10" | Critical Section | schedule(dynamic) | 61.185s
"256 16 OPENMP -i 10000" | Two Atomic Directives | schedule(static) | 57.011s
"256 16 OPENMP -i 10000" | Two Atomic Directives | schedule(static, 4) | 64.039s
"256 16 OPENMP -i 10000" | Two Atomic Directives | schedule(dynamic) | 90.021s
"256 16 OPENMP -i 10000" | Critical Section | schedule(static) | 58.980s
"256 16 OPENMP -i 10000" | Critical Section | schedule(dynamic) | 79.218s */
/* Remarks on parallelising inner loop
The data shows that only parallelising the inner `j` loop over force interactions results in a 3-4x slowdown 
compared to the serial CPU version. This is because of repeated overheads setting up small parallel loops within a larger loop.
The story might be different on a machine with more cores (e.g. 16 cores rather than 4 cores), but when compared to the 
3-4x speedup over serial implementation from parallelising the outer `i` loop over bodies in the system, it is clear which is
preferred. As a general rule, outer loops should be parallelised first (assuming they run for a reasonable number of iterations).
Final remarks and conclusion
Finally, through testing an implementation of nested parallel loops we find the following performance heirarchy for the given
problem: Parallel outer loop > Serial CPU version > Parallel histogram scaling > Nested parallel loops > Parallel inner loop.
For **nested** parallel loops with dynamic scheduling and atomic directives to avoid race conditions when 
1) Incrementing activity map contributions; and 2) Summing acceleration contributions with the inner `j` loop parallel;
given command line arguments "8192 16 OPENMP -i 100", an execution time of 268.809s was recorded, about 75% slower than serial. 
In conclusion, we choose to parallelise the force calculation outer loop over `i` iterating over bodies in the system as it
is the best and only loop parallelisation which improves on the serial CPU version (by a respectable 3-6x speedup), 
we choose dynamic scheduling since it outperforms static and guided scheduling for values of `N` greater than around 100, where
many feasible values of `N` lie (a separate parallel directive to choose static scheduling when N < 100 could be considered).
Finally, to avoid race conditions when each thread uses the position of its local particle to update the shared `activity_map`
histogram, we choose to use an atomic directive, though this only appears to be negligably better than a critical section. */
void step_OpenMP(void) {
	/* The index `i` is used to iterate over the `N` bodies in the system. For each body `i`, we choose to calculate the
	`N-1` interactions of the other bodies `j != i` on `i`, as opposed to the action of `i` on all of the other bodies `j != i`.
	This is because the latter requires an extra synchronisation step before the velocity of each body `i` can be calculated,
	increasing thread idle time. Subsequently, we can also update the position of body `i` and calculate its activity grid
	position within the same parallel loop, reducing overhead. This is known as loop jamming or loop fusion. 
	See http://www.it.uom.gr/teaching/c_optimization/tutorial.html
	Calculating the histogram contribution of each body is far more efficient than iterating over histogram bins/grid cells
	since we exploit the fact that each body can only be in at most one grid cell at a time (D*D times fewer calculations). */
	int i, j; // Counter variables. OpenMP requires these to be `int` type rather than unsigned
	float ax, ay; // Components of resultant acceleration of a particle as a result of gravitational force
	float local_xi, local_yi; // Local position variables to reduce global memory accesses, especially during inner loop
	float local_vxi, local_vyi; // Local velocity variables to exchange two global memory reads for one plus two local reads
	float x_ji, y_ji; // Components of displacement vector from particle `i` to particle `j`
	float dist_ij; // To hold softened distance `sqrt(|r_{ji}|^{2} + eps^{2})` from `i` to `j`

	// Reset histogram values to zero with `memset`. See http://www.cplusplus.com/reference/cstring/memset/
	memset(activity_map, 0, sizeof(activity_map));

	//omp_set_nested(1);
#pragma omp parallel for default(none) private(i, j, ax, ay, local_xi, local_yi, x_ji, y_ji, dist_ij, local_vxi, local_vyi) shared(nbody_in, activity_map, D, out_x, out_y) schedule(dynamic)
	for (i = 0; i < N; i++) { // Iterating over bodies in the Nbody system
		ax = 0; // Reset resultant acceleration in `x` direction to zero for new particle
		ay = 0; // Reset resultant acceleration in `y` direction to zero for new particle
		// Read position data from global memory to the stack
		local_xi = nbody_in->x[i];
		local_yi = nbody_in->y[i];

// Can treat `i` as a shared variable on the inner `j` loop since we read without changing within each outer loop iteration
// Otherwise could use `firstprivate(i)` declaration to pass in the value to each thread
//#pragma omp parallel for default(none) private(j, x_ji, y_ji, dist_ij) shared(i, ax, ay, local_xi, local_yi, nbody_in) schedule(dynamic)
		for (j = 0; j < N; j++) {
			if (j == i) { // Skip the calculation when i = j (saves calculation time; could consider branching effects on GPU)
				continue;
			}
			// Calculate displacement from particle `i` to particle `j`, since common expression in force equation
			// Using local variables for `x[i]`, `y[i]` here removes a global memory read from each inner loop iteration
			x_ji = nbody_in->x[j] - local_xi;
			y_ji = nbody_in->y[j] - local_yi;
			// Calculate distance from `i` to `j` with softening factor since used in denominator of force expression
			// Explicit casting required since `sqrt` function expects `double` type input and output; operation execution order
			dist_ij = (float)sqrt((double)x_ji * x_ji + (double)y_ji * y_ji + eps_sq);
			/* Add unscaled contribution to acceleration due to gravitational force of `j` on `i`
			Universal Gravitation: `F_ij = G * m_i * m_j * r_ji / |r_ji|^3` ; Newton's 2nd Law: F_i = m_i * a_i
			See top of file for further explanation of calculation, physical background */
			// If the inner `j` loop is parallel, adding to `ax[i]` will result in a race condition. 
			// Could try a reduction directive for `ax`, `ay` in the parallel inner loop directive if supported by OpenMP 2.0
			//#pragma omp critical {
			ax += nbody_in->m[j] * x_ji / (dist_ij * dist_ij * dist_ij); // Need to scale by `G` later
			ay += nbody_in->m[j] * y_ji / (dist_ij * dist_ij * dist_ij); // Need to scale by `G` later
			//}
			/* It would be possible to add force/acceleration contributions to `nbody_in->v` directly within this inner loop.
			However this would cause this function to be bound by memory access latency (repeated writes to `nbody_in->v`).
			Therefore we use the temporary/local variables `ax` and `ay` instead */
		}
		/* Use current velocity, acceleration to calculate position, velocity at next time step, respectively. */
		/* Former code uses extra heap memory buffers for velocity, adding extra steps pointer swapping and using more memory
		However this implementation scores highly for readability, as it makes the intended outcome clear (no race conditions)
		out_x[i] = nbody_in->x[i] + nbody_in->vx[i] * dt;
		out_y[i] = nbody_in->y[i] + nbody_in->vy[i] * dt;
		out_vx[i] = nbody_in->vx[i] + G * ax * dt;
		out_vy[i] = nbody_in->vy[i] + G * ay * dt; */
		// Using local velocity variables also reduces global memory reads, but only marginally compared `local_xi`, `local_yi`
		local_vxi = nbody_in->vx[i];
		local_vyi = nbody_in->vy[i];
		// More care has to be taken about the order of execution to ensure the output positions are calculated correctly
		// Use current velocity to calculate next position
		local_xi += local_vxi * dt;
		local_yi += local_vyi * dt;
		// Now the local position variables hold the new positions and can be used to update the activity map
		// Use current acceleration (based on current positions) to calculate the new velocity
		// Scale `ax`, `ay` by gravitational constant `G`. See `NBody.h` for definition and comment.
		nbody_in->vx[i] = local_vxi + G * ax * dt; // Write the new velocity back to `nbody_in->vx[i]`
		nbody_in->vy[i] = local_vyi + G * ay * dt; // Write the new velocity back to `nbody_in->vy[i]`
		// We can update particle velocities in-place without adversely affecting subsequent iterations/other threads

		// Update the activity map - a flat array of D*D float values storing normalised particle density values in a 2D grid
		// First check whether the new position of particle `i` is within the activity grid [0,1)^{2}
//		if ((out_x[i] >= 0) && (out_x[i] < 1) && (out_y[i] >= 0) && (out_y[i] < 1)) {
		if ((local_xi >= 0) && (local_xi < 1) && (local_yi >= 0) && (local_yi < 1)) {
			// If so, calculate the index of the grid element that particle `i` is in and increment the associated histogram bin
			// Multiply position vector by `D` then truncate components to `int` to find position in \{0,...,D-1\}^{2} grid
			// Can result in race condition when outer `i` loop parallel as multiple threads could increment at once
			// Possible solutions: Critical section; atomic operator; move section outside parallel loop (barrier/master method)
			/* Atomic operations can be used to safely increment a shared numeric value; critical regions have other uses too */
			#pragma omp atomic
			activity_map[D * (int)(D * local_yi) + (int)(D * local_xi)]++; // Linearize the index from 2D grid into 1D array
		}
		// Write the new position of particle `i` to the output buffers to avoid interfering with other threads/iterations
		out_x[i] = local_xi;
		out_y[i] = local_yi;
	}
	// Scale activity map values by `D / N` to normalize the histogram values and then scale by D to increase brightness
	const float one_over_N = (float)1 / N; // Store the inverse of global variable `N` as a constant to cache value
	/* Parallelising this histogram scaling loop actually has a negative impact on performance due to fork/join overheads
	outweighing the small gains from parallelising a non-compute intensive loop. Using command line arguments (release mode)
	"2048 1024 OPENMP -i 100" we reliably time 9.8 seconds for serial execution vs 11.5s-11.9s with this loop parallel 
	and using static or guided scheduling (chunk size has little effect) and 14.5s-14.9s for dynamic scheduling. 
	The reason dynamic scheduling is even slower than static scheduling is the extra runtime overheads of dynamic scheduling
	where the workloads are extremely uniform (two multiplications per loop) */
//#pragma omp parallel for default(none) private(i) shared(activity_map, one_over_N, D) schedule(dynamic)
	for (i = 0; i < D * D; i++) {
		activity_map[i] *= one_over_N * D;
	}
	/* Finally, update the `nbody` data pointers to reference the newly calculated arrays of position data.
	We swap the input and output pointers rather than simply overwriting the input pointers because that would result
	in losing the original input pointers, losing allocated heap memory addresses and causing a memory leak! */
	float* temp; // Declare a temporary pointer to `float` to hold addresses whilst swapping the input and output pointers
	temp = nbody_in->x; // Keep track of the old input pointer for later use so we don't lose any allocated memory
	nbody_in->x = out_x; // Update the `nbody_in->x` pointer which is used for visualisation, and the next `step` iteration
	out_x = temp; // Reset `out_x` to a 'fresh', 'empty' piece of memory
	temp = nbody_in->y; // Keep track of the old input pointer for later use so we don't lose any allocated memory
	nbody_in->y = out_y; // Update the `nbody_in->y` pointer which is used for visualisation, and the next `step` iteration
	out_y = temp; // Reset `out_y` to a distinct piece of 'fresh' and 'empty' memory
}

void swap_float_pointers(float** p1, float** p2) {
	// Function arguments are always passed by value, so to swap two pointers, we must pass references to those pointers
	// The arguments `p1` and `p2` are actually addresses of pointers to `float` data (rather than the pointers themselves)
	float* temp = *p1; // Set `temp` to be the pointer referenced by p1
	*p1 = *p2; // Overwrite the pointer addressed by `p1` with the pointer addressed by `p2`
	*p2 = temp; // Overwrite the pointer addressed by `p2` with the pointer addressed by `temp` (originally addressed by `p1`)
}

/* Functions for parsing Command Line Arguments
The expected arguments are: "nbody.exe N D M [-i I] [-f input_file]" */
void print_help() {
	printf("USAGE: \"nbody.exe N D M [-i I] [-f input_file]\", where\n");
	printf("              N  is the number of bodies to simulate.\n");
	printf("              D  is the integer dimension of the activity grid. The Grid has D*D locations.\n");
	printf("              M  is the operation mode, either `CPU` or `OPENMP`\n");
	printf("         [-i I]  [OPTIONAL] Specifies number `I` of simulation iterations to perform. Visualisation mode is used when `-i` flag not set.\n");
	printf("[-f input_file]  [OPTIONAL] Specifies an input file with an initial `N` bodies of data. A random initial state will be generated when `-f` flag not set.\n");
}

void parseNDM(const char* argv[3]) {
	N = parse_str_as_uint(argv[0]);
	checkLastError("Error parsing argument for `N` to `int`");
	if (N == 0) {
		fprintf(stderr, "Error: Argument \"%s\" for number of bodies `N` parsed as 0.\n", argv[0]);
		print_help();
		exit(EXIT_FAILURE);
	}
	D = parse_str_as_uint(argv[1]);
	checkLastError("Error parsing argument for `D` to `int`");
	if (strcmp(argv[2], "CPU") == 0) {
		M = CPU;
	}
	else if (strcmp(argv[2], "OPENMP") == 0) {
		M = OPENMP;
	}
	else if (strcmp(argv[2], "CUDA") == 0) {
		M = CUDA;
	}
	else {
		fprintf(stderr, "Error: Unexpected value %s for operation mode `M` (case sensitive).\n", argv[3]);
		print_help();
		exit(EXIT_FAILURE);
	}
}

void parse_one_option(const char* options[2]) {
	if (strcmp(options[0], "-i") == 0) {
		I = parse_str_as_uint(options[1]);
		checkLastError("Error parsing argument for `I` to `int`");
	}
	else if (strcmp(options[0], "-f") == 0) {
		f_flag = 5;
	}
	else { // Invalid option flag
		fprintf(stderr, "Error: Unexpected optional arguments/flags received.\n");
		print_help();
		exit(EXIT_FAILURE);
	}
}

void parse_two_options(const char* options[4]) {
	if ((strcmp(options[0], "-i") == 0) && (strcmp(options[2], "-f") == 0)) {
		I = parse_str_as_uint(options[1]);
		checkLastError("Error parsing argument for `I` to `int`");
		f_flag = 7;
	}
	else if ((strcmp(options[0], "-f") == 0) && (strcmp(options[2], "-i") == 0)) {
		I = parse_str_as_uint(options[3]);
		checkLastError("Error parsing argument for `I` to `int`");
		f_flag = 5;
	}
	else { // Invalid option flag combination
		fprintf(stderr, "Error: Unexpected combination of optional arguments/flags received.\n");
		print_help();
		exit(EXIT_FAILURE);
	}
}

unsigned int parse_str_as_uint(const char * str) {
	if (isdigit(str[0]) == 0) { // In particular, this excludes leading minus sign/negative input values.
		fprintf(stderr, "Error parsing %s as `int`: First char not decimal digit (negative values not permitted).\n", str);
		print_help();
		exit(EXIT_FAILURE);
	}
	unsigned int val; // To hold parsed `unsigned int` value
	char* pEnd; // Pointer to first character after number in `str`
	val = (unsigned int)strtol(str, &pEnd, 10); // Convert string to long integer in base 10. Set `pEnd`.
	if (pEnd[0] != '\0') { // Check for extra characters in `str` after initial number (can include decimal point)
		fprintf(stderr, "Error: Unexpected characters in string %s when parsing to `int`.\n", str);
		print_help();
		exit(EXIT_FAILURE);
	} 
	return val;
}

/* Functions for reading input files */
void read_nbody_file(const char* filename, const int N) {
	FILE* f; // Input file handle
	char line_buffer[BUFFER_SIZE]; // Buffer to hold lines read from file
	char* ptr_ch = NULL; // Pointer to track character position when reading `line_buffer` string
	int line_number = 0; // Keep track of line number for error messaging
	int body_count = 0; // Count of number of body data lines read to ensure it matches `N`

	f = fopen(filename, "r"); // Open the file in read-only mode
	if (f == NULL) {
		fprintf(stderr, "Error opening file '%s' for reading\n", filename);
		exit(EXIT_FAILURE);
	}

	/* Read file line by line with `fgets` function. See http://www.cplusplus.com/reference/cstdio/fgets/ for reference
	Reads from file into buffer until (soonest of) either `\n` or `EOF` is read, or `BUFFER_SIZE-1` characters read */
	while (fgets(line_buffer, BUFFER_SIZE, f) != NULL) {
		line_number++; // Increment count of lines read
		if (line_buffer[0] == '#') { // If first char in line is `#` skip to next line to ignore comments
			continue;
		}
		if (line_buffer[strlen(line_buffer) - 1] != '\n') { // If last char read from file is not '\n', the line is too long
			// This checks that a full line of data was written from file to buffer when not a comment line
			fprintf(stderr, "Error reading line %u: Line length exceeds buffer size of %d characters\n", line_number, BUFFER_SIZE);
			exit(EXIT_FAILURE);
		}

		/* Read the line of data into `nbody_in`, using comma character `,` as delimiter to separate data values 
		This could be considered as an unrolled while loop over commas counted using `strchr` calls with nontrivial control flow 
		The use of `ptr_ch` as a separate variable from `line_buffer` could probably be removed. */
		ptr_ch = line_buffer; // Place `ptr_ch` at the start of the line to be read
		/* Use `strchr` to search through the line starting at position `ptr_ch` to find the next comma `,` character
		Returns `NULL` pointer if no comma `,` character found in line after position `ptr_ch`
		See http://www.cplusplus.com/reference/cstring/strchr/ for reference */
		if ((strchr(ptr_ch, ',') == NULL)) { // Check for comma after first data value
			fprintf(stderr, "Error reading line %u: No data delimiters (`,`) detected\n", line_number);
			exit(EXIT_FAILURE);
		}
		else { // This appears to be a valid data line. Don't write past memory bounds for `nbody_in`!
			if (body_count > N-1) { // Throw an error if we have already read `N` or more data rows
				fprintf(stderr, "Error reading line %u: Num bodies in file exceeds input N (%d)\n", line_number, N);
				exit(EXIT_FAILURE);
			} 
			/* Read `float x` value or randomly generate if data missing */
			// Move `ptr_ch` past any whitespace, then check if the string starts with `[+-]?[0-9]+`
			while (isspace(ptr_ch[0])) {
				ptr_ch++;
			}
			// If string matches `[+-]?[0-9]+.*` after preceding whitespace, parse with `strtod`
			if (isdigit(ptr_ch[0]) || (((ptr_ch[0] == '+') || (ptr_ch[0] == '-')) && isdigit(ptr_ch[1]))) {
				// Parse and store `x` value, then update `ptr_ch` to point to end of number
				nbody_in->x[body_count] = (float)strtod(ptr_ch, &ptr_ch);
				checkLastError("Error parsing `x` data to `float`");
				// Check there are no further digits before the comma at `strchr(ptr_ch, ',')`
				if ((strpbrk(ptr_ch, "0123456789") < strchr(ptr_ch, ',')) && (strpbrk(ptr_ch, "0123456789") != NULL)) {
					fprintf(stderr, "Error reading line %u: Unexpected format when parsing `x` data to float\n", line_number);
					exit(EXIT_FAILURE);
				}
			}
			else { // Decide data missing or corrupted - means we ignore strings like ".5" and "-.2"
				nbody_in->x[body_count] = (float)rand() / RAND_MAX; // Random position in [0,1]
			}
			ptr_ch = strchr(ptr_ch, ',') + 1; // Update `ptr_ch` to start after the 1st comma
		}
		if ((strchr(ptr_ch, ',') == NULL)) { // Check for comma after second data value
			fprintf(stderr, "Error reading line %u: Only 1 data delimiter (`,`) detected\n", line_number);
			exit(EXIT_FAILURE);
		}
		else { /* Read `float y` value or randomly generate if missing */
			// Move `ptr_ch` past any whitespace, then check if the string starts with `[+-]?[0-9]+`
			while (isspace(ptr_ch[0])) {
				ptr_ch++;
			}
			// If string matches `[+-]?[0-9]+.*` after preceding whitespace, parse with `strtod`
			if (isdigit(ptr_ch[0]) || (((ptr_ch[0] == '+') || (ptr_ch[0] == '-')) && isdigit(ptr_ch[1]))) {
				// Parse and store `y` value, then update `ptr_ch` to point to end of number
				nbody_in->y[body_count] = (float)strtod(ptr_ch, &ptr_ch);
				checkLastError("Error parsing `y` data to `float`");
				// Check there are no further digits before the comma at `strchr(ptr_ch, ',')`
				if ((strpbrk(ptr_ch, "0123456789") < strchr(ptr_ch, ',')) && (strpbrk(ptr_ch, "0123456789") != NULL)) {
					fprintf(stderr, "Error reading line %u: Unexpected format when parsing `y` data to float\n", line_number);
					exit(EXIT_FAILURE);
				}
			}
			else { // Decide data missing or corrupted - means we ignore strings like ".5" and "-.2"
				nbody_in->y[body_count] = (float)rand() / RAND_MAX; // Random position in [0,1]
			}
			ptr_ch = strchr(ptr_ch, ',') + 1; // Update `ptr_ch` to start after 2nd comma
		}
		if ((strchr(ptr_ch, ',') == NULL)) { // Check for comma after third data value
			fprintf(stderr, "Error reading line %u: Only 2 data delimiters (`,`) detected\n", line_number);
			exit(EXIT_FAILURE);
		}
		else { /* Read `float vx` value or set to zero if missing */
			// Move `ptr_ch` past any whitespace, then check if the string starts with `[+-]?[0-9]+`
			while (isspace(ptr_ch[0])) {
				ptr_ch++;
			}
			// If string matches `[+-]?[0-9]+.*` after preceding whitespace, parse with `strtod`
			if (isdigit(ptr_ch[0]) || (((ptr_ch[0] == '+') || (ptr_ch[0] == '-')) && isdigit(ptr_ch[1]))) {
				// Parse and store `vx` value, then update `ptr_ch` to point to end of number
				nbody_in->vx[body_count] = (float)strtod(ptr_ch, &ptr_ch);
				checkLastError("Error parsing `vx` data to `float`");
				// Check there are no further digits before the comma at `strchr(ptr_ch, ',')`
				if ((strpbrk(ptr_ch, "0123456789") < strchr(ptr_ch, ',')) && (strpbrk(ptr_ch, "0123456789") != NULL)) {
					fprintf(stderr, "Error reading line %u: Unexpected format when parsing `vx` data to float\n", line_number);
					exit(EXIT_FAILURE);
				}
			} // Otherwise decide data is missing or corrupted - means strings like ".5" and "-.2" are ignored
			// In this case we don't change `vx` since velocity array filled with zeroes by default
			ptr_ch = strchr(ptr_ch, ',') + 1; // Update `ptr_ch` to start after 3rd comma
		}
		if ((strchr(ptr_ch, ',') == NULL)) { // Check for comma after fourth data value
			fprintf(stderr, "Error reading line %u: Only 3 data delimiters (`,`) detected\n", line_number);
			exit(EXIT_FAILURE);
		}
		else { /* Read `float vy` value or set to zero if missing */
			// Move `ptr_ch` past any whitespace, then check if the string starts with `[+-]?[0-9]+`
			while (isspace(ptr_ch[0])) {
				ptr_ch++;
			}
			// If string matches `[+-]?[0-9]+.*` after preceding whitespace, parse with `strtod`
			if (isdigit(ptr_ch[0]) || (((ptr_ch[0] == '+') || (ptr_ch[0] == '-')) && isdigit(ptr_ch[1]))) {
				// Parse and store `vx` value, then update `ptr_ch` to point to end of number
				nbody_in->vy[body_count] = (float)strtod(ptr_ch, &ptr_ch);
				checkLastError("Error parsing `vy` data to `float`");
				// Check there are no further digits before the comma at `strchr(ptr_ch, ',')`
				if ((strpbrk(ptr_ch, "0123456789") < strchr(ptr_ch, ',')) && (strpbrk(ptr_ch, "0123456789") != NULL)) {
					fprintf(stderr, "Error reading line %u: Unexpected format when parsing `vy` data to float\n", line_number);
					exit(EXIT_FAILURE);
				}
			} // Otherwise decide data is missing or corrupted - means strings like ".5" and "-.2" are ignored
			// In this case we don't change `vy` since velocity array filled with zeroes by default
			ptr_ch = strchr(ptr_ch, ',') + 1; // Update `ptr_ch` to start after 4th comma
		}
		if ((strchr(ptr_ch, ',') != NULL)) { // Ensure no more commas after fifth data value
			fprintf(stderr, "Error reading line %u: Too many data columns detected (5 expected)\n", line_number);
			exit(EXIT_FAILURE);
		}
		else { // Else read from after the 4th/last comma (`ptr_ch`) to the end of the line
		/* Read `float m` value or set to 1/N if data missing, corrupted, or zero (no massless bodies) */
			if (strtod(ptr_ch, NULL) == 0) { // If zero returned, then input data was either missing, corrupted, or zero
				fprintf(stderr, "Error reading line %u: Mass data missing, corrupted, or set to zero. Replacing with default value (1/N) to avoid massless bodies\n", line_number);
				// Set mass to 1/N to avoid creating massless objects (and divide-by-zero problems later)
				nbody_in->m[body_count] = (float)1 / N; // Mass distributed equally among N bodies
			}
			else { // Otherwise non-zero `float` value for mass read successfully, so write to `m`
				// Parse and store `m` value, then update `ptr_ch` to point to end of number
				nbody_in->m[body_count] = (float)strtod(ptr_ch, &ptr_ch);
				checkLastError("Error parsing mass data to `float`");
				if (strpbrk(ptr_ch, "0123456789") != NULL) { // Check there are no further digits before the end of the line
					fprintf(stderr, "Error reading line %u: Unexpected format when parsing mass data\n", line_number);
					exit(EXIT_FAILURE);
				}
			}
		} // One line of nbody data has been read successfully. Increment the body count.
		body_count++;
		// Read new line if not end of file. Thus data file can be terminated with single empty line.
	}
	if (body_count != N) { // Check fails when fewer than N bodies in file
		fprintf(stderr, "Error: Num bodies in file (%u) does not match input N (%d)\n", body_count, N);
		exit(EXIT_FAILURE);
	}
	fclose(f);
}

void checkLastError(const char* msg) {
	if (errno != 0) {
		perror(msg);
		print_help();
		exit(EXIT_FAILURE);
	}
}
