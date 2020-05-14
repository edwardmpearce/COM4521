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
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h> // http://www.cplusplus.com/reference/cstdio/
#include <stdlib.h> // http://www.cplusplus.com/reference/cstdlib/
#include <string.h> // http://www.cplusplus.com/reference/cstring/
#include <ctype.h> // http://www.cplusplus.com/reference/cctype/
#include <time.h> // http://www.cplusplus.com/reference/ctime/
#include <math.h> // http://www.cplusplus.com/reference/cmath/
#include "NBody.h"
#include "NBodyVisualiser.h"

#define USER_NAME "smp16emp" // Replace with your username
#define BUFFER_SIZE 128 // Maximum line length accepted from input file (reasonable as only 5 (comma separated) floating point numbers expected)
void parseNDM(char* argv[]);
void parse_one_option(char* argv[]);
void parse_two_options(char* argv[]);
int parse_str_as_int(char* str);
void read_nbody_file(char* filename, int N);
void print_help();
void checkLastError(const char* msg);
void step(void);

int N; // Number of bodies in the system
int D; // Dimension of the activity grid
MODE M; // Operation mode. Allows CPU = 0, OPENMP = 1, CUDA = 2, but CUDA mode rendering not supported in part 1
int I = 0; // Number of iterations of the simulation to calculate when the `-i` flag is set, else 0
unsigned char f_flag = 0; // Input file flag. 0 if not specified, else such that `input_filename = options[f_flag]` in `main`.

nbody* nbody_in; // Pointer to a structure of arrays (preferred over an array of structures for coalesced memory access)
nbody* nbody_out; // Double buffering to enable pointer swapping to reduce memory copying when updating system state
float* activity_map; // Pointer to flattened array of D*D float values storing normalised particle density values in a 2D grid

/* For information on how to parse command line parameters, see http://www.cplusplus.com/articles/DEN36Up4/ 
`argc` in the count of the command arguments, and `argv` is an array (of length `argc`) of the arguments. 
The first argument is always the executable name (including path)*/
int main(int argc, char *argv[]) {
	/* Process the command line arguments */
	switch (argc) {
	case 4: // No optional flags used
		parseNDM(argv);
		break;
	case 6: // One optional flag and argument used
		parse_one_option(&argv[4]);
		parseNDM(argv);
		break;
	case 8: // Two optional flags with arguments used
		parse_two_options(&argv[4]);
		parseNDM(argv);
		break;
	default: // The expected arguments are: "nbody.exe N D M [-i I] [-f input_file]"
		fprintf(stderr, "Error: Unexpected number of arguments. %d arguments (including executable name) received\n", argc);
		print_help();
		exit(EXIT_FAILURE);
	}

	/* Allocate Heap Memory */
	// Calculate memory requirements
	//unsigned int nbody_system_size = sizeof(float) * N * 5;
	unsigned int data_column_size = sizeof(float) * N;
	unsigned int activity_grid_size = sizeof(float) * D * D;

	// Memory allocation. See http://www.cplusplus.com/reference/cstdlib/malloc/
	nbody_in = (nbody*)malloc(sizeof(nbody));
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
	nbody_out = (nbody*)malloc(sizeof(nbody));
	nbody_out->x = (float*)malloc(data_column_size);
	nbody_out->y = (float*)malloc(data_column_size);
	nbody_out->vx = (float*)malloc(data_column_size);
	nbody_out->vy = (float*)malloc(data_column_size);
	nbody_out->m = (float*)malloc(data_column_size);
	if ((nbody_out == NULL) || (nbody_out->x == NULL) || (nbody_out->y == NULL) || (nbody_out->vx == NULL) || (nbody_out->vy == NULL) || (nbody_out->m == NULL)) {
		fprintf(stderr, "Error allocating host memory (`nbody_out`) for system with %d bodies\n", N);
		exit(EXIT_FAILURE);
	}
	activity_map = (float*)malloc(activity_grid_size);
	if (activity_map == NULL) {
		fprintf(stderr, "Error allocating host memory (`activity map`) for system with %d bodies, grid size %d\n", N, D);
		exit(EXIT_FAILURE);
	}

	/* Read initial data from file, or generate random initial state according to optional program flag `-f`. */
	if (f_flag == 0) { // No input file specified, so a random initial N-body state will be generated
		for (int i = 0; i < N; i++) {
			nbody_in->x[i] = (float)rand() / RAND_MAX; // Random position in [0,1]
			nbody_in->y[i] = (float)rand() / RAND_MAX; // Random position in [0,1]
			nbody_in->m[i] = (float) 1/N; // Mass distributed equally among N bodies
			// Note that velocity data has already been initialized to zero for all bodies
		}
	}
	else { // Attempt to read initial N-body system state from input csv file
		read_nbody_file(argv[f_flag], N);
	}

	/* According to the value of program argument `I` either configure and start the visualiser, 
	or perform a fixed number of simulation steps and output the timing results. */
	if (I == 0) { // Run visualiser when number of iterations not specified with `-i` flag, or otherwise `I` was set to 0
		initViewer(N, D, M, &step);
		setNBodyPositions(nbody_in);
		setActivityMapData(activity_map);
		startVisualisationLoop();
	}
	else { // Run the simulation for `I` iterations and output the timing results
		// Declare timing variables
		clock_t begin, end;	// Timestamps
		double seconds;

		begin = clock();
		for (int i = 0; i < I; i++) {
			step();
		}
		end = clock();
		seconds = ((double)end - begin) / CLOCKS_PER_SEC;
		printf("Execution time %d seconds %d milliseconds for %d iterations\n", (int)seconds, (int)(seconds-(int)seconds)*1000, I);
	}

	// Cleanup
	free(nbody_in->x);
	free(nbody_in->y);
	free(nbody_in->vx);
	free(nbody_in->vy);
	free(nbody_in->m);
	free(nbody_out->x);
	free(nbody_out->y);
	free(nbody_out->vx);
	free(nbody_out->vy);
	free(nbody_out->m);
	free(nbody_in);
	free(nbody_out);
	free(activity_map);

	return 0;
}

void step(void) {
	//TODO: Perform the main simulation of the NBody system
	// Update the state of the Nbody system

	// Update the activity map
	
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

void parseNDM(char* argv[]) {
	N = parse_str_as_int(argv[1]);
	checkLastError("Error parsing argument for `N` to `int`");
	D = parse_str_as_int(argv[2]);
	checkLastError("Error parsing argument for `D` to `int`");
	if (strcmp(argv[3], "CPU") == 0) {
		M = CPU;
	}
	else if (strcmp(argv[3], "OPENMP") == 0) {
		M = OPENMP;
	}
	else if (strcmp(argv[3], "CUDA") == 0) {
		M = CUDA;
	}
	else {
		fprintf(stderr, "Error: Unexpected value %s for operation mode `M` (case sensitive).\n", argv[3]);
		print_help();
		exit(EXIT_FAILURE);
	}
}

void parse_one_option(char* options[]) {
	if (strcmp(options[0], "-i") == 0) {
		I = parse_str_as_int(options[1]);
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

void parse_two_options(char* options[]) {
	if ((strcmp(options[0], "-i") == 0) && (strcmp(options[2], "-f") == 0)) {
		I = parse_str_as_int(options[1]);
		checkLastError("Error parsing argument for `I` to `int`");
		f_flag = 7;
	}
	else if ((strcmp(options[0], "-f") == 0) && (strcmp(options[2], "-i") == 0)) {
		I = parse_str_as_int(options[3]);
		checkLastError("Error parsing argument for `I` to `int`");
		f_flag = 5;
	}
	else { // Invalid option flag combination
		fprintf(stderr, "Error: Unexpected combination of optional arguments/flags received.\n");
		print_help();
		exit(EXIT_FAILURE);
	}
}

int parse_str_as_int(char * str) {
	if (isdigit(str[0]) == 0) { // In particular, this excludes leading minus sign/negative input values.
		fprintf(stderr, "Error parsing %s as `int`: First char not decimal digit (negative values not permitted).\n", str);
		print_help();
		exit(EXIT_FAILURE);
	}
	int val; // To hold parsed `int` value
	char* pEnd; // Pointer to first character after number in `str`
	val = strtol(str, &pEnd, 10); // Convert string to long integer in base 10. Set `pEnd`.
	if (pEnd[0] != '\0') { // Check for extra characters in `str` after initial number (can include decimal point)
		fprintf(stderr, "Error: Unexpected characters in string %s when parsing to `int`.\n", str);
		print_help();
		exit(EXIT_FAILURE);
	} 
	return val;
}

/* Functions for reading input files */
void read_nbody_file(char* filename, int N) {
	FILE* f; // Input file handle
	char line_buffer[BUFFER_SIZE]; // Buffer to hold lines read from file
	char* ptr_ch = NULL; // Pointer to track character position when reading `line_buffer` string
	unsigned int line_number = 0; // Keep track of line number for error messaging
	unsigned int body_count = 0; // Count of number of body data lines read to ensure it matches `N`
	unsigned int comma_count = 0; // Used to check data is formatted correctly

	f = fopen(filename, "r"); // Open the file in read-only mode
	if (f == NULL) {
		fprintf(stderr, "Error opening file '%s' for reading\n", filename);
		exit(EXIT_FAILURE);
	}

	/* Read file line by line with `fgets` function. See http://www.cplusplus.com/reference/cstdio/fgets/ for reference
	Reads from file into buffer until (soonest of) either `\n` or `EOF` is read, or `BUFFER_SIZE-1` characters read */
	while (fgets(line_buffer, BUFFER_SIZE, f) != NULL) {
		line_number++; // Increment count of lines read
		if (line_buffer[0] == "#") { // If first char in line is `#` skip to next line to ignore comments
			continue;
		}
		if (line_buffer[strlen(line_buffer) - 1] != "\n") { // If last char read from file is not "\n", the line is too long
			// This checks that a full line of data was written from file to buffer when not a comment line
			fprintf(stderr, "Error reading line %ud: Line length exceeds buffer size of %d characters\n", line_number, BUFFER_SIZE);
			exit(EXIT_FAILURE);
		}

		/* Read the line of data into the `nbody` data structure referenced by `nbody_in`
		Use the `,` character as a delimiter to separate data values, whilst keeping count to ensure correct format */
		ptr_ch = line_buffer; // Place `ptr_ch` at the start of the line to be read
		// Just unroll this while loop to make this less confusing since it is hard to escape the loop otherwise?
		while ((strchr(ptr_ch, ',') != NULL) || (comma_count == 4)) {
			/* Use `strchr` to search through the line starting at position `ptr_ch` to find the next comma `,` character
			Returns `NULL` pointer if no comma `,` character found in line after position `ptr_ch`
			See http://www.cplusplus.com/reference/cstring/strchr/ for reference */
			// Either a comma was detected or 4 commas read (special case)
			switch (comma_count) {
			case 0: /* Read `float x` value or randomly generate if data missing */
				// Move `ptr_ch` past any whitespace, then check if the string starts with `[+-]?[0-9]+`
				while (isspace(ptr_ch[0])) {
					ptr_ch++;
				}
				// If string matches `[+-]?[0-9]+.*` after preceding whitespace, parse with `strtod`
				if (isdigit(ptr_ch[0]) || (((ptr_ch[0] == "+") || (ptr_ch[0] == "-")) && isdigit(ptr_ch[1]))) {
					// Parse and store `x` value, then update `ptr_ch` to point to end of number
					nbody_in->x[body_count] = strtod(ptr_ch, &ptr_ch);
					// Check there are no further digits before the comma at `strchr(ptr_ch, ',')`
					if ((strpbrk(ptr_ch, "0123456789") < strchr(ptr_ch, ',')) && (strpbrk(ptr_ch, "0123456789") != NULL)) {
						fprintf(stderr, "Error reading line %ud: Unexpected format when parsing string to float\n", line_number);
						exit(EXIT_FAILURE);
					}
				}
				else { // Decide data missing or corrupted - means we ignore strings like ".5" and "-.2"
					nbody_in->x[body_count] = (float)rand() / RAND_MAX; // Random position in [0,1]
				}
				break;
			case 1: /* Read `float y` value or randomly generate if missing */
				// Move `ptr_ch` past any whitespace, then check if the string starts with `[+-]?[0-9]+`
				while (isspace(ptr_ch[0])) {
					ptr_ch++;
				}
				// If string matches `[+-]?[0-9]+.*` after preceding whitespace, parse with `strtod`
				if (isdigit(ptr_ch[0]) || (((ptr_ch[0] == "+") || (ptr_ch[0] == "-")) && isdigit(ptr_ch[1]))) {
					// Parse and store `y` value, then update `ptr_ch` to point to end of number
					nbody_in->y[body_count] = strtod(ptr_ch, &ptr_ch);
					// Check there are no further digits before the comma at `strchr(ptr_ch, ',')`
					if ((strpbrk(ptr_ch, "0123456789") < strchr(ptr_ch, ',')) && (strpbrk(ptr_ch, "0123456789") != NULL)) {
						fprintf(stderr, "Error reading line %ud: Unexpected format when parsing string to float\n", line_number);
						exit(EXIT_FAILURE);
					}
				}
				else { // Decide data missing or corrupted - means we ignore strings like ".5" and "-.2"
					nbody_in->y[body_count] = (float)rand() / RAND_MAX; // Random position in [0,1]
				}
				break;
			case 2: /* Read `float vx` value or set to zero if missing */
				// Move `ptr_ch` past any whitespace, then check if the string starts with `[+-]?[0-9]+`
				while (isspace(ptr_ch[0])) {
					ptr_ch++;
				}
				// If string matches `[+-]?[0-9]+.*` after preceding whitespace, parse with `strtod`
				// Otherwise decide data is missing or corrupted - means we ignore strings like ".5" and "-.2"
				if (isdigit(ptr_ch[0]) || (((ptr_ch[0] == "+") || (ptr_ch[0] == "-")) && isdigit(ptr_ch[1]))) {
					// Parse and store `y` value, then update `ptr_ch` to point to end of number
					nbody_in->vx[body_count] = strtod(ptr_ch, &ptr_ch);
					// Check there are no further digits before the comma at `strchr(ptr_ch, ',')`
					if ((strpbrk(ptr_ch, "0123456789") < strchr(ptr_ch, ',')) && (strpbrk(ptr_ch, "0123456789") != NULL)) {
						fprintf(stderr, "Error reading line %ud: Unexpected format when parsing string to float\n", line_number);
						exit(EXIT_FAILURE);
					}
				} // Do nothing in `else` case since velocity array filled with zeroes by default
				break;
			case 3: /* Read `float vy` value or set to zero if missing */
				// Move `ptr_ch` past any whitespace, then check if the string starts with `[+-]?[0-9]+`
				while (isspace(ptr_ch[0])) {
					ptr_ch++;
				}
				// If string matches `[+-]?[0-9]+.*` after preceding whitespace, parse with `strtod`
				// Otherwise decide data is missing or corrupted - means we ignore strings like ".5" and "-.2"
				if (isdigit(ptr_ch[0]) || (((ptr_ch[0] == "+") || (ptr_ch[0] == "-")) && isdigit(ptr_ch[1]))) {
					// Parse and store `y` value, then update `ptr_ch` to point to end of number
					nbody_in->vy[body_count] = strtod(ptr_ch, &ptr_ch);
					// Check there are no further digits before the comma at `strchr(ptr_ch, ',')`
					if ((strpbrk(ptr_ch, "0123456789") < strchr(ptr_ch, ',')) && (strpbrk(ptr_ch, "0123456789") != NULL)) {
						fprintf(stderr, "Error reading line %ud: Unexpected format when parsing string to float\n", line_number);
						exit(EXIT_FAILURE);
					}
				} // Do nothing in `else` case since velocity array filled with zeroes by default
				break;
			case 4: // Read `float mass` value or set to 1/N if data missing, corrupted, or zero (no massless bodies)
				if (strchr(ptr_ch, ',') != NULL) { // Runs if the total number of commas on the line is more than 4
					fprintf(stderr, "Error reading line %ud: Too many columns (5 expected)\n", line_number);
					exit(EXIT_FAILURE);
				} // Else read from after the last comma (`ptr_ch`) to the end of the line
				if (strtod(ptr_ch, NULL) == 0) { // If zero returned, then input data was either missing, corrupted, or zero
					fprintf(stderr, "Error reading line %ud: Mass data missing, corrupted, or set to zero. Replacing with default value (1/N)\n", line_number);
					nbody_in->m[body_count] = (float)1 / N; // Mass distributed equally among N bodies
				}
				else { // This avoids creating massless objects (and divide-by-zero problems later)
					nbody_in->m[body_count] = strtod(ptr_ch, &ptr_ch);
					if (strpbrk(ptr_ch, "0123456789") != NULL) { // Check there are no further digits before the end of the line
						fprintf(stderr, "Error reading line %ud: Unexpected format when parsing mass data\n", line_number);
						exit(EXIT_FAILURE);
					}
				}
				break;
			}
			comma_count++; // Increment comma count after reading a piece of data
			ptr_ch = strchr(ptr_ch, ',') + 1; // Update `ptr_ch` to start after the comma just found



		}
		if (comma_count != 4) { // Check fails when too few comma `,` delimiters detected in line
			fprintf(stderr, "Error reading line %ud: %ud delimiters detected (5 expected)\n", line_number, comma_count);
			exit(EXIT_FAILURE);
		}
		// Don't write past memory bounds for nbody_in!!!
		if (++body_count >= N) { // Increment body count, and throw an error if this exceeds N
			fprintf(stderr, "Error: Num bodies in file exceeds input N (%d)\n", N);
			exit(EXIT_FAILURE);
		}
		comma_count = 0; // Reset comma count for reading next line

		// Split line into 5 pieces of data (strings, could be empty or whitespace)
		// Fill missing data with default values
		// Parse data as float values carefully with `strtod` function
		// Write data into allocated Nbody structure of arrays, increment body count

	}
	if (body_count != N) { // Check fails when fewer than N bodies in file
		fprintf(stderr, "Error: Num bodies in file (%ud) does not match input N (%d)\n", body_count, N);
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
