/* Lab 3 Exercise 1 Program
We are going to start with the matrix multiplication code from the previous lab 
to see what effect OpenMP has on improving the performance. 
Set 'OpenMP Support' to 'Yes' (for both Debug and Release builds) in Project->Properties->C/C++->Language 
Add `_CRT_SECURE_NO_WARNINGS` to 'Preprocessor Definitions' in Project->Properties->C/C++->Preprocessor  */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
/* To enable OpenMP support in your project you will need to include the OpenMP header file
`omp.h` and enable the compiler to use the OpenMP runtime. */
#include <omp.h>

#define N 1024							// Number of rows/columns in our randomly generated matrices
typedef double element_type;			// The data type of the matrix elements
typedef element_type** matrixNN;		// Use 2-dimensional pointer to represent an N by N matrix
// Function declarations
void init_random_matrix(matrixNN m);
void init_zero_matrix(matrixNN m);
void write_matrix_to_file(const char *filename, const matrixNN r);
void transpose(matrixNN t);
void multiply(matrixNN r, const matrixNN a, const matrixNN b);

/* Execute the program with and without parallelisation and compare the outputs
using the Windows `FC` file comparison command (similar to `unix diff`) in a terminal
to ensure our results are consistent and we haven't made any mistakes in parallelisation.
This will print any file differences (you will need to name the output files differently/give different paths).
After verifying that the correct output is produced after each modification of the code, we record performance results below
| Machine | Optimisation | Execution time(s) | Timing method |
| :-----: | :----------: | :---------------: | :-----------: |
| Laptop  | Serial       | 1.93s, 1.96s      | `clock()` |
| Laptop  | Serial       | 1.92s, 1.91s      | `omp_get_wtime()` |
| Laptop  | Parallel     | 0.59s, 0.57s      | `omp_get_wtime()` | 4 threads
| Library Desktop | Serial   | 1.04s         | `omp_get_wtime()` |
| Library Desktop | Parallel | 0.11s         | `omp_get_wtime()` | */
void main(){
	// Variable declarations
	double begin, end;
	double seconds;
	matrixNN a;
	matrixNN b;
	matrixNN c;
	int i;	// Iteration variable

	// For each matrix, allocate memory for the pointers to each row, then allocate the memory for the elements in each row
	a = (matrixNN)malloc(sizeof(element_type) * N);
	for (i = 0; i < N; i++)
		a[i] = (element_type*)malloc(sizeof(element_type) * N);
	b = (matrixNN)malloc(sizeof(element_type) * N);
	for (i = 0; i < N; i++)
		b[i] = (element_type*)malloc(sizeof(element_type) * N);
	c = (matrixNN)malloc(sizeof(element_type) * N);
	for (i = 0; i < N; i++)
		c[i] = (element_type*)malloc(sizeof(element_type) * N);

	init_random_matrix(a);
	init_random_matrix(b);
	init_zero_matrix(c);

	int max_threads = omp_get_max_threads();
	printf("OpenMP using %d threads\n", max_threads);

	begin = omp_get_wtime();

	// Calculate the matrix product of `a` and `b` and write the result to `c`
	multiply(c, a, b);

	end = omp_get_wtime();
	seconds = end - begin;
	printf("Matrix multiply complete in %.2f seconds\n", seconds);

	// Write the results matrix `c` to the file specified below
	printf("Writing results...\n");
	write_matrix_to_file("matrix_mul.txt", c);
	printf("Done writing results\n");

	// Free the memory allocated for each matrix
	for (i = 0; i < N; i++)
		free(a[i]);
	free(a);
	for (i = 0; i < N; i++)
		free(b[i]);
	free(b);
	for (i = 0; i < N; i++)
		free(c[i]);
	free(c);
}

void init_random_matrix(matrixNN m) {
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			//m[i][j] = rand() % 100; // For randomly generated integers between 0 and 99
			m[i][j] = rand() / (element_type)RAND_MAX; // Normalize for `float` or `double` numbers between 0 and 1
		}
	}
}

void init_zero_matrix(matrixNN m) {
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			m[i][j] = 0;
		}
	}
}

void transpose(matrixNN t) {
	int i, j;
	element_type temp;
	// Iterate over the upper triangle of the matrix, swapping elements to transpose the matrix in place (saving memory)
	for (i = 0; i < N; i++) {
		for (j = i + 1; j < N; j++) {
			temp = t[i][j];
			t[i][j] = t[j][i];
			t[j][i] = temp;
		}
	}
}

/* 1.1 We will parallelise the outer loop (within the `multiply` function). 
Create a directive to parallelise over the outer loop. 
Run your parallelised code and compare the text file output to the original (serial version) 
using the file compare command `FC` in a Windows terminal.
1.2  Set the OpenMP clause `default(none)`. This will give a compiler error for any variables 
which you have not explicitly defined the scope. Now try defining the scope for all variables of the parallel block. 
This should achieve both both a speedup and return the correct result
The variable `i` is the parallel loop counter so is implicitly defined as `private`.
The variables`a` and `b` are `const` so are implicitly `shared`. */
void multiply(matrixNN r, const matrixNN a, const matrixNN b){
	int i, j, k;
	element_type temp;	// Variable to hold the sum in the calculation of each entry of the matrix product
	transpose(b);		// Transpose the matrix inplace so that we can access entries by row during multiplication
// Define the scope for all variables of the parallel block. `private` to each thread, vs. `shared` between threads
#pragma omp parallel for default(none) private(i, j, k, temp) shared(r, a, b)
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			temp = 0;
			for (k = 0; k < N; k++){
				// Note that we access the transposed matrix `b` by rows
				temp += a[i][k] * b[j][k];
			}
			r[i][j] = temp;
		}
	}
}

void write_matrix_to_file(const char* filename, const matrixNN r) {
	FILE* f;
	int i, j;

	f = fopen(filename, "w");
	if (f == NULL) {
		fprintf(stderr, "Error opening file '%s' for write\n", filename);
		return;
	}
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			fprintf(f, "%0.2f\t", r[i][j]);
		}
		fprintf(f, "\n");
	}
	fclose(f);
}
