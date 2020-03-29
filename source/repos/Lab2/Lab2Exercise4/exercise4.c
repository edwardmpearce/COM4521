/* Lab 2 Exercise 4 Program
A program has been provided for you that performs matrix multiplication and writes the result out to a text file. 
The function `multiply_A` is the most compute intensive function of the program. 
You can confirm this by running the visual studio profiler if you wish. Timing code has been written to profile the function. 
Execute the program after various improvements and compare the output to the original for verification
using the Windows `FC` file comparison command (similar to `unix diff`) in a terminal.
This will print any file differences (you will need to name the output files differently/give different paths). 
After verifying that the correct output is produced after each modification of the code, we record performance results below 
| Function | Optimisation | Execution time(s) |
| :------: | :----------: | :---------------: |
| `multiply_A` | None | 14.32s, 13.53s |
| `multiply_B` | Write to local variable (avoiding unnecessary memory accesses) | 10.95s, 11.24s |
| `multiply_B` | Run in `Release` mode (`O2` compiler optimisation) | 5.41s, 5.73s |
| `multiply_C` | Matrix transpose (for row-wise access) | 1.93s, 1.96s |
| `multiply_C_unrolled` | Loop unrolling | 1.97s, 1.93s | */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 1024							// Number of rows/columns in our randomly generated matrices
typedef double element_type;			// The data type of the matrix elements
typedef element_type** matrixNN;		// Use 2-dimensional pointer to represent an N by N matrix
// Function declarations
void init_random_matrix(matrixNN m);
void init_zero_matrix(matrixNN m);
void write_matrix_to_file(const char* filename, const matrixNN r);
void multiply_A(matrixNN r, const matrixNN a, const matrixNN b);	// The pointer `r` is where the result will be written
void multiply_B(matrixNN r, const matrixNN a, const matrixNN b);
void transpose(matrixNN t);											// Transpose a matrix in place (to preserve memory) 
void multiply_C(matrixNN r, const matrixNN a, const matrixNN b);
void multiply_C_unrolled(matrixNN r, const matrixNN a, const matrixNN b);	// Unroll some loops for performance speedup

/* We are going to optimise this function by performing a number of changes to it.
4.1 The innermost loop writes directly to `r[i][j]`. This is unnecessary and will cause the program to be memory bound. 
Create a copy of `multiply_A` and call it `multiply_B`. Modify the inner loop to write to a local variable 
and make only a single write to `r[i][j]` at the end of the inner loop. Compile and execute your program and record the result.
4.2 Change the build mode to `Release`. This will set the compiler optimisation to `O2` (Maximum speed). 
Confirm this in the project properties and then compile and execute. Note your results in the table.
4.3 The innermost loop of the matrix multiply code access both matrix `a` and `b`. One of these matrices is accessed row wise
and the other column wise. To avoid a column wise access we can transpose the matrix. 
Write a function `transpose` which swaps elements `[i][j]` with `[j][i]`.
Transpose the column accessed matrix and then create a copy of `multiply_B` called `multiply_C` where you can 
update your code so that both matrices are accessed using a row wise pattern. Compile, execute and record the result. 
How is the performance now? Not a bad speed up huh? The compiler has most likely already performed some loop unrolling for us.
4.4 You could try loop unrolling yourself to see if this improves performance.
If it does not, then the compiler will have done this for us. */
void main() {
	// Variable declarations
	clock_t begin, end;	// Timestamps
	double seconds;
	matrixNN a;
	matrixNN b;
	matrixNN c;
	int i;				// Iteration variable

	// For each matrix, allocate memory for the pointers to each row, then allocate the memory for the elements in each row
	a = (matrixNN)malloc(sizeof(element_type*) * N);
	for (i = 0; i < N; i++)
		a[i] = (element_type*)malloc(sizeof(element_type) * N);
	b = (matrixNN)malloc(sizeof(element_type*) * N);
	for (i = 0; i < N; i++)
		b[i] = (element_type*)malloc(sizeof(element_type) * N);
	c = (matrixNN)malloc(sizeof(element_type*) * N);
	for (i = 0; i < N; i++)
		c[i] = (element_type*)malloc(sizeof(element_type) * N);

	init_random_matrix(a);
	init_random_matrix(b);
	init_zero_matrix(c);

	begin = clock();

	// Calculate the matrix product of `a` and `b` and write the result to `c`
	multiply_C_unrolled(c, a, b);

	end = clock();
	seconds = ((double) end - begin) / CLOCKS_PER_SEC;
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

void multiply_A(matrixNN r, const matrixNN a, const matrixNN b) {
	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			r[i][j] = 0;
			for (k = 0; k < N; k++) {
				// Repeated access to `r[i][j]` during the summation causes this function to be bound by memory access latency
				r[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

/* 4.1 The innermost loop writes directly to `r[i][j]`. This is unnecessary and will cause the program to be memory bound. 
Create a copy of `multiply_A` and call it `multiply_B`. Modify the inner loop to write to a local variable 
and make only a single write to `r[i][j]` at the end of the inner loop. Compile and execute your program and record the result.
4.2 Change the build mode to `Release`. This will set the compiler optimisation to `O2` (Maximum speed). 
Confirm this in the project properties and then compile and execute. Note your results in the table. */
void multiply_B(matrixNN r, const matrixNN a, const matrixNN b) {
	int i, j, k;
	element_type temp;	// Variable to hold the sum in the calculation of each entry of the matrix product
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			temp = 0;
			for (k = 0; k < N; k++) {
				temp += a[i][k] * b[k][j];
			}
			r[i][j] = temp;
		}
	}
}

/* 4.3 The innermost loop of the matrix multiply code accesses both matrix `a` and `b`. One of these matrices is 
accessed row wise and the other column wise. To avoid a column wise access we can transpose the matrix. 
Write a function `transpose` which swaps elements `[i][j]` with `[j][i]`.
Transpose the column accessed matrix and then create a copy of `multiply_B` called `multiply_C` where you can 
update your code so that both matrices are accessed using a row wise pattern. Compile, execute and record the result. */
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

void multiply_C(matrixNN r, const matrixNN a, const matrixNN b) {
	int i, j, k;
	element_type temp;	// Variable to hold the sum in the calculation of each entry of the matrix product
	transpose(b);		// Transpose the matrix inplace so that we can access entries by row during multiplication
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			temp = 0;
			for (k = 0; k < N; k++) {
				// Note that we access the transposed matrix `b` by rows
				temp += a[i][k] * b[j][k];
			}
			r[i][j] = temp;
		}
	}
}

/* 4.4 The compiler has most likely already performed some loop unrolling for us.
You could try loop unrolling yourself to see if this improves performance.
If it does not, then the compiler will have done this for us. */
void multiply_C_unrolled(matrixNN r, const matrixNN a, const matrixNN b) {
	int i, j, k;
	element_type temp;	// Variable to hold the sum in the calculation of each entry of the matrix product
	transpose(b);		// Transpose the matrix inplace so that we can access entries by row during multiplication
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			temp = 0;
			// This level of loop unrolling assumes/requires that `N` will be a multiple of 8
			for (k = 0; k < N; k += 8) {
				// Note that we access the transposed matrix `b` by rows
				temp += a[i][k] * b[j][k];
				temp += a[i][k + 1] * b[j][k + 1];
				temp += a[i][k + 2] * b[j][k + 2];
				temp += a[i][k + 3] * b[j][k + 3];
				temp += a[i][k + 4] * b[j][k + 4];
				temp += a[i][k + 5] * b[j][k + 5];
				temp += a[i][k + 6] * b[j][k + 6];
				temp += a[i][k + 7] * b[j][k + 7];
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
}
