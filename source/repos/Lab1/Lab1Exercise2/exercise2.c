/* Lab 1 Exercise 2 program
Extend the previous exercise by implementing a better random function. 
The problem with the existing `rand` function is that it only returns values in the range of 0-32767
(the positive range of a signed short) despite returning a 32 bit integer. This is due to Microsoft
preserving backwards compatibility with code utilising the function when it was first implemented
(and when 16 bit integers were more common). This is a "feature" of the msvc runtime. */
#include <stdio.h>
// Include the header file `random.h`
#include "random.h"

// Create a pre-processor definition called `NUM_VALUES` and assign it the value of `250`.
#define NUM_VALUES 250

// 2.6.3 Modify the type of the `values` array to a 64 bit (8 byte) signed integer
// using your pre-processor definition to define the array size.
// `values` is signed so it can hold the normalised values later
signed long long int values [NUM_VALUES];


int main() {
	// 2.6.2 Our variable sum is now too small to hold the summed values.
	// Modify it to a 64 bit (8 byte) unsigned integer and ensure it is printed to the console correctly.
	unsigned long long sum = 0;

	// Define a local variable `i` in the `main` function using a data type which can 
	// hold values in the range of 0 - 255  (unsigned 8 bit = 1 byte), initialise it to `0`.
	unsigned char i = 0;

	// Call the function `init_random` (defined in `random.c`) in the `main` function
	init_random();

	for (i = 0; i < NUM_VALUES; ++i) {
		// 2.6.1 Replace the call to `random_ushort` with a call to `random_uint` in `exercise2.c`.
		// `values` holds 64 bit signed integers so it will not overflow holding a 32 bit unsigned int
		values[i] = random_uint();

		// Print statement for manual debugging
		// printf("i = %hu, value = %u\n", i, values[i]); 

		sum += values[i];
	}

	// Calculate and store the average value of the random numbers 
	// (up to integer precision) in a new local variable `average`. 
	unsigned int average = sum / NUM_VALUES;

	// Normalise the random numbers by subtracting the average. 
	// Calculate the maximum and minimum of the normalised values.
	// 2.6.4 You will also need to modify min and max to use 64 bit signed integers,
	// as the normalised values may still be outside of the range of the current 32 bit integers.
	long long min = 0;
	long long max = 0;

	for (i = 0; i < NUM_VALUES; ++i) {
		values[i] -= average;
		// To calculate the max and min of the normalized values, we use the ternary `if` operator
		// whose structure is `(conditional_expr) ? expr_true : expr_false;`
		// We update the value of `min`, `max` whenever we see a lower (resp. higher) normalised value
		// Unless all values equal (in which case `min`=`max`=`0`), `min` will be negative, `max` positive
		min = (values[i] < min) ? values[i] : min;
		max = (values[i] > max) ? values[i] : max;
	}
	// Print the `sum`, `average`, and normalised `min` and `max` values.
	// 2.6.5 Ensure that your `printf` formats are correct for the data types.
	/* See https ://www.geeksforgeeks.org/data-types-in-c/ for more on data types and formatting in C */
	printf("We calculated %u unsigned 32 bit integers using a linear congruential generator\n", NUM_VALUES);
	printf("to simulate sampling from a discrete uniform distribution over 0,1,...,4294967295.\n");
	printf("The sum of our sample is %llu\n", sum);
	printf("The average of our sample is %u\n", average);
	printf("After subtracting the average, the minimum normalised value is %lld\n", min);
	printf("The maximum normalised value is %lld\n", max);
	return 0;
}
