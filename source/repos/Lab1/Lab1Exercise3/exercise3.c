/* Lab 1 Exercise 3 program 
Extend the previous exercise by implementing a floating point random function. 
3.1 Add a new function definition and declaration (`random_float`) returning a random `float`. 
Modify the example so that floating point values are calculated for `sum`, `average`, `min` and `max`. 
Ensure that the values are printed with `0` decimal places. What is the sum, average, min and max? */

#include <stdio.h>
// Include the header file `random.h`
#include "random.h"

// Create a pre-processor definition called `NUM_VALUES` and assign it the value of `250`.
#define NUM_VALUES 250

// Modify the type of the `values` array to `float`,
// using your pre-processor definition for `NUM_VALUES` to define the array size.
float values [NUM_VALUES];

int main() {
	float sum = 0;

	// Define a local variable `i` in the `main` function using a data type which can 
	// hold values in the range of 0 - 255  (unsigned 8 bit = 1 byte), initialise it to `0`.
	unsigned char i = 0;

	// Call the function `init_random` (defined in `random.c`) in the `main` function
	init_random();

	for (i = 0; i < NUM_VALUES; ++i) {
		// Replace the call to `random_uint` with a call to `random_float`.
		values[i] = random_float();

		// Print statement for manual debugging
		// printf("i = %hu, value = %f\n", i, values[i]); 

		sum += values[i];
	}

	// Calculate, store the average value of the random numbers in a local variable called `average`. 
	float average = sum / NUM_VALUES;

	// Normalise the random numbers by subtracting the average. 
	// Calculate the maximum and minimum of the normalised values.
	float min = 0;
	float max = 0;

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
	// Ensure that your `printf` formats are correct for the data types.
	// Ensure that the values are printed with 0 decimal places.
	/* See https://www.geeksforgeeks.org/data-types-in-c/ for more on data types and formatting in C */
	printf("To simulate sampling  %u values from a discrete uniform distribution over 0,1,...,4294967295.\n", NUM_VALUES);
	printf("We sampled unsigned 32 bit integers using an LCG (then cast as floating point numbers, print to integer precision).\n");
	printf("The sum of our sample is %.0f\n", sum);
	printf("The average of our sample is %.0f\n", average);
	printf("After subtracting the average, the minimum normalised value is %.0f\n", min);
	printf("The maximum normalised value is %.0f\n", max);
	return 0;
}
