/* Lab 1 Exercise 1 program */
/* Creates a list of normalised random integers */
/* See https://www.geeksforgeeks.org/data-types-in-c/ for more on data types in C */
#include <stdio.h>
// 1.5 Include the header file `random.h`
#include "random.h"

// 1.1 Create a pre-processor definition called `NUM_VALUES` and assign it the value of `250`.
#define NUM_VALUES 250

// 1.2 Declare a global signed 32-bit (4 byte) integer array called `values`
// using your pre-processor definition to define the array size.
signed int values [NUM_VALUES];

int main() {
	// 1.3 Define a local unsigned 32-bit (4 byte) integer variable called `sum` in the main function 
	// capable of holding only positive values and initialise it to `0`.
	unsigned int sum = 0;

	// 1.4 Define a local variable `i` in the `main` function using a data type which can 
	// hold values in the range of 0 - 255  (unsigned 8 bit = 1 byte), initialise it to `0`.
	unsigned char i = 0;

	// 1.5 Call the function `init_random` (defined in `random.h`) in the `main` function
	init_random();

	// 1.6 Write a simple `for` loop (using the integer `i` as a counter) in the range of `0` and `NUM_VALUES`.
	for (i = 0; i < NUM_VALUES; ++i) {
		// 1.6 Within the loop make a call to the function `random_ushort` and 
		// save the value in the `values` array at index `i`.
		values[i] = random_ushort();

		// 1.6 Within the loop create a print statement to the console which outputs in a
		// single line the value of `i` and the value you have stored in the array.
		// We can use this to debug the output.
		// printf("i = %hu, value = %hu\n", i, values[i]); 

		// 1.8 Modify your loop by commenting out the debug statement and 
		// summing the value into the variable `sum`.
		sum += values[i];
	}

	// 1.8 Calculate and store the average value of the random numbers 
	// (up to integer precision) in a new local variable `average`. 
	unsigned int average = sum / NUM_VALUES;

	// 1.9 Normalise the random numbers by subtracting the average. 
	// Calculate the maximum and minimum of the normalised values.

	int min = 0;
	int max = 0;

	for (i = 0; i < NUM_VALUES; ++i) {
		values[i] -= average;
		// To calculate the max and min of the normalized values, we use the ternary `if` operator
		// whose structure is `(conditional_expr) ? expr_true : expr_false;`
		// We update the value of `min`, `max` whenever we see a lower (resp. higher) normalised value
		// Unless all values equal (in which case `min`=`max`=`0`), `min` will be negative, `max` positive
		min = (values[i] < min) ? values[i] : min;
		max = (values[i] > max) ? values[i] : max;
	}
	// 1.9 Print the `sum`, `average`, and normalised `min` and `max` values.
	printf("We sampled %u values from discrete uniform distribution over 0,1,...,32767.\n", NUM_VALUES);
	printf("The sum of our sample is %u\n", sum);
	printf("The average of our sample is %u\n", average);
	printf("After subtracting the average, the minimum normalised value is %d\n", min);
	printf("The maximum normalised value is %d\n", max);

	// 1.8 Output the sum value after the loop has returned.
	return sum;
}
