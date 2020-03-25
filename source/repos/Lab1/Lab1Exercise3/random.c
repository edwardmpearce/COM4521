#include <stdlib.h>
#include "random.h"

// Create a 32 bit (4 byte) unsigned global integer variable called `rseed` in `random.c`. 
// Modify `init_random` to set the value of `rseed` to `RAND_SEED`.
unsigned int rseed;

void init_random() {
	srand(RAND_SEED);
	rseed = RAND_SEED;
}

unsigned short random_ushort() {
	// Explicitly cast the output of the `rand` function from `int` to `unsigned short`.
	return (unsigned short)rand();
}

/* Create a function definition (in `random.c`) for a function `random_uint` returning an `unsigned int`.
 Implement a linear generator where `x` can use the variable `rseed` with an initial value `x0 = RAND_SEED`, 
 and parameters `RANDOM_A` and `RANDOM_C`. */
unsigned int random_uint() {
	rseed = RANDOM_A * rseed + RANDOM_C;
	return rseed;
}

// 3.1 Add a new function definition and declaration (`random_float`) returning a random `float`. 
// This should be a value cast from the `random_uint` function.
float random_float() {
	return (float) random_uint();
}
