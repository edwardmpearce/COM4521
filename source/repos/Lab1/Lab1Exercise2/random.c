/* 2.1 Let us start by separating the random function definitions into a header and separate source module.
Create a new file in the project called `random.c`. 
Move the `init_random` and `random_ushort` function definitions into the new source module. 
Move the inclusion of `stdlib.h` to the new source module and include `random.h` in `random.c`. 
The `random.h` file should now only contain the seed. Build the application. */

#include <stdlib.h>
#include "random.h"

// 2.4 Create a 32 bit (4 byte) unsigned global integer variable called `rseed` in `random.c`. 
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

/* 2.5 Create a function declaration (in `random.h`) and definition (in `random.c`) for a function
`random_uint` returning an `unsigned int`. Implement a linear generator where `x` can use the variable `rseed`
(with an initial value `x0 = RAND_SEED` as per exercise 2.4), and `A` and `C` are your parameters. */
unsigned int random_uint() {
	rseed = RANDOM_A * rseed + RANDOM_C;
	return rseed;
}
