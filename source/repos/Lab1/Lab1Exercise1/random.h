#include <stdlib.h>

#define RAND_SEED 123

void init_random(){
	srand(RAND_SEED);
}

unsigned short random_ushort(){
	// 1.7 The `random_ushort` function contains an implicit cast from `int` to `unsigned short`.
	// Modify this so that it uses an explicit cast. This won't change the program but is good practice.
	return (unsigned short) rand();
}
