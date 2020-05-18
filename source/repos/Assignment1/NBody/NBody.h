// Header guards prevent the contents of the header from being defined multiple times where there are circular dependencies
#ifndef __NBODY_HEADER__
#define __NBODY_HEADER__

/* Our gravitational constant differs from https://en.wikipedia.org/wiki/Gravitational_constant and is closer to 
https://en.wikipedia.org/wiki/Acceleration_due_to_gravity in value, but our distances are normalised anyway */
#define G      9.8f  // Gravitational constant
#define dt     0.01f // Time step
#define eps_sq 4.0f  // Softening parameter to help with numerical instability

struct nbody{ // Changed to structure of arrays to facilitate coalesced memory access
	float *x, *y, *vx, *vy, *m;
};

typedef enum MODE { CPU, OPENMP, CUDA } MODE;
typedef struct nbody nbody;

#endif	// __NBODY_HEADER__
