// Header guards prevent the contents of the header from being defined multiple times where there are circular dependencies
#ifndef __NBODY_HEADER__
#define __NBODY_HEADER__

/* Our gravitational constant differs from https://en.wikipedia.org/wiki/Gravitational_constant and in assignment 1 was 
closer to https://en.wikipedia.org/wiki/Acceleration_due_to_gravity in value (previously set as 9.8f), but our distances 
are normalised anyway, so now we use a normalised value of G (1.0f) to avoid issues with numeric precision */
#define G      1.0f  // Gravitational constant
#define dt     0.01f // Time step
#define eps_sq 4.0f  // Softening parameter to help with numerical instability

struct nbody{ // Requires passing as an Array of Structures (AoS)
	float x, y, vx, vy, m;
};

struct nbody_soa { // Structure of Arrays (SoA) to facilitate coalesced memory access
	float *x, *y, *vx, *vy, *m;
};

typedef enum MODE { CPU, OPENMP, CUDA } MODE;
typedef struct nbody nbody;
typedef struct nbody_soa nbody_soa;

#endif	//__NBODY_HEADER__
